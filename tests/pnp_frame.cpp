#include <fmt/core.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/armor_track/armor_target.hpp"
#include "tasks/auto_aim/armor_track/armor_tracker.hpp"
#include "tasks/auto_aim/armor_control/very_aimer.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/base/common.hpp"
#include "utils/utils.hpp"

using namespace awakening;

namespace {
constexpr double kRadToDeg = 57.29577951308232;

std::string aimer_type(const YAML::Node& config) {
    if (config["very_aimer"] && config["very_aimer"]["type"]) {
        return config["very_aimer"]["type"].as<std::string>();
    }
    return "unknown";
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fmt::print("usage: {} <config.yaml> <image>\n", argv[0]);
        return 1;
    }
    const auto config = YAML::LoadFile(argv[1]);
    cv::Mat img = cv::imread(argv[2]);
    if (img.empty()) {
        fmt::print("cannot read image {}\n", argv[2]);
        return 1;
    }

    awakening::CameraInfo camera_info;
    camera_info.load(config["camera"]["camera_info"]);

    awakening::auto_aim::ArmorDetector detector(config["armor_detector"]);
    awakening::CommonFrame frame;
    frame.img_frame.src_img = img;
    frame.img_frame.format = awakening::PixelFormat::BGR;
    frame.img_frame.timestamp = awakening::Clock::now();
    frame.id = 0;
    frame.frame_id = 0;

    const cv::Rect net_focus(0, 0, img.cols, img.rows);
    auto [lights, armors] = detector.detect(frame, net_focus);

    awakening::ISO3 camera_cv_in_gimbal =
        awakening::utils::load_isometry3(config["tf"]["camera_in_gimbal"]);
    camera_cv_in_gimbal.linear() = camera_cv_in_gimbal.linear() * awakening::R_CV2PHYSICS;

    fmt::print("WUST detected armor count = {}\n", armors.size());
    for (auto& armor : armors) {
        if (!awakening::auto_aim::ArmorTarget::armor_pnp(armor, camera_info, camera_cv_in_gimbal)) {
            fmt::print("  pnp failed class={}\n", static_cast<int>(armor.number));
            continue;
        }
        const auto t = armor.pose.translation();
        const auto rpy = awakening::utils::matrix2rpy<double>(
            armor.pose.linear(), awakening::utils::RPYOrder::ZYX);
        const auto& pts = armor.key_points.landmarks();
        if (pts.empty()) {
            fmt::print("  no key points class={}\n", static_cast<int>(armor.number));
            continue;
        }
        const std::vector<cv::Point3f> obj_points =
            awakening::auto_aim::getArmorKeyPoints3D<cv::Point3f>(armor.number);
        const awakening::ISO3 pose_in_camera = camera_cv_in_gimbal.inverse() * armor.pose;
        cv::Mat R_cv;
        const Eigen::Matrix3d R_pose = pose_in_camera.linear();
        cv::eigen2cv(R_pose, R_cv);
        cv::Mat rvec;
        cv::Rodrigues(R_cv, rvec);
        const Eigen::Vector3d t_cam = pose_in_camera.translation();
        const cv::Vec3d tvec(t_cam.x(), t_cam.y(), t_cam.z());
        std::vector<cv::Point2f> proj_points;
        cv::projectPoints(
            obj_points,
            rvec,
            tvec,
            camera_info.camera_matrix,
            camera_info.distortion_coefficients,
            proj_points
        );
        double reproj_rms = 0.0;
        for (size_t i = 0; i < pts.size(); ++i) {
            const auto e = cv::norm(proj_points[i] - pts[i]);
            reproj_rms += e * e;
        }
        reproj_rms = std::sqrt(reproj_rms / pts.size());

        double min_x = pts.front().x;
        double max_x = pts.front().x;
        double min_y = pts.front().y;
        double max_y = pts.front().y;
        for (const auto& p : pts) {
            min_x = std::min(min_x, static_cast<double>(p.x));
            max_x = std::max(max_x, static_cast<double>(p.x));
            min_y = std::min(min_y, static_cast<double>(p.y));
            max_y = std::max(max_y, static_cast<double>(p.y));
        }
        const double width_px = max_x - min_x;
        const double height_px = max_y - min_y;

        fmt::print("  reproj_rms_px={:.4f}\n", reproj_rms);
        fmt::print("  points:");
        for (const auto& p : pts) fmt::print(" ({:.2f},{:.2f})", p.x, p.y);
        fmt::print("\n");
        fmt::print(
            "  WUST_ARMOR class={} width_px={:.4f} height_px={:.4f} "
            "xyz_gimbal=[{:.6f},{:.6f},{:.6f}] distance_m={:.6f} "
            "reproj_rms_px={:.4f} yaw_deg={:.4f} pitch_deg={:.4f} roll_deg={:.4f}\n",
            static_cast<int>(armor.number), width_px, height_px,
            t.x(), t.y(), t.z(), t.norm(), reproj_rms,
            rpy[2] * kRadToDeg, rpy[1] * kRadToDeg, rpy[0] * kRadToDeg);
    }

    // 把这一帧检测结果送进与 standard 相同的 tracker/aimer 链路，
    // 输出真正会进 GimbalCmd 的 yaw/pitch，而不是只看单块装甲板的 PnP 角。
    auto_aim::Armors detected;
    detected.timestamp = frame.img_frame.timestamp;
    detected.frame_id = frame.frame_id;
    detected.lights = std::move(lights);
    detected.armors = std::move(armors);

    auto_aim::ArmorTracker tracker(config["armor_tracker"]);
    const auto target = tracker.track(detected, camera_info, camera_cv_in_gimbal, frame.frame_id);
    auto_aim::VeryAimer aimer(config["very_aimer"]);
    const auto shoot_in_gimbal_odom =
        awakening::utils::load_isometry3(config["tf"]["shoot_in_gimbal"]);
    const awakening::ISO3 gimbal_in_gimbal_odom = awakening::ISO3::Identity();
    const double bullet_speed = config["bullet_speed"] ? config["bullet_speed"].as<double>() : 0.0;

    GimbalCmd cmd;
    if (target.target_number != auto_aim::ArmorClass::UNKNOWN) {
        cmd = aimer.very_aim(
            target,
            bullet_speed,
            auto_aim::AutoAimFsm::AIM_SINGLE_ARMOR,
            shoot_in_gimbal_odom,
            gimbal_in_gimbal_odom
        );
    }

    double cmd_distance_m = -1.0;
    if (cmd.appear && cmd.is_valid()) {
        cmd_distance_m =
            (cmd.aim_point.pose.translation() - shoot_in_gimbal_odom.translation()).norm();
    }

    fmt::print(
        "WUST_CMD image={} armor_count={} target_number={} this_id={} match_count={} "
        "aim_id={} yaw_deg={:.6f} pitch_deg={:.6f} "
        "send_yaw_deg={:.6f} send_pitch_deg={:.6f} send_appear={} send_fire={} "
        "target_yaw_deg={:.6f} "
        "target_pitch_deg={:.6f} v_yaw_deg_s={:.6f} a_yaw_deg_s2={:.6f} "
        "v_pitch_deg_s={:.6f} a_pitch_deg_s2={:.6f} appear={} fire={} "
        "distance_m={:.6f} fly_time_s={:.6f} type={}\n",
        argv[2],
        static_cast<int>(detected.armors.size()),
        static_cast<int>(target.target_number),
        target.this_id,
        target.last_match_count,
        cmd.select_id,
        cmd.yaw,
        cmd.pitch,
        cmd.yaw,
        -cmd.pitch,
        cmd.appear ? 1 : -1,
        cmd.fire_advice ? 1 : 0,
        cmd.target_yaw,
        cmd.target_pitch,
        cmd.v_yaw,
        cmd.a_yaw,
        cmd.v_pitch,
        cmd.a_pitch,
        cmd.appear ? 1 : 0,
        cmd.fire_advice ? 1 : 0,
        cmd_distance_m,
        cmd.fly_time,
        aimer_type(config)
    );
    return 0;
}
