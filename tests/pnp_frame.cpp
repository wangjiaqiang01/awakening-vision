
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/armor_track/armor_target.hpp"
#include "tasks/base/common.hpp"
#include "utils/utils.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        fmt::print("usage: {} <config.yaml> <image>\n", argv[0]);
        return 1;
    }
    auto config = YAML::LoadFile(argv[1]);
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

    cv::Rect net_focus(0, 0, img.cols, img.rows);
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
        std::vector<cv::Point3f> obj_points =
            awakening::auto_aim::getArmorKeyPoints3D<cv::Point3f>(armor.number);
        awakening::ISO3 pose_in_camera = camera_cv_in_gimbal.inverse() * armor.pose;
        cv::Mat R_cv;
        Eigen::Matrix3d R_pose = pose_in_camera.linear();
        cv::eigen2cv(R_pose, R_cv);
        cv::Mat rvec;
        cv::Rodrigues(R_cv, rvec);
        const Eigen::Vector3d t_cam = pose_in_camera.translation();
        cv::Vec3d tvec(t_cam.x(), t_cam.y(), t_cam.z());
        std::vector<cv::Point2f> proj_points;
        cv::projectPoints(obj_points, rvec, tvec,
                          camera_info.camera_matrix,
                          camera_info.distortion_coefficients,
                          proj_points);
        double reproj_rms = 0.0;
        for (size_t i = 0; i < pts.size(); ++i) {
            reproj_rms += cv::norm(proj_points[i] - pts[i]) * cv::norm(proj_points[i] - pts[i]);
        }
        reproj_rms = std::sqrt(reproj_rms / pts.size());
        fmt::print("  reproj_rms_px={:.4f}\n", reproj_rms);
        fmt::print("  points:");
        for (const auto& p : pts) fmt::print(" ({:.2f},{:.2f})", p.x, p.y);
        fmt::print("\n");
        fmt::print(
            "  class={} xyz_gimbal=[{:.6f}, {:.6f}, {:.6f}] "
            "yaw_deg={:.4f} pitch_deg={:.4f} roll_deg={:.4f}\n",
            static_cast<int>(armor.number), t.x(), t.y(), t.z(),
            rpy[2] * 57.29577951308232, rpy[1] * 57.29577951308232,
            rpy[0] * 57.29577951308232);
    }
    return 0;
}
