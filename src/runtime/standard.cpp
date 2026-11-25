#include "angles.h"
#include "ascii_banner.hpp"
#include "daedalus_interface/shm_layout.hpp"
#include "tasks/auto_aim/armor_track/motion_model.hpp"
#include "tasks/auto_buff/rune_control/rune_aimer.hpp"
#include "tasks/auto_buff/rune_detect/rune_detector.hpp"
#include "tasks/auto_buff/rune_track/rune_target.hpp"
#include "tasks/auto_buff/rune_track/rune_tracker.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/base/ballistic_trajectory.hpp"
#include "tasks/base/wheel_odometry.hpp"
#include "utils/drivers/camera_factory.hpp"
#include "utils/io/video_save.hpp"
#include "utils/io/serial_trace.hpp"
#include "utils/io/timing_csv.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#ifdef USE_RERUN
    #include "_rerun/recorder.hpp"
    #include "_rerun/tf.hpp"
#endif
#ifdef USE_ROS2
    #include "_rcl/node.hpp"
    #include "_rcl/tf.hpp"
    #include "_rcl/visual/armor.hpp"
    #include "_rcl/visual/armor_target.hpp"
    #include "sensor_msgs/msg/camera_info.hpp"
    #include "sensor_msgs/msg/image.hpp"
    #include <rclcpp/qos.hpp>
#endif
#include "backward-cpp/backward.hpp"
#ifdef USE_Daedalus
    #include "daedalus_interface/shm_client.hpp"
#endif
#include "param_deliver.h"
#include "runtime/config.hpp"
#include "runtime/frame_plan_loop.hpp"
#include "tasks/auto_aim/armor_control/very_aimer.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/armor_track/armor_target.hpp"
#include "tasks/auto_aim/armor_track/armor_tracker.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/web.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/command_protocol.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/drivers/telemetry_protocol.hpp"
#include "utils/drivers/vision_command_guard.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_tf.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
namespace backward {
static backward::SignalHandling sh;
}
using namespace awakening;
enum class Mode : int { AutoAim = 0, AutoBuff = 1 };
Mode str_to_Mode(std::string str) {
    str = utils::to_upper(str);
    if (str == "AUTO_AIM") {
        return Mode::AutoAim;
    } else if (str == "AUTO_BUFF") {
        return Mode::AutoBuff;
    } else {
        throw std::invalid_argument("Invalid mode string");
    }
}
enum class SimpleFrame : int { ODOM, GIMBAL_ODOM, GIMBAL, CAMERA, CAMERA_CV, SHOOT, N };

using SimpleRobotTF = utils::tf::RobotTF<SimpleFrame, static_cast<size_t>(SimpleFrame::N), false>;

std::string SimpleFrame_to_str(int frame) {
    constexpr const char* details[] = { "odom",   "gimbal_odom", "gimbal",
                                        "camera", "camera_cv",   "shoot" };
    return std::string(details[frame]);
}
std::string SimpleFrame_to_str(SimpleFrame frame) {
    return SimpleFrame_to_str(std::to_underlying(frame));
}
struct CameraTag {};
struct SerialTag {};
struct DetectTag {};
struct FrameTag {};

using CameraIO = IOPair<CameraTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
struct Detection {
    std::optional<std::vector<auto_aim::Armors>> armors;
    std::optional<std::vector<auto_buff::RuneDetection>> rune;
    std::optional<TimePoint> detect_start;
};
using DetIo = IOPair<DetectTag, Detection>;
struct LogCtx {
    int camera_count = 0;
    int detect_count = 0;
    int track_count = 0;
    int solve_count = 0;
    int serial_count = 0;
    int found_count = 0;
    int ekf_count = 0;
    double latency_ms_total = 0.0;
    void reset() {
        *this = {};
    }
};
struct AutoExposureCfg {
    double target_brightness = 0.0;
    double step_gain = 0.0;
    double decay_step = 0.0;
    double tolerance = 0.0;
    double exposure_min = 0.0;
    double exposure_max = 0.0;
    double control_interval_ms = 0.0;

    explicit AutoExposureCfg(const YAML::Node& c):
        target_brightness(c["target_brightness"].as<double>()),
        step_gain(c["step_gain"].as<double>()),
        decay_step(c["decay_step"].as<double>()),
        tolerance(c["tolerance"].as<double>()),
        exposure_min(c["exposure_min"].as<double>()),
        exposure_max(c["exposure_max"].as<double>()),
        control_interval_ms(c["control_interval_ms"].as<double>()) {}
};
bool is_web_running() {
    static std::atomic<bool> cached { true };
    utils::dt_once(
        [&]() {
            const int ret = std::system("pgrep -x wust_vision_web > /dev/null 2>&1");
            cached = (ret == 0);
        },
        std::chrono::duration<double>(1.0)
    );
    bool running = cached.load();
#ifdef USE_RERUN
    running = running || rerun_visual::Recorder::instance().enabled();
#endif
    return running;
}
static constexpr auto RECORD_FOLDER_PATH_ARR = utils::concat(ROOT_DIR, "/record/auto_aim");
static constexpr std::string_view RECORD_FOLDER_PATH(RECORD_FOLDER_PATH_ARR.data());

int main(int argc, char** argv) {
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);

    bool debug = false;
    std::string config_path;
    std::string robot_name;
    auto first_arg = utils::get_arg(1, argc, argv);
    if (first_arg) {
        robot_name = first_arg.value();
        config_path = get_robot_config_path(robot_name).value_or(robot_name);
    } else {
        return 1;
    }
    auto second_arg = utils::get_arg(2, argc, argv);
    if (second_arg) {
        debug = second_arg.value() == "true";
    }
    auto config = YAML::LoadFile(config_path);
    Scheduler s;
    std::atomic<Mode> mode = str_to_Mode(config["mode"].as<std::string>());
    std::atomic<EnemyColor> enemy_color =
        enemy_color_from_string(config["enemy_color"].as<std::string>());
    std::atomic<double> bullet_speed = config["bullet_speed"].as<double>();
#ifdef USE_ROS2
    rcl::RclcppNode rcl_node("awakening");
    rcl::TF rcl_tf(rcl_node);
#endif
    bool use_daedalus = false;
#ifdef USE_Daedalus
    std::unique_ptr<talos::ipc::ShmClient> daedalus_shm_client;
    if (config["use_sim"].as<bool>()) {
        auto client = talos::ipc::ShmClient::connect();
        if (!client) {
            AWAKENING_ERROR("Failed to connect to talos::ipc::ShmClient");
            return 1;
        } else {
            use_daedalus = true;
            daedalus_shm_client = std::make_unique<talos::ipc::ShmClient>(std::move(*client));
        }
    }
#endif
    std::unique_ptr<SerialDriver> serial;
    if (!use_daedalus && config["serial"]["enable"].as<bool>()) {
        serial = std::make_unique<SerialDriver>(config["serial"], s);
    }
    VisionCommandGuard::Params command_guard_params;
    if (const auto guard = config["serial"]["command_guard"]) {
        command_guard_params.acquire_ms =
            guard["acquire_ms"].as<int>(command_guard_params.acquire_ms);
        command_guard_params.fire_settle_ms =
            guard["fire_settle_ms"].as<int>(command_guard_params.fire_settle_ms);
        command_guard_params.min_distance_m =
            guard["min_distance_m"].as<double>(command_guard_params.min_distance_m);
        command_guard_params.max_distance_m =
            guard["max_distance_m"].as<double>(command_guard_params.max_distance_m);
        command_guard_params.max_yaw_jump_deg =
            guard["max_yaw_jump_deg"].as<double>(command_guard_params.max_yaw_jump_deg);
        command_guard_params.max_pitch_jump_deg =
            guard["max_pitch_jump_deg"].as<double>(command_guard_params.max_pitch_jump_deg);
        command_guard_params.max_yaw_rate_deg_s =
            guard["max_yaw_rate_deg_s"].as<double>(command_guard_params.max_yaw_rate_deg_s);
        command_guard_params.max_pitch_rate_deg_s =
            guard["max_pitch_rate_deg_s"].as<double>(command_guard_params.max_pitch_rate_deg_s);
        command_guard_params.filter_time_constant_s = guard["filter_time_constant_s"].as<double>(
            command_guard_params.filter_time_constant_s
        );
        command_guard_params.fire_alignment_tolerance_deg =
            guard["fire_alignment_tolerance_deg"].as<double>(
                command_guard_params.fire_alignment_tolerance_deg
            );
        command_guard_params.reset_on_target_change =
            guard["reset_on_target_change"].as<bool>(
                command_guard_params.reset_on_target_change
            );
    }
    VisionCommandGuard command_guard(command_guard_params);
    const std::string command_guard_profile =
        config["serial"]["command_guard"]["profile"].as<std::string>("custom");
    const bool command_guard_bypass = command_guard_profile == "raw";

    auto camera_config = config["camera"];
    std::unique_ptr<Camera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });
    if (!use_daedalus) {
        camera = create_camera(camera_config, s, "daheng");
        camera->init();
        if (!camera->is_running()) {
            return 0;
        }
    }
    std::unique_ptr<VideoSaver> video_saver;
    std::shared_ptr<SerialTrace> serial_trace;
    std::atomic<uint64_t> video_frame_index {0};
    if (config["record"]["enable"].as<bool>()) {
        const std::string video_path =
            VideoSaver::generate_record_filename(RECORD_FOLDER_PATH.data());
        video_saver =
            std::make_unique<VideoSaver>(video_path, VideoSaver::Mode::NonBlocking);
        std::string trace_path = video_path;
        if (trace_path.size() >= 4) {
            trace_path = trace_path.substr(0, trace_path.size() - 4) + ".csv";
        }
        serial_trace = std::make_shared<SerialTrace>(trace_path, video_path);
        AWAKENING_INFO("record video={} serial_trace={}", video_path, trace_path);
    }
    if (serial && serial_trace) {
        serial->set_trace_callbacks(
            [serial_trace](
                const TongjiTelemetry& telemetry,
                const uint8_t* raw,
                size_t size,
                TelemetryStream::Clock::time_point
            ) { serial_trace->log_rx(telemetry, raw, size); },
            [serial_trace](
                const std::vector<uint8_t>& command,
                TelemetryStream::Clock::time_point
            ) { serial_trace->log_tx(command); }
        );
    }
    CameraInfo camera_info;
    camera_info.load(camera_config["camera_info"]);
    auto_aim::ArmorDetector armor_detector(config["armor_detector"]);
    auto_aim::ArmorTracker armor_tracker(config["armor_tracker"]);
    auto_aim::AutoAimFsmController auto_aim_fsm_controller(config["auto_aim_fsm"]);
    auto_aim::VeryAimer very_aimer(config["very_aimer"]);
    auto_buff::RuneDetector rune_detector(config["rune_detector"]);
    auto_buff::RuneTracker rune_tracker(config["rune_tracker"]);
    auto_buff::RuneAimer rune_aimer(config["rune_aimer"]);
    utils::OrderedQueue<auto_aim::Armors> armors_queue;
    utils::OrderedQueue<auto_buff::RuneDetection> rune_detections_queue;
    utils::SWMR<auto_aim::ArmorTarget> armor_target;
    utils::SWMR<auto_buff::RuneTarget> rune_target;
    BulletPickUp bullet_pick_up(config["bullet_pick_up"]);
    LogCtx log_ctx;
    std::optional<VisionDebugCtx> dbg;
    std::pair<double, double> operator_offset = std::make_pair(0, 0);
    if (debug) {
        dbg.emplace();
        dbg->camera_info_ = camera_info;
    }
    WheelOdometry wheel_odometry(config["wheel_odometry"], Clock::now());
    auto tf = SimpleRobotTF::create();
    {
        for (auto [from, to]: std::array {
                 std::pair { SimpleFrame::ODOM, SimpleFrame::GIMBAL_ODOM },
                 std::pair { SimpleFrame::GIMBAL_ODOM, SimpleFrame::GIMBAL },
                 std::pair { SimpleFrame::GIMBAL, SimpleFrame::CAMERA },
                 std::pair { SimpleFrame::GIMBAL, SimpleFrame::SHOOT },
                 std::pair { SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV },
             })
        {
            tf->add_edge(from, to);
        }
        ISO3 cv_in_camera = ISO3::Identity();
        cv_in_camera.linear() = R_CV2PHYSICS;
        tf->push(SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV, Clock::now(), cv_in_camera);
        tf->push(
            SimpleFrame::GIMBAL,
            SimpleFrame::CAMERA,
            Clock::now(),
            utils::load_isometry3(config["tf"]["camera_in_gimbal"])
        );
        tf->push(
            SimpleFrame::GIMBAL,
            SimpleFrame::SHOOT,
            Clock::now(),
            utils::load_isometry3(config["tf"]["shoot_in_gimbal"])
        );
    }
    auto serial_send_to_image_microseconds = config["serial_send_to_image_microseconds"].as<int>();
    // 同济状态回传的消息 ID，必须与下位机 InitParam() 的注册一致（见 config/dong.yaml）。
    const int telemetry_gimbal_id = config["serial"]["telemetry_gimbal_id"].as<int>(-1);
    const int telemetry_shoot_id = config["serial"]["telemetry_shoot_id"].as<int>(-1);
    // 视觉 pitch（抬头为正）到固件 IMU Pitch 的符号，只在发送边界使用。
    // 哨兵（IMU 安装 .Pitch = 0）实机验证为 -1；IMU 倒装的车（如六轮步兵 .Pitch = 180）需要 +1。
    // 首次上车先小角度点动：云台往反方向走就说明这一项要取反（改 config/dong.yaml，无需重编译）。
    const double pitch_sign = config["serial"]["pitch_sign"].as<double>(-1.0);
    // ---- debug timing CSV: 运行中实时落盘，不改变控制路径 ----
    const bool debug_timing_enable =
        config["debug_timing"] && config["debug_timing"]["enable"].as<bool>(false);
    const int debug_timing_every_n = std::max(
        1,
        config["debug_timing"] && config["debug_timing"]["every_n"]
            ? config["debug_timing"]["every_n"].as<int>(10)
            : 10
    );
    std::unique_ptr<utils::TimingCsvLogger> timing_logger;
    if (debug_timing_enable) {
        std::string timing_path = config["debug_timing"] && config["debug_timing"]["path"]
            ? config["debug_timing"]["path"].as<std::string>()
            : "log/timing.csv";
        if (!timing_path.empty() && timing_path.front() != '/') {
            timing_path = std::string(ROOT_DIR) + "/" + timing_path;
        }
        timing_logger = std::make_unique<utils::TimingCsvLogger>(timing_path);
        if (timing_logger->is_open()) {
            AWAKENING_INFO(
                "timing csv: {} every_n={}", timing_logger->path(), debug_timing_every_n
            );
        } else {
            AWAKENING_WARN("timing csv open failed: {}", timing_path);
            timing_logger.reset();
        }
    }
    const double camera_fps = camera_config["hik_camera"]
        && camera_config["hik_camera"]["acquisition_frame_rate"]
        ? camera_config["hik_camera"]["acquisition_frame_rate"].as<double>(-1.0)
        : -1.0;
    const double camera_exposure_cfg_us = camera_config["hik_camera"]
        && camera_config["hik_camera"]["exposure_time"]
        ? camera_config["hik_camera"]["exposure_time"].as<double>(-1.0)
        : -1.0;

    if (serial) {
        AWAKENING_INFO("serial pitch_sign = {} (按实车云台方向在 config/dong.yaml 调整)", pitch_sign);
        if (telemetry_gimbal_id <= 0 || telemetry_shoot_id <= 0) {
            AWAKENING_WARN(
                "serial.telemetry_gimbal_id/shoot_id 未配置：不再解析旧协议回传，"
                "本车拿不到云台姿态（仅发送控制帧）"
            );
        }
    }
#ifdef USE_Daedalus
    if (daedalus_shm_client) {
        auto daedalus_imgs = s.register_source<CameraIO>("daedalus_img");

        auto push_pose = [&](talos::ipc::PoseIndex idx, SimpleFrame parent, SimpleFrame child) {
            if (auto pose = daedalus_shm_client->recv_pose(idx)) {
                ISO3 T = ISO3::Identity();
                T.translation() << pose->x, pose->y, pose->z;
                T.linear() =
                    Quaternion { pose->qw, pose->qx, pose->qy, pose->qz }.toRotationMatrix();

                tf->push(parent, child, TimePoint(std::chrono::nanoseconds(pose->timestamp_ns)), T);
            }
        };

        s.add_rate_source<>("daedalus_tick", 300.0, [&]() {
            static bool has_camera_info = false;

            if (!has_camera_info) {
                auto info = daedalus_shm_client->camera_info();
                has_camera_info = true;
                camera_info.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
                camera_info.camera_matrix.at<double>(0, 0) = info.fx;
                camera_info.camera_matrix.at<double>(1, 1) = info.fy;
                camera_info.camera_matrix.at<double>(0, 2) = info.cx;
                camera_info.camera_matrix.at<double>(1, 2) = info.cy;
                camera_info.distortion_coefficients = cv::Mat(1, 5, CV_64F);
                std::memcpy(
                    camera_info.distortion_coefficients.ptr<double>(),
                    info.distortion,
                    5 * sizeof(double)
                );
            }

            if (auto frame = daedalus_shm_client->recv_image()) {
                ImageFrame img_frame {
                    .src_img = std::move(frame->image),
                    .format = PixelFormat::RGB,
                    .timestamp = TimePoint(std::chrono::nanoseconds(frame->timestamp_ns)),
                };
                s.runtime_push_source<CameraIO>(
                    daedalus_imgs,
                    [f = std::move(img_frame)]() mutable {
                        return std::make_tuple(std::optional<CameraIO::second_type>(std::move(f)));
                    }
                );
            }
            if (auto ground_truth = daedalus_shm_client->recv_ground_truth()) {
                if (dbg) {
                    dbg->ground_truth.set(ground_truth.value());
                }
            }

            push_pose(talos::ipc::PoseIndex::POSE_CAMERA, SimpleFrame::GIMBAL, SimpleFrame::CAMERA);
            push_pose(talos::ipc::PoseIndex::POSE_MUZZLE, SimpleFrame::GIMBAL, SimpleFrame::SHOOT);
            push_pose(
                talos::ipc::PoseIndex::POSE_GIMBAL,
                SimpleFrame::GIMBAL_ODOM,
                SimpleFrame::GIMBAL
            );
        });
    }
#endif
    if (video_saver) {
        s.register_task<CameraIO>("save_video", [&](CameraIO::second_type&& f) {
            if (!f.src_img.empty()) {
                const auto frame_index = video_frame_index.load();
                cv::Mat annotated = f.src_img.clone();
                std::ostringstream label;
                label << SerialTrace::wall_now() << "  #" << frame_index;
                cv::putText(
                    annotated,
                    label.str(),
                    cv::Point(12, 36),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA
                );
                if (video_saver->write_frame(annotated)) {
                    // 只有真正进入 VideoWriter 队列的帧才记录，视频帧号从 0 连续计数。
                    if (serial_trace) {
                        serial_trace->log_video(f.timestamp, frame_index);
                    }
                    video_frame_index.fetch_add(1);
                }
            }
            return std::make_tuple(std::optional<CameraIO::second_type>(std::nullopt));
        });
    }
    s.register_task<CameraIO, CommonFrameIo>("push_common_frame", [&](CameraIO::second_type&& f) {
        static int current_id = 0;
        if (f.src_img.empty()) {
            return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::nullopt));
        }
        log_ctx.camera_count++;
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = std::to_underlying(SimpleFrame::CAMERA_CV),
        };

        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });

    const bool direct_telemetry_tf =
        serial && telemetry_gimbal_id > 0 && telemetry_shoot_id > 0;
    const auto serial_tf_offset = direct_telemetry_tf
        ? std::chrono::microseconds(serial_send_to_image_microseconds)
        : std::chrono::microseconds(0);
    auto serial_rx_counter = std::make_shared<std::atomic<int>>(0);
    if (direct_telemetry_tf) {
        // 直接在 SerialDriver 单 IO 线程入 TF：按单次 USB 读取批次取最新帧并打接收时刻，
        // 避免经由 Scheduler 16 线程 TBB 队列乱序与微秒级假间隔污染 TimePoseBuffer。
        serial->set_telemetry_callback(
            [tf, serial_tf_offset, serial_rx_counter](
                const TongjiTelemetry& state,
                std::chrono::system_clock::time_point rx_wall_time,
                std::size_t batch_frames
            ) {
                serial_rx_counter->fetch_add(
                    static_cast<int>(batch_frames), std::memory_order_relaxed
                );
                const auto packet_time = rx_wall_time - serial_tf_offset;
                ISO3 gimbal_in_gimbal_odom = ISO3::Identity();
                gimbal_in_gimbal_odom.linear() = utils::rpy2matrix(
                    Vec3(
                        angles::from_degrees(state.roll),
                        angles::from_degrees(state.pitch),
                        angles::from_degrees(state.yaw)
                    ),
                    utils::RPYOrder::ZYX
                );
                tf->push(
                    SimpleFrame::GIMBAL_ODOM,
                    SimpleFrame::GIMBAL,
                    packet_time,
                    gimbal_in_gimbal_odom
                );
            }
        );
    }

    if (serial) {
        s.register_task<SerialIO>("receive_serial", [direct_telemetry_tf](SerialIO::second_type&&) {
            // 同济协议已由 SerialDriver IO 线程直通写入 TF；
            // 未配置回传 ID 的旧配置不再解析旧协议，这里仅保持 Scheduler 拓扑连接。
            if (direct_telemetry_tf) return;
        });
    }
    if (camera) {
        s.register_task<CommonFrameIo>("auto_exposure", [&](CommonFrameIo::second_type&& frame) {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            static std::optional<AutoExposureCfg> auto_exposure_cfg =
                config["auto_exposure"]["enable"].as<bool>()
                ? std::optional<AutoExposureCfg>(std::in_place, config["auto_exposure"])
                : std::nullopt;
            if (auto_exposure_cfg) { // 平均亮度pid
                auto& cfg = auto_exposure_cfg.value();
                utils::dt_once(
                    [&]() {
                        cv::Mat gray;
                        cv::cvtColor(frame.img_frame.src_img, gray, cv::COLOR_BGR2GRAY);
                        double exposure_time = camera->get_ExposureTime();
                        static double last_exposure_time = 0.0;
                        const double diff = cv::mean(gray)[0] - cfg.target_brightness;
                        if (std::fabs(diff) > cfg.tolerance && exposure_time > 0.0) {
                            exposure_time -= diff * cfg.step_gain;
                        } else {
                            exposure_time -= cfg.decay_step;
                        }
                        exposure_time =
                            std::clamp(exposure_time, cfg.exposure_min, cfg.exposure_max);
                        if (std::abs(exposure_time - last_exposure_time) > 10) {
                            camera->set_ExposureTime(exposure_time);
                            last_exposure_time = exposure_time;
                        }
                    },
                    std::chrono::milliseconds((int)cfg.control_interval_ms)
                );
            }
        });
    }

    s.register_task<CommonFrameIo, DetIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        Detection r;
        cv::Rect net_focus =
            cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows);
        if (mode == Mode::AutoAim) {
            static std::unique_ptr<std::counting_semaphore<>> detector_sem;
            if (!detector_sem) {
                detector_sem = std::make_unique<std::counting_semaphore<>>(
                    config["armor_detector"]["max_infer_num"].as<int>()
                );
            }
            std::optional<cv::Rect> detect_light = std::nullopt;
            auto target = armor_target.read();
            if (target.need_focus()) {
                auto camera_cv_in_old = tf->pose_a_in_b(
                    SimpleFrame(frame.frame_id),
                    SimpleFrame(target.get_target_state().frame_id),
                    frame.img_frame.timestamp
                );
                net_focus = target.get_net_focus_roi(
                    frame.img_frame.timestamp,
                    camera_cv_in_old,
                    camera_info,
                    frame.img_frame.src_img.size(),
                    armor_detector.get_net_wh_ratio()
                );
                if (target.need_detect_lights()) {
                    detect_light = target.expanded( // 送给传统越小越好
                        frame.img_frame.timestamp,
                        camera_cv_in_old,
                        camera_info,
                        frame.img_frame.src_img.size()
                    );
                    detect_light = utils::expand_and_clip_rect(
                        detect_light.value(),
                        1.6,
                        frame.img_frame.src_img.size()
                    );
                }
            }

            auto_aim::Armors armors { .timestamp = frame.img_frame.timestamp,
                                      .id = frame.id,
                                      .frame_id = frame.frame_id };
            {
                bool got = detector_sem->try_acquire();
                utils::SemaphoreGuard guard(*detector_sem, got); //并发控制
                if (got) {
                    auto start = Clock::now();
                    r.detect_start = start;
                    auto [ls, as] = armor_detector.detect(frame, net_focus, detect_light);
                    armors.armors = as;
                    armors.lights = ls;
                    log_ctx.detect_count++;
                }
            }
            armors_queue.enqueue(armors);
            auto batch_armors = armors_queue.dequeue_batch(); // 根据id有序输出
            r.armors = batch_armors;
        } else {
            static std::unique_ptr<std::counting_semaphore<>> detector_sem;
            if (!detector_sem) {
                detector_sem = std::make_unique<std::counting_semaphore<>>(
                    config["rune_detector"]["max_infer_num"].as<int>()
                );
            }
            auto target = rune_target.read();
            if (target.need_focus()) {
                auto camera_cv_in_old = tf->pose_a_in_b(
                    SimpleFrame(frame.frame_id),
                    SimpleFrame(target.get_target_state().frame_id),
                    frame.img_frame.timestamp
                );
                net_focus = target.get_net_focus_roi(
                    frame.img_frame.timestamp,
                    camera_cv_in_old,
                    camera_info,
                    frame.img_frame.src_img.size(),
                    armor_detector.get_net_wh_ratio()
                );
            }
            auto_buff::RuneDetection rune_detection;
            {
                bool got = detector_sem->try_acquire();
                utils::SemaphoreGuard guard(*detector_sem, got); //并发控制
                if (got) {
                    auto start = Clock::now();
                    r.detect_start = start;
                    rune_detection = rune_detector.detect(frame, net_focus, enemy_color);
                    log_ctx.detect_count++;
                }
            }
            rune_detection.frame_id = frame.frame_id;
            rune_detection.id = frame.id;
            rune_detection.timestamp = frame.img_frame.timestamp;
            rune_detections_queue.enqueue(rune_detection);
            auto batch_runes = rune_detections_queue.dequeue_batch();
            r.rune = batch_runes;
        }

        if (dbg && is_web_running()) {
            dbg->expanded.set(net_focus);
            dbg->img_frame.set(std::move(frame.img_frame));
        }

        return std::make_tuple(std::optional<DetIo::second_type>(std::move(r)));
    });

    const bool vision_d_plan_loop = very_aimer.uses_vision_d_mpc();
    auto plan_signal = std::make_shared<runtime::FramePlanLoop::Signal>();
    std::mutex solver_mutex;
    std::function<void()> run_solver_step;

    s.register_task<DetIo>("tracker", [&, plan_signal](DetIo::second_type&& io) {
        bool updated = false;
        {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
        if (io.armors) {
            for (const auto& armors_raw: io.armors.value()) {
                auto armors = armors_raw;
                auto is_ally = [&](const auto& obj) {
                    return (enemy_color == EnemyColor::BLUE
                            && obj.color == auto_aim::ArmorColor::RED)
                        || (enemy_color == EnemyColor::RED
                            && obj.color == auto_aim::ArmorColor::BLUE);
                };
                armors.armors.erase(
                    std::remove_if(armors.armors.begin(), armors.armors.end(), is_ally),
                    armors.armors.end()
                );
                armors.lights.erase(
                    std::remove_if(armors.lights.begin(), armors.lights.end(), is_ally),
                    armors.lights.end()
                );
                auto camera_cv_in_odom = tf->pose_a_in_b(
                    SimpleFrame(armors.frame_id),
                    SimpleFrame::ODOM,
                    armors.timestamp
                ); //转到odom
                armors.frame_id = std::to_underlying(SimpleFrame::ODOM);
                auto __armor_target =
                    armor_tracker.track(armors, camera_info, camera_cv_in_odom, armors.frame_id);
                auto_aim_fsm_controller.update(
                    __armor_target.get_target_state().vyaw(),
                    __armor_target.jumped,
                    __armor_target.get_target_state().timestamp
                );
                armor_target.write(__armor_target);
                updated = true;
                if (io.detect_start) {
                    auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          Clock::now() - io.detect_start.value()
                    )
                                          .count();
                    log_ctx.latency_ms_total += latency_ms;
                }

                log_ctx.found_count += armor_tracker.get_count();
                log_ctx.ekf_count += armor_tracker.get_ekf_count();
                armor_tracker.reset_count();
                armor_tracker.reset_ekf_count();
                if (dbg) {
                    dbg->armors.set(armors);
                }

                log_ctx.track_count++;
            }
        } else if (io.rune) {
            for (auto& rune: io.rune.value()) {
                auto camera_cv_in_odom = tf->pose_a_in_b(
                    SimpleFrame(rune.frame_id),
                    SimpleFrame::ODOM,
                    rune.timestamp
                ); //转到odom
                rune.frame_id = std::to_underlying(SimpleFrame::ODOM);

                auto __rune_target =
                    rune_tracker.track(rune, camera_info, camera_cv_in_odom, rune.frame_id);

                rune_target.write(__rune_target);
                updated = true;
                if (io.detect_start) {
                    auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          Clock::now() - io.detect_start.value()
                    )
                                          .count();
                    log_ctx.latency_ms_total += latency_ms;
                }
                log_ctx.found_count += rune_tracker.get_count();
                rune_tracker.reset_count();

                if (dbg) {
                    dbg->rune.set(rune);
                }
                log_ctx.track_count++;
            }
        }
        }
        if (vision_d_plan_loop && updated) {
            plan_signal->notify();
        }
    });
    run_solver_step = [&]() {
        std::lock_guard<std::mutex> solver_lock(solver_mutex);
        log_ctx.solve_count++;
        auto a_target = armor_target.read(); //需要转为相对gimbal_odom
        int old_this_id = a_target.this_id;
        auto gimbal_odom_state_in_odom = wheel_odometry.state;
        gimbal_odom_state_in_odom.predict(Clock::now());
        a_target.set_target_state([&](auto& s) {
            namespace idx = auto_aim::armor_point_motion_model::idx;
            s.frame_id = std::to_underlying(SimpleFrame::GIMBAL_ODOM);
            const auto pos = gimbal_odom_state_in_odom.pos();
            const auto vel = gimbal_odom_state_in_odom.vel();
            s.x[idx::CX] -= pos.x();
            s.x[idx::CY] -= pos.y();
            s.x[idx::CZ] -= pos.z();
            s.x[idx::VCX] -= vel.x();
            s.x[idx::VCY] -= vel.y();
            s.x[idx::VCZ] -= vel.z();
            // s.x[idx::VCX] =0;
            // s.x[idx::VCY] =0;
            // s.x[idx::VCZ] =0;
            // s.x[idx::VYAW] =12.0;
        });
        a_target.this_id = old_this_id;
        auto r_target = rune_target.read();
        old_this_id = r_target.this_id;
        r_target.set_target_state([&](auto& s) {
            namespace idx = auto_buff::motion_model::idx;
            s.frame_id = std::to_underlying(SimpleFrame::GIMBAL_ODOM);
            const auto pos = gimbal_odom_state_in_odom.pos();
            const auto vel = gimbal_odom_state_in_odom.vel();
            s.x[idx::CX] -= pos.x();
            s.x[idx::CY] -= pos.y();
            s.x[idx::CZ] -= pos.z();
        });
        r_target.this_id = old_this_id;
        GimbalCmd cmd {
            .appear = false,
        };
        int command_target_key = -1;
        const auto solver_tf_now = Clock::now() - serial_tf_offset;
        auto shoot_in_gimbal_odom =
            tf->pose_a_in_b(SimpleFrame::SHOOT, SimpleFrame::GIMBAL_ODOM, solver_tf_now);
        auto gimbal_in_gimbal_odom =
            tf->pose_a_in_b(SimpleFrame::GIMBAL, SimpleFrame::GIMBAL_ODOM, solver_tf_now);
        if (a_target.check()) {
            command_target_key = std::to_underlying(a_target.target_number);
            very_aimer.set_operator_offset(operator_offset);
            cmd = very_aimer.very_aim(
                a_target,
                bullet_speed,
                auto_aim_fsm_controller.get_state(),
                shoot_in_gimbal_odom,
                gimbal_in_gimbal_odom
            );

        } else if (r_target.check()) {
            command_target_key = 1000;
            rune_aimer.set_operator_offset(operator_offset);
            cmd =
                rune_aimer.aim(r_target, bullet_speed, shoot_in_gimbal_odom, gimbal_in_gimbal_odom);
            // 与 sentry 一致：打符用 tick_fire 限频，避免每毫秒都请求一次开火。
            cmd.fire_advice = rune_aimer.tick_fire(r_target, cmd);
        }


        if (timing_logger) {
            static int timing_tick = 0;
            if (++timing_tick >= debug_timing_every_n) {
                timing_tick = 0;
                utils::TimingCsvSample sample;
                sample.wall_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::system_clock::now().time_since_epoch()
                                      ).count();
                const auto to_ns = [](const TimePoint& t) {
                    return std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t.time_since_epoch()
                           ).count();
                };
                const auto t_img = a_target.last_update;
                const auto t_state = a_target.get_target_state().timestamp;
                const auto t_cmd =
                    (cmd.timestamp.time_since_epoch().count() != 0) ? cmd.timestamp : Clock::now();
                sample.t_img_ns = to_ns(t_img);
                sample.t_cmd_ns = to_ns(t_cmd);
                sample.target_state_ts_ns = to_ns(t_state);
                if (t_img.time_since_epoch().count() != 0) {
                    sample.dt_img_cmd_ms =
                        std::chrono::duration<double, std::milli>(t_cmd - t_img).count();
                }
                if (t_state.time_since_epoch().count() != 0) {
                    sample.dt_state_cmd_ms =
                        std::chrono::duration<double, std::milli>(t_cmd - t_state).count();
                }
                if (dbg) {
                    const auto frame_ts = dbg->img_frame.get().timestamp;
                    sample.frame_ts_ns = to_ns(frame_ts);
                    if (frame_ts.time_since_epoch().count() != 0) {
                        sample.frame_dt_cmd_ms =
                            std::chrono::duration<double, std::milli>(t_cmd - frame_ts).count();
                        if (t_img.time_since_epoch().count() != 0) {
                            sample.target_frame_age_ms =
                                std::chrono::duration<double, std::milli>(frame_ts - t_img).count();
                        }
                    }
                }

                const auto rpy_cmd = utils::matrix2rpy<double>(
                    gimbal_in_gimbal_odom.linear(), utils::RPYOrder::ZYX
                );
                sample.gimbal_yaw_cmd_deg = angles::to_degrees(rpy_cmd[2]);
                sample.gimbal_pitch_cmd_deg = -angles::to_degrees(rpy_cmd[1]);
                if (t_img.time_since_epoch().count() != 0) {
                    try {
                        const auto gimbal_at_img = tf->pose_a_in_b(
                            SimpleFrame::GIMBAL, SimpleFrame::GIMBAL_ODOM, t_img
                        );
                        const auto rpy_img = utils::matrix2rpy<double>(
                            gimbal_at_img.linear(), utils::RPYOrder::ZYX
                        );
                        sample.gimbal_yaw_img_deg = angles::to_degrees(rpy_img[2]);
                        sample.gimbal_pitch_img_deg = -angles::to_degrees(rpy_img[1]);
                        sample.tf_img_valid = 1;
                    } catch (...) {
                        sample.tf_img_valid = 0;
                    }
                }

                if (serial) {
                    if (const auto angles = serial->latest_gimbal_angles()) {
                        sample.gimbal_yaw_deg = angles->first;
                        sample.telemetry_yaw_deg = angles->first;
                        sample.telemetry_pitch_deg = angles->second;
                        sample.telemetry_valid = 1;
                    }
                    if (const auto rx = serial->latest_gimbal_rx_time()) {
                        sample.telemetry_age_ms =
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - *rx
                            ).count();
                    }
                }

                if (tf) {
                    if (const auto tf_stamp = tf->latest_stamp(
                            SimpleFrame::GIMBAL_ODOM, SimpleFrame::GIMBAL
                        )) {
                        sample.tf_age_ms = std::chrono::duration<double, std::milli>(
                            (t_cmd - serial_tf_offset) - *tf_stamp
                        ).count();
                    }
                }

                sample.cmd_yaw_deg = cmd.yaw;
                sample.cmd_pitch_deg = cmd.pitch;
                sample.target_yaw_deg = cmd.target_yaw;
                sample.target_pitch_deg = cmd.target_pitch;
                if (a_target.check()) {
                    const auto& target_state = a_target.get_target_state();
                    const auto target_pos = target_state.pos();
                    sample.target_bearing_deg =
                        angles::to_degrees(std::atan2(target_pos.y(), target_pos.x()));
                    const auto target_from_shoot =
                        target_pos - shoot_in_gimbal_odom.translation();
                    sample.shoot_bearing_deg = angles::to_degrees(
                        std::atan2(target_from_shoot.y(), target_from_shoot.x())
                    );
                }
                sample.err_cmd_target_yaw_deg =
                    angles::shortest_angular_distance_degrees(cmd.yaw, cmd.target_yaw);
                sample.err_cmd_gimbal_yaw_deg =
                    angles::shortest_angular_distance_degrees(cmd.yaw, sample.gimbal_yaw_cmd_deg);
                sample.err_target_gimbal_yaw_deg = angles::shortest_angular_distance_degrees(
                    cmd.target_yaw, sample.gimbal_yaw_cmd_deg
                );
                sample.err_cmd_target_pitch_deg =
                    angles::shortest_angular_distance_degrees(cmd.pitch, cmd.target_pitch);
                sample.err_cmd_gimbal_pitch_deg =
                    angles::shortest_angular_distance_degrees(cmd.pitch, sample.gimbal_pitch_cmd_deg);
                sample.err_target_gimbal_pitch_deg = angles::shortest_angular_distance_degrees(
                    cmd.target_pitch, sample.gimbal_pitch_cmd_deg
                );
                sample.cmd_v_yaw = cmd.v_yaw;
                sample.cmd_a_yaw = cmd.a_yaw;
                sample.cmd_v_pitch = cmd.v_pitch;
                sample.cmd_a_pitch = cmd.a_pitch;

                sample.select_id = cmd.select_id;
                sample.target_number = static_cast<int>(a_target.target_number);
                sample.this_id = a_target.this_id;
                sample.match_count = a_target.last_match_count;
                sample.aim_id = cmd.select_id;
                sample.aimer_type = [] (const YAML::Node& cfg) {
                    return cfg["very_aimer"] && cfg["very_aimer"]["type"]
                        ? cfg["very_aimer"]["type"].as<std::string>()
                        : std::string("unknown");
                }(config);
                {
                    // 只记录 this_id 变化时的直接原因，避免把同一目标的重复采样算成切换。
                    static int last_this_id = -1;
                    static int last_target_number = -1;
                    const int cur_this_id = a_target.this_id;
                    const int cur_target_number = static_cast<int>(a_target.target_number);
                    if (cur_this_id < 0) {
                        sample.this_id_change_reason = "no_target";
                    } else if (last_this_id < 0) {
                        sample.this_id_change_reason = "first_target";
                    } else if (cur_this_id == last_this_id) {
                        sample.this_id_change_reason = "same";
                    } else if (a_target.jumped) {
                        sample.this_id_change_reason = "jumped";
                    } else if (cur_target_number != last_target_number) {
                        sample.this_id_change_reason = "target_number_changed";
                    } else if (a_target.track_state.tracker_state
                               == auto_aim::ArmorTarget::TrackState::DETECTING) {
                        sample.this_id_change_reason = "new_target_detecting";
                    } else if (a_target.track_state.tracker_state
                               == auto_aim::ArmorTarget::TrackState::TEMP_LOST) {
                        sample.this_id_change_reason = "temp_lost_reacquire";
                    } else {
                        sample.this_id_change_reason = "new_id";
                    }
                    last_this_id = cur_this_id;
                    last_target_number = cur_target_number;
                }
                if (a_target.check()) {
                    const auto aim_debug = very_aimer.get_debug_info();
                    sample.this_id_same = aim_debug.this_id_same ? 1 : 0;
                    sample.fsm_same = aim_debug.fsm_same ? 1 : 0;
                    sample.bullet_speed_same = aim_debug.bullet_speed_same ? 1 : 0;
                    sample.shoot_pose_same = aim_debug.shoot_pose_same ? 1 : 0;
                    sample.gimbal_pose_same = aim_debug.gimbal_pose_same ? 1 : 0;
                    sample.time_in_traj_s = aim_debug.time_in_traj_s;
                }
                sample.frame_id = a_target.get_target_state().frame_id;
                sample.fsm = static_cast<int>(auto_aim_fsm_controller.get_state());
                sample.track_state =
                    static_cast<int>(a_target.track_state.tracker_state);
                sample.target_valid = a_target.check() ? 1 : 0;
                sample.cmd_appear = cmd.appear ? 1 : 0;
                sample.cmd_valid = cmd.is_valid() ? 1 : 0;
                sample.fire = cmd.fire_advice ? 1 : 0;

                if (cmd.appear && cmd.is_valid()) {
                    sample.distance_m =
                        (cmd.aim_point.pose.translation() - shoot_in_gimbal_odom.translation()).norm();
                    sample.fly_time_s = cmd.fly_time;
                }
                sample.exposure_us = camera_exposure_cfg_us;
                if (camera) {
                    const double actual = camera->get_ExposureTime();
                    if (actual > 0.0) sample.exposure_us = actual;
                }
                sample.camera_fps = camera_fps;
                sample.bullet_speed = bullet_speed;
                sample.serial_send_to_image_us = serial_send_to_image_microseconds;
                sample.mode = static_cast<int>(mode.load());
                timing_logger->write(sample);
            }
        }

        if (serial) {
            // 下位机（control-2026）只接受「长度 + ID」帧：
            // 每帧 38 字节 = id1{yaw,pitch} + id2{fire} + id3{v_yaw,v_pitch} + id4{a_yaw,a_pitch}。
            // 角度是绝对角（度）：yaw 同向；视觉 pitch 抬头为正，固件 IMU Pitch 抬头为负，
            // 因此只在发送边界乘 pitch_sign（见 config/dong.yaml，哨兵实机验证为 -1）。
            //
            // 发送前门控：只处理输出，不改 very_aimer / 选板 / PnP。
            // profile=raw 时完全旁路 command_guard，和 srm_auto_aim 的
            // auto_aim_debug_mpc 一样直接发送 MPC 解出的位置。
            const auto now = Clock::now();
            const bool raw_target = cmd.appear && cmd.is_valid();
            const float raw_yaw_deg = static_cast<float>(cmd.yaw);
            const float raw_pitch_deg = static_cast<float>(pitch_sign * cmd.pitch);
            const float raw_v_yaw_deg_s = static_cast<float>(cmd.v_yaw);
            const float raw_v_pitch_deg_s = static_cast<float>(pitch_sign * cmd.v_pitch);
            const float raw_a_yaw_deg_s2 = static_cast<float>(cmd.a_yaw);
            const float raw_a_pitch_deg_s2 = static_cast<float>(pitch_sign * cmd.a_pitch);
            const double target_distance_m = raw_target
                ? (cmd.aim_point.pose.translation() - shoot_in_gimbal_odom.translation()).norm()
                : 0.0;
            VisionCommandGuard::Output guarded;
            if (command_guard_bypass) {
                static bool has_raw_hold = false;
                static float raw_hold_yaw_deg = 0.0F;
                static float raw_hold_pitch_deg = 0.0F;
                if (raw_target) {
                    raw_hold_yaw_deg = raw_yaw_deg;
                    raw_hold_pitch_deg = raw_pitch_deg;
                    has_raw_hold = true;
                    guarded.has_output = true;
                    guarded.tracking = true;
                    guarded.fire = cmd.fire_advice;
                    guarded.yaw_deg = raw_yaw_deg;
                    guarded.pitch_deg = raw_pitch_deg;
                    guarded.state = VisionCommandGuard::State::Tracking;
                } else if (has_raw_hold) {
                    // 与 SrmLink 一致：失目标时保留最后有效角度，只撤开火。
                    guarded.has_output = true;
                    guarded.tracking = false;
                    guarded.fire = false;
                    guarded.yaw_deg = raw_hold_yaw_deg;
                    guarded.pitch_deg = raw_hold_pitch_deg;
                    guarded.state = VisionCommandGuard::State::NoTarget;
                }
            } else {
                if (!command_guard.has_output()) {
                    if (const auto telemetry = serial->latest_gimbal_angles()) {
                        command_guard.seed(telemetry->first, telemetry->second, now);
                    }
                }
                guarded = command_guard.update(
                    {
                        .target = raw_target,
                        .target_key = command_target_key,
                        .yaw_deg = raw_yaw_deg,
                        .pitch_deg = raw_pitch_deg,
                        .distance_m = target_distance_m,
                        .fire = cmd.fire_advice,
                    },
                    now
                );
            }
            bool command_accepted = false;
            if (guarded.has_output) {
                const float guarded_yaw_deg = static_cast<float>(guarded.yaw_deg);
                const float guarded_pitch_deg = static_cast<float>(guarded.pitch_deg);
                const auto hold =
                    VisionCommand::encode_hold(guarded_yaw_deg, guarded_pitch_deg);
                command_accepted = serial->set_latest_command(
                    guarded.tracking
                        ? VisionCommand::encode(
                              guarded_yaw_deg,
                              guarded_pitch_deg,
                              guarded.fire,
                              raw_v_yaw_deg_s,
                              raw_v_pitch_deg_s,
                              raw_a_yaw_deg_s2,
                              raw_a_pitch_deg_s2
                          )
                        : hold,
                    hold
                );
            } else {
                auto idle = VisionCommand::encode_idle();
                command_accepted = serial->set_latest_command(idle, idle);
            }

            static bool first_command_log = true;
            static bool last_command_accepted = false;
            static bool last_tracking = false;
            static bool last_fire_request = false;
            static int last_select_id = -1;
            static VisionCommandGuard::State last_guard_state =
                VisionCommandGuard::State::NoTarget;
            static TimePoint last_command_log;
            const bool fire_request = guarded.tracking && guarded.fire;
            const bool command_state_changed = first_command_log
                || command_accepted != last_command_accepted
                || guarded.tracking != last_tracking || fire_request != last_fire_request
                || cmd.select_id != last_select_id
                || guarded.state != last_guard_state;
            if (command_state_changed
                && Clock::now() - last_command_log >= std::chrono::milliseconds(100))
            {
                const Vec3 aim_pos = cmd.appear
                    ? cmd.aim_point.pose.translation()
                    : Vec3(0.0, 0.0, 0.0);
                AWAKENING_INFO(
                    "Vision command: profile={} serial_ready={} raw_target={} tracking={} fire={} guard={} distance_m={} raw_yaw_deg={} raw_pitch_deg={} tx_yaw_deg={} tx_pitch_deg={} select_id={} target_yaw_deg={} target_pitch_deg={} aim_x={} aim_y={} aim_z={} match_count={} target_class={}",
                    command_guard_profile,
                    command_accepted,
                    raw_target,
                    guarded.tracking,
                    fire_request,
                    VisionCommandGuard::state_name(guarded.state),
                    target_distance_m,
                    raw_yaw_deg,
                    raw_pitch_deg,
                    guarded.yaw_deg,
                    guarded.pitch_deg,
                    cmd.select_id,
                    cmd.target_yaw,
                    cmd.target_pitch,
                    aim_pos.x(),
                    aim_pos.y(),
                    aim_pos.z(),
                    a_target.last_match_count,
                    string_by_armor_class(a_target.target_number)
                );
                first_command_log = false;
                last_command_accepted = command_accepted;
                last_tracking = guarded.tracking;
                last_fire_request = fire_request;
                last_select_id = cmd.select_id;
                last_guard_state = guarded.state;
                last_command_log = Clock::now();
            }
        }

#ifdef USE_Daedalus
        if (daedalus_shm_client) {
            auto gimbal_in_gimbal_odom =
                tf->pose_a_in_b(SimpleFrame::GIMBAL, SimpleFrame::GIMBAL_ODOM, Clock::now());
            auto rpy =
                utils::matrix2rpy<double>(gimbal_in_gimbal_odom.linear(), utils::RPYOrder::ZYX);
            static TimePoint last_fire;
            bool fire = cmd.fire_advice
                && (std::abs(angles::shortest_angular_distance_degrees(
                        angles::to_degrees(rpy[2]),
                        cmd.target_yaw
                    )) < cmd.enable_yaw_diff
                    && std::abs(angles::shortest_angular_distance_degrees(
                           angles::to_degrees(-rpy[1]),
                           cmd.target_pitch
                       ))
                        < cmd.enable_pitch_diff);
            if (fire) {
                last_fire = Clock::now();
            }
            bool send_fire = false;
            if (Clock::now() - last_fire < std::chrono::milliseconds(100)) {
                send_fire = true;
            }
            daedalus_shm_client
                ->send_gimbal_cmd(cmd.yaw, -cmd.pitch, cmd.appear ? 1.0 : -1.0, send_fire);
            // daedalus_shm_client
            //     ->send_gimbal_cmd(cmd.yaw, -cmd.pitch, cmd.appear ? 1.0 : -1.0, false);
        }
#endif
        auto old_in_camera_cv = tf->pose_a_in_b(
            SimpleFrame(cmd.aim_point.frame_id),
            SimpleFrame::CAMERA_CV,
            cmd.timestamp - serial_tf_offset
        );
        cmd.aim_point.transform(old_in_camera_cv, std::to_underlying(SimpleFrame::CAMERA_CV));
        if (dbg && is_web_running()) {
            dbg->gimbal_cmd.set(cmd);
        }
    };
    runtime::FramePlanLoop plan_loop(plan_signal);
    if (vision_d_plan_loop) {
        AWAKENING_INFO(
            "solver loop: vision_d dedicated frame-driven plan thread (1000 Hz rate source disabled)"
        );
    } else {
        s.add_rate_source<>("solver", 1000.0, [&]() { run_solver_step(); });
        AWAKENING_INFO(
            "solver loop: 1000 Hz rate source (type={}, mpc_mode={})",
            config["very_aimer"]["type"].as<std::string>("unknown"),
            auto_aim::VeryAimer::mpc_mode_name(very_aimer.mpc_mode())
        );
    }
    s.add_rate_source<>("logger", 1.0, [&]() {
        double avg_latency_ms = log_ctx.latency_ms_total / log_ctx.detect_count;
        const int serial_count =
            log_ctx.serial_count + serial_rx_counter->exchange(0, std::memory_order_relaxed);
        AWAKENING_INFO(
            "detect: {} track: {} found: {} ekf_frames: {} solve: {} serial: {} camera: {} avg_latency: {:.3} ms",
            log_ctx.detect_count,
            log_ctx.track_count,
            log_ctx.found_count,
            log_ctx.ekf_count,
            log_ctx.solve_count,
            serial_count,
            log_ctx.camera_count,
            avg_latency_ms
        );
        if (dbg) {
            dbg->avg_latency_ms.set(avg_latency_ms);
        }
        log_ctx.reset();
    });
    if (dbg) {
        s.add_rate_source<>("debug", 45.0, [&]() {
            auto __armor_target = armor_target.read();
            if (is_web_running()) {
                static Web web;
                dbg->type =
                    (mode == Mode::AutoAim) ? VisionDebugCtx::AUTO_AIM : VisionDebugCtx::AUTO_BUFF;
                auto __rune_target = rune_target.read();
                int current_select_id = -1;
                if (dbg) current_select_id = dbg->gimbal_cmd.get().select_id;
                __armor_target.write_log(current_select_id);
                __rune_target.write_log();
                wheel_odometry.write_log();
                auto img_now = dbg->img_frame.get().timestamp;
                dbg->armor_target.set(__armor_target);
                dbg->rune_target.set(__rune_target);
                dbg->auto_aim_fsm_state.set(auto_aim_fsm_controller.get_state());
                auto gimbal_in_gimbal_odom = tf->pose_a_in_b(
                    SimpleFrame::GIMBAL, SimpleFrame::GIMBAL_ODOM, Clock::now() - serial_tf_offset
                );
                auto rpy =
                    utils::matrix2rpy<double>(gimbal_in_gimbal_odom.linear(), utils::RPYOrder::ZYX);
                auto gimbal_yaw_pitch =
                    std::make_pair(angles::to_degrees(rpy[2]), -angles::to_degrees(rpy[1]));
                dbg->gimbal_yaw_pitch.set(gimbal_yaw_pitch);
                bullet_pick_up.update(
                    Clock::now(),
                    dbg->gimbal_cmd.get().appear ? dbg->gimbal_cmd.get().fly_time : 0.4
                );
                auto bullet_poss =
                    bullet_pick_up.get_bullet_positions(img_now, very_aimer.get_yaw_pitch_offset());
                auto odom_in_camera_cv =
                    tf->pose_a_in_b(SimpleFrame::ODOM, SimpleFrame::CAMERA_CV, img_now);
                for (auto& pos: bullet_poss) {
                    pos = odom_in_camera_cv * pos;
                }
                dbg->odom_in_camera_cv.set(odom_in_camera_cv);
                dbg->bullet_positions.set(bullet_poss);
                web.write_debug_data(dbg.value());
                auto img = dbg->img_frame.get();
                auto debug_img = img.src_img.clone();
                if (img.format == PixelFormat::RGB) {
                    cv::cvtColor(debug_img, debug_img, cv::COLOR_RGB2BGR);
                }
                if (!debug_img.empty()) {
                    web.draw(debug_img, dbg.value());
                    web.write_shm(debug_img);
                }
            }

#ifdef USE_ROS2
            auto armors = dbg->armors.get();
            rcl::pub_armor_target_marker(
                rcl_node,
                SimpleFrame_to_str(__armor_target.get_target_state().frame_id),
                __armor_target
            );
#endif
        });
#ifdef USE_ROS2
        if (debug) {
            s.add_rate_source<>("tf_pub", 100.0, [&]() {
                rcl_tf.pub_robot_tf(*tf, [](SimpleFrame frame) {
                    return SimpleFrame_to_str(frame);
                });
            });
        }

#endif
#ifdef USE_RERUN
        if (debug) {
            s.add_rate_source<>("rerun_tf", 15.0, [&]() {
                rerun_visual::log_robot_tf(*tf, [](SimpleFrame frame) {
                    return SimpleFrame_to_str(frame);
                });
            });
        }
#endif
    }

    if (camera) {
        camera->start<CameraTag>("hik");
    }
    if (serial) {
        serial->start<SerialTag>("serial");
    }
    s.build();
    if (vision_d_plan_loop) {
        plan_loop.start(run_solver_step);
    }
    s.run();

#ifdef USE_ROS2
    std::thread([&]() { rcl_node.spin(); }).detach();
#endif
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();
    plan_loop.stop();
    if (serial) {
        serial->stop();
    }
#ifdef USE_ROS2
    rcl_node.shutdown();
#endif
    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}
