#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

namespace awakening::utils {

struct TimingCsvSample {
    std::int64_t wall_time_us = 0;
    std::int64_t t_img_ns = 0;
    std::int64_t t_cmd_ns = 0;
    std::int64_t target_state_ts_ns = 0;
    std::int64_t frame_ts_ns = 0;
    double dt_img_cmd_ms = -1.0;
    double dt_state_cmd_ms = -1.0;
    double frame_dt_cmd_ms = -1.0;
    double target_frame_age_ms = -1.0;
    double telemetry_age_ms = -1.0;
    double tf_age_ms = -1.0;

    double gimbal_yaw_img_deg = std::numeric_limits<double>::quiet_NaN();
    double gimbal_yaw_cmd_deg = std::numeric_limits<double>::quiet_NaN();
    double gimbal_pitch_img_deg = std::numeric_limits<double>::quiet_NaN();
    double gimbal_pitch_cmd_deg = std::numeric_limits<double>::quiet_NaN();
    double gimbal_yaw_deg = std::numeric_limits<double>::quiet_NaN();
    double telemetry_yaw_deg = std::numeric_limits<double>::quiet_NaN();
    double telemetry_pitch_deg = std::numeric_limits<double>::quiet_NaN();

    double cmd_yaw_deg = 0.0;
    double cmd_pitch_deg = 0.0;
    double target_yaw_deg = 0.0;
    double target_pitch_deg = 0.0;
    double target_bearing_deg = std::numeric_limits<double>::quiet_NaN();
    double shoot_bearing_deg = std::numeric_limits<double>::quiet_NaN();

    double err_cmd_target_yaw_deg = 0.0;
    double err_cmd_gimbal_yaw_deg = 0.0;
    double err_target_gimbal_yaw_deg = 0.0;
    double err_cmd_target_pitch_deg = 0.0;
    double err_cmd_gimbal_pitch_deg = 0.0;
    double err_target_gimbal_pitch_deg = 0.0;

    double cmd_v_yaw = 0.0;
    double cmd_a_yaw = 0.0;
    double cmd_v_pitch = 0.0;
    double cmd_a_pitch = 0.0;

    int select_id = -1;
    int target_number = -1;
    int this_id = -1;
    int match_count = 0;
    int frame_id = -1;
    int fsm = -1;
    int track_state = -1;
    int target_valid = 0;
    int cmd_appear = 0;
    int cmd_valid = 0;
    int fire = 0;

    double distance_m = -1.0;
    double fly_time_s = -1.0;
    double exposure_us = -1.0;
    double camera_fps = -1.0;
    double bullet_speed = -1.0;
    double serial_send_to_image_us = -1.0;

    int mode = -1;
    int tf_img_valid = 0;
    int telemetry_valid = 0;

    int this_id_same = -1;
    int fsm_same = -1;
    int bullet_speed_same = -1;
    int shoot_pose_same = -1;
    int gimbal_pose_same = -1;
    double time_in_traj_s = -1.0;

    // 末尾追加，避免改变旧列顺序。
    int aim_id = -1;
    std::string this_id_change_reason = "none";
    std::string aimer_type = "unknown";
};

class TimingCsvLogger {
public:
    explicit TimingCsvLogger(const std::string& path) : path_(path) {
        const auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        out_.open(path, std::ios::out | std::ios::trunc);
        out_ << std::fixed << std::setprecision(9);
        write_header();
    }

    ~TimingCsvLogger() {
        if (out_) out_.flush();
    }

    bool is_open() const noexcept {
        return out_.is_open();
    }

    const std::string& path() const noexcept {
        return path_;
    }

    void write(const TimingCsvSample& s) {
        if (!out_) return;
        out_ << s.wall_time_us << ','
             << s.t_img_ns << ','
             << s.t_cmd_ns << ','
             << s.target_state_ts_ns << ','
             << s.frame_ts_ns << ','
             << s.dt_img_cmd_ms << ','
             << s.dt_state_cmd_ms << ','
             << s.frame_dt_cmd_ms << ','
             << s.target_frame_age_ms << ','
             << s.telemetry_age_ms << ','
             << s.tf_age_ms << ','
             << s.gimbal_yaw_img_deg << ','
             << s.gimbal_yaw_cmd_deg << ','
             << s.gimbal_pitch_img_deg << ','
             << s.gimbal_pitch_cmd_deg << ','
             << s.gimbal_yaw_deg << ','
             << s.telemetry_yaw_deg << ','
             << s.telemetry_pitch_deg << ','
             << s.cmd_yaw_deg << ','
             << s.cmd_pitch_deg << ','
             << s.target_yaw_deg << ','
             << s.target_pitch_deg << ','
             << s.target_bearing_deg << ','
             << s.shoot_bearing_deg << ','
             << s.err_cmd_target_yaw_deg << ','
             << s.err_cmd_gimbal_yaw_deg << ','
             << s.err_target_gimbal_yaw_deg << ','
             << s.err_cmd_target_pitch_deg << ','
             << s.err_cmd_gimbal_pitch_deg << ','
             << s.err_target_gimbal_pitch_deg << ','
             << s.cmd_v_yaw << ','
             << s.cmd_a_yaw << ','
             << s.cmd_v_pitch << ','
             << s.cmd_a_pitch << ','
             << s.select_id << ','
             << s.target_number << ','
             << s.this_id << ','
             << s.match_count << ','
             << s.frame_id << ','
             << s.fsm << ','
             << s.track_state << ','
             << s.target_valid << ','
             << s.cmd_appear << ','
             << s.cmd_valid << ','
             << s.fire << ','
             << s.distance_m << ','
             << s.fly_time_s << ','
             << s.exposure_us << ','
             << s.camera_fps << ','
             << s.bullet_speed << ','
             << s.serial_send_to_image_us << ','
             << s.mode << ','
             << s.tf_img_valid << ','
             << s.telemetry_valid << ','
             << s.this_id_same << ','
             << s.fsm_same << ','
             << s.bullet_speed_same << ','
             << s.shoot_pose_same << ','
             << s.gimbal_pose_same << ','
             << s.time_in_traj_s << ','
             << s.aim_id << ','
             << s.this_id_change_reason << ','
             << s.aimer_type << '\n';
        if (++rows_ % 20 == 0) out_.flush();
    }

    void flush() {
        if (out_) out_.flush();
    }

private:
    void write_header() {
        out_ << "wall_time_us,t_img_ns,t_cmd_ns,target_state_ts_ns,frame_ts_ns,"
             << "dt_img_cmd_ms,dt_state_cmd_ms,frame_dt_cmd_ms,target_frame_age_ms,telemetry_age_ms,tf_age_ms,"
             << "gimbal_yaw_img_deg,gimbal_yaw_cmd_deg,gimbal_pitch_img_deg,gimbal_pitch_cmd_deg,"
             << "gimbal_yaw_deg,telemetry_yaw_deg,telemetry_pitch_deg,"
             << "cmd_yaw_deg,cmd_pitch_deg,target_yaw_deg,target_pitch_deg,"
             << "target_bearing_deg,shoot_bearing_deg,"
             << "err_cmd_target_yaw_deg,err_cmd_gimbal_yaw_deg,err_target_gimbal_yaw_deg,"
             << "err_cmd_target_pitch_deg,err_cmd_gimbal_pitch_deg,err_target_gimbal_pitch_deg,"
             << "cmd_v_yaw,cmd_a_yaw,cmd_v_pitch,cmd_a_pitch,"
             << "select_id,target_number,this_id,match_count,frame_id,fsm,track_state,"
             << "target_valid,cmd_appear,cmd_valid,fire,distance_m,fly_time_s,exposure_us,camera_fps,"
             << "bullet_speed,serial_send_to_image_us,mode,tf_img_valid,telemetry_valid,"
             << "this_id_same,fsm_same,bullet_speed_same,shoot_pose_same,gimbal_pose_same,time_in_traj_s,"
             << "aim_id,this_id_change_reason,aimer_type\n";
    }

    std::string path_;
    std::ofstream out_;
    int rows_ = 0;
};

} // namespace awakening::utils
