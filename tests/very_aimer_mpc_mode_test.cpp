#include "angles.h"
#include "runtime/frame_plan_loop.hpp"
#include "tasks/auto_aim/armor_control/very_aimer.hpp"
#include "tasks/auto_aim/armor_track/armor_target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "utils/buffer.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/tf.hpp"
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using namespace awakening;
using namespace awakening::auto_aim;

namespace {
void require(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error("TEST FAILED: " + msg);
    }
}

YAML::Node make_base_config(
    const std::string& type,
    const std::string& mpc_mode,
    double delay_ms = 0.0
) {
    YAML::Node c;
    c["type"] = type;
    if (!mpc_mode.empty()) {
        c["mpc_mode"] = mpc_mode;
    }
    c["gimbal_response_delay_ms"] = delay_ms;
    c["base_yaw_offset"] = 0.0;
    c["base_pitch_offset"] = 0.0;
    c["sample_total_time"] = 1.0;
    c["sample_horizon"] = 100; // dt = 0.01s, 2*dt = 20ms
    c["fire_delay_min"] = 0.01;
    c["fire_delay_max"] = 0.05;
    c["max_yaw_acc"] = 50.0;
    c["max_pitch_acc"] = 100.0;
    c["prediction_delay"] = 0.0;
    c["aim_center_more_prediction_time"] = 0.0;
    c["shooting_range_h"] = 0.12;
    c["shooting_range_w_small"] = 0.12;
    c["shooting_range_w_large"] = 0.24;
    c["min_enable_pitch_deg"] = 0.25;
    c["min_enable_yaw_deg"] = 0.25;
    c["ballistic_trajectory"]["gravity"] = 9.8;
    c["ballistic_trajectory"]["resistance"] = 0.001;
    return c;
}

ArmorTarget make_moving_target(TimePoint ts, double vx, double vy, double vz, double vyaw) {
    ArmorTarget target;
    target.target_number = ArmorClass::NO3;
    target.this_id = 42;
    target.track_state.tracker_state = ArmorTarget::TrackState::TRACKING;
    target.cfg.lost_time_thres = 0.5;
    target.last_update = ts;
    target.set_target_state([&](armor_point_motion_model::State& s) {
        namespace idx = armor_point_motion_model::idx;
        s.x.setZero();
        s.x[idx::CX] = 4.0;
        s.x[idx::VCX] = vx;
        s.x[idx::CY] = 0.0;
        s.x[idx::VCY] = vy;
        s.x[idx::CZ] = 0.0;
        s.x[idx::VCZ] = vz;
        s.x[idx::C_ROT_Z] = 0.0;
        s.x[idx::VYAW] = vyaw;
        s.x[idx::LOG_R1] = std::log(0.25);
        s.x[idx::LOG_R2] = std::log(0.25);
        s.x[idx::H] = 0.0;
        s.timestamp = ts;
        s.frame_id = 1;
    });
    return target;
}

void test_mode_parsing_and_yaml_configs() {
    // Default when mpc_mode is omitted -> vision_d
    {
        VeryAimer aimer(make_base_config("mpc", ""));
        require(aimer.mpc_mode() == MpcMode::VisionD, "default mpc_mode must be vision_d");
        require(aimer.uses_vision_d_mpc(), "type=mpc + default mpc_mode must enable vision_d MPC");
    }
    // Explicit awakening
    {
        VeryAimer aimer(make_base_config("mpc", "awakening"));
        require(aimer.mpc_mode() == MpcMode::Awakening, "explicit awakening mpc_mode");
        require(!aimer.uses_vision_d_mpc(), "awakening mpc_mode must not report vision_d MPC");
    }
    // type=limit with mpc_mode=vision_d must keep limit path unaffected
    {
        VeryAimer aimer(make_base_config("limit", "vision_d"));
        require(aimer.mpc_mode() == MpcMode::VisionD, "limit config still stores parsed mpc_mode");
        require(!aimer.uses_vision_d_mpc(), "type=limit must not enable vision_d MPC");
    }
    // Invalid mpc_mode throws
    bool threw = false;
    try {
        VeryAimer bad(make_base_config("mpc", "bogus_mode"));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "invalid mpc_mode must throw std::invalid_argument");

    // Invalid gimbal_response_delay_ms throws
    threw = false;
    try {
        VeryAimer bad(make_base_config("mpc", "vision_d", 150.0));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "out-of-range gimbal_response_delay_ms must throw std::invalid_argument");

    // Production configs load cleanly with expected mpc_mode
    {
        auto dong = YAML::LoadFile("config/dong.yaml");
        VeryAimer dong_aimer(dong["very_aimer"]);
        const std::string dong_type = dong["very_aimer"]["type"].as<std::string>();
        require(dong_type == "limit" || dong_type == "mpc" || dong_type == "MPC",
                "dong.yaml has valid very_aimer.type");
        require(dong_aimer.mpc_mode() == MpcMode::VisionD, "dong.yaml sets mpc_mode=vision_d");
        const bool expect_vision_d = (dong_type == "mpc" || dong_type == "MPC");
        require(dong_aimer.uses_vision_d_mpc() == expect_vision_d,
                "dong.yaml uses_vision_d_mpc matches configured type");
    }
    {
        auto sentry = YAML::LoadFile("config/sentry.yaml");
        VeryAimer sentry_aimer(sentry["very_aimer"]);
        require(sentry["very_aimer"]["type"].as<std::string>() == "mpc", "sentry.yaml keeps type=mpc");
        require(sentry_aimer.mpc_mode() == MpcMode::VisionD, "sentry.yaml sets mpc_mode=vision_d");
        require(sentry_aimer.uses_vision_d_mpc(), "sentry.yaml activates vision_d MPC");
    }
}

void test_reuse_vs_recompute_and_lookahead() {
    const ISO3 shoot = ISO3::Identity();
    const ISO3 gimbal = ISO3::Identity();
    const double bullet_speed = 22.0;

    // Lateral velocity vy = 2.0 m/s at x = 4.0 m (armor at x = 3.75 m) => angular rate ~ 0.533 rad/s (~30.6 deg/s)
    // Over 20ms (2*dt), yaw lead should be ~ +0.61 deg.
    // With 55ms gimbal_response_delay_ms, extra_lead = 35ms => total lead 55ms => ~ +1.68 deg (~ +1.07 deg above 0ms delay).
    const auto t0 = Clock::now();
    const auto target = make_moving_target(t0, 0.0, 2.0, 0.0, 0.0);

    VeryAimer awakening_aimer(make_base_config("mpc", "awakening", 0.0));
    VeryAimer vision_d_0ms(make_base_config("mpc", "vision_d", 0.0));
    VeryAimer vision_d_55ms(make_base_config("mpc", "vision_d", 55.0));

    const auto cmd_awake_1 =
        awakening_aimer.very_aim(target, bullet_speed, AutoAimFsm::AIM_SINGLE_ARMOR, shoot, gimbal);
    const auto cmd_vd0_1 =
        vision_d_0ms.very_aim(target, bullet_speed, AutoAimFsm::AIM_SINGLE_ARMOR, shoot, gimbal);
    const auto cmd_vd55_1 =
        vision_d_55ms.very_aim(target, bullet_speed, AutoAimFsm::AIM_SINGLE_ARMOR, shoot, gimbal);

    require(cmd_awake_1.appear && cmd_awake_1.is_valid(), "awakening cmd1 valid");
    require(cmd_vd0_1.appear && cmd_vd0_1.is_valid(), "vision_d 0ms cmd1 valid");
    require(cmd_vd55_1.appear && cmd_vd55_1.is_valid(), "vision_d 55ms cmd1 valid");

    const double lead_vd0_vs_awake =
        angles::shortest_angular_distance_degrees(cmd_awake_1.yaw, cmd_vd0_1.yaw);
    const double lead_vd55_vs_vd0 =
        angles::shortest_angular_distance_degrees(cmd_vd0_1.yaw, cmd_vd55_1.yaw);
    const double err_vd0_cmd_target =
        std::abs(angles::shortest_angular_distance_degrees(cmd_vd0_1.yaw, cmd_vd0_1.target_yaw));

    std::cout << "lead_vd0_vs_awake_deg=" << lead_vd0_vs_awake
              << " lead_vd55_vs_vd0_deg=" << lead_vd55_vs_vd0
              << " err_vd0_cmd_target_deg=" << err_vd0_cmd_target << "\n";

    // Expected ~0.61 deg for +20ms lookahead (allow generous tolerance for ballistic fly_time shift & execution jitter)
    require(lead_vd0_vs_awake > 0.45 && lead_vd0_vs_awake < 0.85,
            "vision_d (delay=0) should lead awakening by ~20ms (~0.61 deg)");
    // Expected ~1.07 deg for additional 35ms (55ms - 20ms) extra lead
    require(lead_vd55_vs_vd0 > 0.85 && lead_vd55_vs_vd0 < 1.35,
            "vision_d (delay=55ms) should lead vision_d (delay=0) by ~35ms (~1.07 deg)");
    // Command yaw (+20ms) and target_yaw (+20ms) should closely agree on smooth constant-velocity motion
    require(err_vd0_cmd_target < 0.15,
            "vision_d cmd.yaw and cmd.target_yaw must be time-aligned at +2*dt");

    // Now sleep 15ms and call again with the EXACT same target (same this_id):
    // - awakening mode MUST reuse cached trajectory (is_same == true, time_in_traj_s > 0.01)
    // - vision_d mode MUST NOT reuse cached trajectory (is_same == false, time_in_traj_s == 0.0)
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    const auto cmd_awake_2 =
        awakening_aimer.very_aim(target, bullet_speed, AutoAimFsm::AIM_SINGLE_ARMOR, shoot, gimbal);
    const auto awake_dbg = awakening_aimer.get_debug_info();
    const auto cmd_vd0_2 =
        vision_d_0ms.very_aim(target, bullet_speed, AutoAimFsm::AIM_SINGLE_ARMOR, shoot, gimbal);
    const auto vd_dbg = vision_d_0ms.get_debug_info();

    require(cmd_awake_2.appear && cmd_vd0_2.appear, "second calls valid");
    require(awake_dbg.is_same && awake_dbg.time_in_traj_s >= 0.010,
            "awakening mode must reuse cached trajectory on unchanged input");
    require(!vd_dbg.is_same && vd_dbg.time_in_traj_s == 0.0,
            "vision_d mode must never reuse cached trajectory");
    require(vd_dbg.this_id_same,
            "vision_d debug info still reports this_id_same=true for diagnostics");

    // Because target is moving continuously, the recomputed vision_d yaw should advance by ~15ms (~0.46 deg)
    const double vd_advance_deg =
        angles::shortest_angular_distance_degrees(cmd_vd0_1.yaw, cmd_vd0_2.yaw);
    std::cout << "vd_advance_15ms_deg=" << vd_advance_deg << "\n";
    require(vd_advance_deg > 0.30 && vd_advance_deg < 0.80,
            "vision_d recomputed yaw should advance with real elapsed time");

    // Verify delay <= 2*dt (e.g. 15ms <= 20ms) gives zero extra lead (same output as delay=0ms,
    // compensating for wall time elapsed between the two sequential very_aim calls)
    VeryAimer vision_d_15ms(make_base_config("mpc", "vision_d", 15.0));
    const auto cmd_vd0_ref =
        vision_d_0ms.very_aim(target, bullet_speed, AutoAimFsm::AIM_SINGLE_ARMOR, shoot, gimbal);
    const auto cmd_vd15_ref =
        vision_d_15ms.very_aim(target, bullet_speed, AutoAimFsm::AIM_SINGLE_ARMOR, shoot, gimbal);
    const double between_calls_s =
        std::chrono::duration<double>(cmd_vd15_ref.timestamp - cmd_vd0_ref.timestamp).count();
    const double expected_motion_deg = cmd_vd0_ref.v_yaw * between_calls_s;
    const double actual_diff_deg =
        angles::shortest_angular_distance_degrees(cmd_vd0_ref.yaw, cmd_vd15_ref.yaw);
    const double residual_15_vs_0 = std::abs(actual_diff_deg - expected_motion_deg);
    std::cout << "diff_15ms_vs_0ms_deg=" << actual_diff_deg
              << " expected_motion_deg=" << expected_motion_deg
              << " residual_deg=" << residual_15_vs_0 << "\n";
    require(residual_15_vs_0 < 0.02,
            "delay <= 2*dt must produce extra_lead = 0 (same output as delay=0)");

    // Also verify sample_horizon=200 (dt=0.005s, 2*dt=10ms, as in sentry.yaml/dong.yaml)
    // and AIM_WHOLE_CAR_CENTER work cleanly in vision_d mode.
    auto cfg_200 = make_base_config("mpc", "vision_d", 55.0);
    cfg_200["sample_horizon"] = 200;
    VeryAimer vision_d_200(cfg_200);
    const auto cmd_center = vision_d_200.very_aim(
        target, bullet_speed, AutoAimFsm::AIM_WHOLE_CAR_CENTER, shoot, gimbal
    );
    require(cmd_center.appear && cmd_center.is_valid(),
            "vision_d mode with horizon=200 and AIM_WHOLE_CAR_CENTER must produce valid command");
}

void test_frame_plan_loop_coalescing() {
    runtime::FramePlanLoop loop;
    auto sig = loop.signal();

    std::mutex gate_mtx;
    std::condition_variable gate_cv;
    bool in_first_solve = false;
    bool allow_release = false;
    std::atomic<int> solve_count { 0 };
    std::atomic<int> latest_value { 0 };
    std::atomic<int> observed_last_value { 0 };

    loop.start([&]() {
        const int c = solve_count.fetch_add(1) + 1;
        if (c == 1) {
            std::unique_lock<std::mutex> lk(gate_mtx);
            in_first_solve = true;
            gate_cv.notify_all();
            gate_cv.wait(lk, [&]() { return allow_release; });
        }
        observed_last_value.store(latest_value.load());
    });

    latest_value.store(1);
    sig->notify();
    {
        std::unique_lock<std::mutex> lk(gate_mtx);
        require(
            gate_cv.wait_for(lk, std::chrono::seconds(2), [&]() { return in_first_solve; }),
            "FramePlanLoop did not wake for first notification"
        );
    }

    // Flood 50 updates while the first solve is running; they must coalesce into ONE follow-up solve.
    for (int i = 2; i <= 51; ++i) {
        latest_value.store(i);
        sig->notify();
    }
    {
        std::lock_guard<std::mutex> lk(gate_mtx);
        allow_release = true;
    }
    gate_cv.notify_all();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (observed_last_value.load() != 51 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    loop.stop();

    std::cout << "coalesced_solve_count=" << solve_count.load()
              << " observed_last_value=" << observed_last_value.load() << "\n";
    require(observed_last_value.load() == 51, "FramePlanLoop must observe latest coalesced state");
    require(solve_count.load() == 2, "50 notifications during in-flight solve must coalesce into 1 follow-up solve");
}

void test_tbb_work_stealing_no_deadlock() {
    // Regression test for the exact freeze bug:
    // Running TinyMpcTrajectory::solve() (which uses tbb::parallel_invoke) inside
    // Scheduler's tbb::task_group while holding a non-recursive std::mutex used to
    // self-deadlock via TBB work-stealing within ~1-4 seconds.
    struct TickTag {};
    using TickIO = IOPair<TickTag, int>;

    auto cfg = make_base_config("mpc", "vision_d", 0.0);
    cfg["sample_horizon"] = 200;
    VeryAimer aimer(cfg);
    const ISO3 shoot = ISO3::Identity();
    const ISO3 gimbal = ISO3::Identity();
    const double bullet_speed = 21.0;

    Scheduler s;
    auto src = s.register_source<TickIO>("tick");
    std::mutex task_mutex;
    std::atomic<int> completed { 0 };
    std::atomic<int> valid_cmds { 0 };

    s.register_task<TickIO>("locked_mpc_task", [&](TickIO::second_type&& id) {
        std::lock_guard<std::mutex> lock(task_mutex);
        auto target = make_moving_target(Clock::now(), 0.0, 2.0, 0.0, 4.0);
        target.this_id = id;
        const auto cmd =
            aimer.very_aim(target, bullet_speed, AutoAimFsm::AIM_WHOLE_CAR_ARMOR, shoot, gimbal);
        if (cmd.appear && cmd.is_valid()) {
            valid_cmds.fetch_add(1);
        }
        completed.fetch_add(1);
    });

    s.build();
    s.run();

    constexpr int kFrames = 250;
    for (int i = 0; i < kFrames; ++i) {
        s.runtime_push_source<TickIO>(src, [i]() {
            return std::make_tuple(std::optional<int>(i));
        });
        if (i % 4 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (completed.load() < kFrames && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const int done = completed.load();
    s.stop();

    std::cout << "tbb_isolation_completed=" << done << "/" << kFrames
              << " valid_cmds=" << valid_cmds.load() << "\n";
    require(done == kFrames, "TBB work-stealing isolation must prevent self-deadlock");
    require(valid_cmds.load() == kFrames, "all isolated MPC solves must yield valid commands");
}

void test_scheduler_with_frame_plan_loop_stress() {
    // End-to-end replica of standard.cpp / sentry.cpp vision_d pipeline:
    // 300 Hz camera frames + 1000 Hz serial telemetry inside Scheduler, decoupled via
    // SWMR + FramePlanLoop running horizon=200 vision_d MPC.
    struct CamTag {};
    struct SerTag {};
    using CamIO = IOPair<CamTag, int>;
    using SerIO = IOPair<SerTag, int>;

    auto cfg = make_base_config("mpc", "vision_d", 25.0);
    cfg["sample_horizon"] = 200;
    VeryAimer aimer(cfg);
    const ISO3 shoot = ISO3::Identity();
    const ISO3 gimbal = ISO3::Identity();
    const double bullet_speed = 21.0;

    Scheduler s;
    utils::SWMR<ArmorTarget> target_swmr;
    auto plan_signal = std::make_shared<runtime::FramePlanLoop::Signal>();
    std::atomic<int> tracked { 0 };
    std::atomic<int> solved { 0 };
    std::atomic<int> serial_rx { 0 };

    auto cam_src = s.register_source<CamIO>("cam");
    auto ser_src = s.register_source<SerIO>("ser");

    s.register_task<SerIO>("serial_rx", [&](SerIO::second_type&&) {
        static std::mutex ser_mtx;
        std::lock_guard<std::mutex> lk(ser_mtx);
        serial_rx.fetch_add(1);
    });

    s.register_task<CamIO>("tracker", [&, plan_signal](CamIO::second_type&& id) {
        {
            static std::mutex track_mtx;
            std::lock_guard<std::mutex> lk(track_mtx);
            auto t = make_moving_target(Clock::now(), 0.0, 2.0, 0.0, 4.0);
            t.this_id = id;
            target_swmr.write(t);
            tracked.fetch_add(1);
        }
        plan_signal->notify();
    });

    runtime::FramePlanLoop plan_loop(plan_signal);
    auto step_fn = [&]() {
        auto t = target_swmr.read();
        if (!t.check()) {
            return;
        }
        const auto cmd =
            aimer.very_aim(t, bullet_speed, AutoAimFsm::AIM_WHOLE_CAR_ARMOR, shoot, gimbal);
        if (cmd.appear && cmd.is_valid()) {
            solved.fetch_add(1);
        }
    };

    s.build();
    plan_loop.start(step_fn);
    s.run();

    constexpr int kCamFrames = 450; // 1.5s at 300 Hz
    std::atomic<bool> stop_ser { false };
    std::thread ser_producer([&]() {
        int seq = 0;
        while (!stop_ser.load()) {
            s.runtime_push_source<SerIO>(ser_src, [seq]() {
                return std::make_tuple(std::optional<int>(seq));
            });
            ++seq;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    for (int i = 1; i <= kCamFrames; ++i) {
        s.runtime_push_source<CamIO>(cam_src, [i]() {
            return std::make_tuple(std::optional<int>(i));
        });
        std::this_thread::sleep_for(std::chrono::microseconds(3300));
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (tracked.load() < kCamFrames && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    stop_ser.store(true);
    ser_producer.join();

    s.stop();
    plan_loop.stop();

    std::cout << "stress_tracked=" << tracked.load() << "/" << kCamFrames
              << " stress_solved=" << solved.load()
              << " stress_serial=" << serial_rx.load() << std::endl;
    require(tracked.load() == kCamFrames, "all 300 Hz camera frames must be tracked without TBB stall");
    require(solved.load() > 100, "dedicated FramePlanLoop must continuously solve MPC commands");
    require(serial_rx.load() > 100, "concurrent 1000 Hz serial TBB tasks must not starve");
}

void test_time_pose_buffer_extrapolation_guards() {
    auto make_yaw_pose = [](double yaw_deg) {
        ISO3 T = ISO3::Identity();
        T.linear() = Eigen::AngleAxisd(angles::from_degrees(yaw_deg), Vec3::UnitZ()).toRotationMatrix();
        return T;
    };
    auto yaw_deg_of = [](const ISO3& T) {
        return angles::to_degrees(std::atan2(T.linear()(1, 0), T.linear()(0, 0)));
    };

    const auto t0 = Clock::now();

    // 1. USB CDC microsecond burst must NOT produce 20-deg extrapolation spikes.
    // Smooth 100 deg/s sweep for 10ms, then an extra burst sample 10us later with +0.1 deg step
    // (which used to produce dq/dt = 10000 deg/s => +25 deg error over 2.5ms!).
    {
        utils::TimePoseBuffer buf;
        for (int ms = 0; ms <= 10; ++ms) {
            buf.push(t0 + std::chrono::milliseconds(ms), make_yaw_pose(0.1 * ms)); // 100 deg/s
        }
        // Duplicate timestamp must overwrite cleanly instead of leaving dt=0 duplicate
        buf.push(t0 + std::chrono::milliseconds(10), make_yaw_pose(1.0));
        // Burst packet arriving 10 microseconds later with a 0.1 deg step
        const auto t_burst = t0 + std::chrono::milliseconds(10) + std::chrono::microseconds(10);
        buf.push(t_burst, make_yaw_pose(1.1));

        // Query +2.5ms in the future: true 100 deg/s sweep would be ~1.35 deg, NOT +26 deg!
        const double extrap_yaw = yaw_deg_of(buf.get(t_burst + std::chrono::microseconds(2500)));
        std::cout << "tf_burst_extrap_yaw_deg=" << extrap_yaw << "\n";
        require(std::abs(extrap_yaw - 1.35) < 0.15,
                "TimePoseBuffer must use multi-ms baseline and avoid USB burst spikes");

        // Query +500ms in the future: must be bounded by kMaxExtrapolateFutureDt (20ms => <= ~3.5 deg)
        const double capped_yaw = yaw_deg_of(buf.get(t_burst + std::chrono::milliseconds(500)));
        std::cout << "tf_capped_future_yaw_deg=" << capped_yaw << "\n";
        require(capped_yaw > 2.5 && capped_yaw < 4.5,
                "TimePoseBuffer future extrapolation must be bounded");
    }

    // 2. Only two samples 50us apart (no >=4ms baseline): must safely hold latest pose (ZOH)
    {
        utils::TimePoseBuffer buf;
        buf.push(t0, make_yaw_pose(0.0));
        const auto t1 = t0 + std::chrono::microseconds(50);
        buf.push(t1, make_yaw_pose(0.2));
        const double held_yaw = yaw_deg_of(buf.get(t1 + std::chrono::milliseconds(5)));
        std::cout << "tf_short_baseline_hold_yaw_deg=" << held_yaw << "\n";
        require(std::abs(held_yaw - 0.2) < 1e-6,
                "TimePoseBuffer without >=4ms baseline must hold latest pose without extrapolating");
    }
}
} // namespace

int main() {
    test_mode_parsing_and_yaml_configs();
    test_reuse_vs_recompute_and_lookahead();
    test_frame_plan_loop_coalescing();
    test_tbb_work_stealing_no_deadlock();
    test_scheduler_with_frame_plan_loop_stress();
    test_time_pose_buffer_extrapolation_guards();
    std::cout << "PASS: very_aimer_mpc_mode_test\n";
    return 0;
}
