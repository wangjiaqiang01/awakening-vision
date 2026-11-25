#pragma once
#include "tasks/auto_aim/armor_track/armor_target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/base/common.hpp"
#include "utils/impl.hpp"
#include <memory>
#include <yaml-cpp/node/node.h>

namespace awakening::auto_aim {
enum class MpcMode {
    VisionD,
    Awakening,
};

struct VeryAimDebugInfo {
    bool valid = false;
    bool is_same = false;
    bool this_id_same = false;
    bool fsm_same = false;
    bool bullet_speed_same = false;
    bool shoot_pose_same = false;
    bool gimbal_pose_same = false;
    double time_in_traj_s = 0.0;
};

class VeryAimer {
public:
    VeryAimer(const YAML::Node& config);
    AWAKENING_IMPL_DEFINITION(VeryAimer)
    [[nodiscard]] GimbalCmd very_aim(
        const ArmorTarget& target,
        double bullet_speed,
        const AutoAimFsm& fsm,
        const ISO3& shoot_in_gimbal_odom,
        const ISO3& gimbal_in_gimbal_odom
    );
    std::pair<double, double> get_yaw_pitch_offset();
    void set_operator_offset(std::pair<double, double> offset);
    [[nodiscard]] VeryAimDebugInfo get_debug_info() const;
    [[nodiscard]] bool uses_vision_d_mpc() const noexcept;
    [[nodiscard]] MpcMode mpc_mode() const noexcept;
    [[nodiscard]] static const char* mpc_mode_name(MpcMode mode) noexcept;
};
} // namespace awakening::auto_aim