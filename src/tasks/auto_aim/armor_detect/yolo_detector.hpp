#pragma once

#include "tasks/auto_aim/type.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace awakening::auto_aim {

// YOLOv8 姿态模型（来自 srm_auto_aim）的 WUST 适配层。
// 输入：BGR 图像 + ROI；输出：带 net 关键点的 Armor，坐标为原图坐标。
// 数字分类和颜色复核仍交给 ArmorDetector 现有流程。
class YoloDetector {
public:
    explicit YoloDetector(const YAML::Node& config);
    ~YoloDetector() noexcept;

    std::vector<Armor> detect(const cv::Mat& bgr_img, const cv::Rect& roi);

    int input_w() const noexcept;
    int input_h() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace awakening::auto_aim
