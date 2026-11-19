#include "yolo_detector.hpp"

#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

#ifdef USE_OPENVINO
    #include <openvino/openvino.hpp>
#endif

namespace awakening::auto_aim {

#ifdef USE_OPENVINO
struct YoloDetector::Impl {
    struct Params {
        std::string model_path;
        std::string device_name = "CPU";
        double conf_threshold = 0.35;
        double nms_threshold = 0.3;
        int input_w = 416;
        int input_h = 416;
        int class_num = 2;

        void load(const YAML::Node& config) {
            model_path = replace_root_dir(config["model_path"].as<std::string>());
            if (config["device_name"]) {
                device_name = config["device_name"].as<std::string>();
            }
            if (config["conf_threshold"]) {
                conf_threshold = config["conf_threshold"].as<double>();
            }
            if (config["nms_threshold"]) {
                nms_threshold = config["nms_threshold"].as<double>();
            }
            if (config["input_w"]) {
                input_w = config["input_w"].as<int>();
            }
            if (config["input_h"]) {
                input_h = config["input_h"].as<int>();
            }
            if (config["class_num"]) {
                class_num = config["class_num"].as<int>();
            }
            if (input_w != 416 || input_h != 416) {
                throw std::runtime_error("YoloDetector: only 416x416 YOLOv8 is supported");
            }
            if (class_num != 2) {
                throw std::runtime_error("YoloDetector: expect YOLOv8 with 2 classes (blue/red)");
            }
        }
    } params;

    ov::Core core;
    ov::CompiledModel compiled_model;
    int input_w = 416;
    int input_h = 416;

    Impl(const YAML::Node& config) {
        params.load(config);
        input_w = params.input_w;
        input_h = params.input_h;

        auto model = core.read_model(params.model_path);
        ov::preprocess::PrePostProcessor ppp(model);
        auto& input = ppp.input();
        input.tensor()
            .set_element_type(ov::element::u8)
            .set_shape({ 1, 416, 416, 3 })
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        input.model().set_layout("NCHW");
        input.preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale(255.0);
        model = ppp.build();
        compiled_model = core.compile_model(
            model,
            params.device_name,
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
        );
        AWAKENING_INFO("YOLOv8 detector loaded: {}", params.model_path);
    }

    static void sort_keypoints(std::vector<cv::Point2f>& keypoints) {
        if (keypoints.size() != 4) {
            return;
        }
        std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
        std::vector<cv::Point2f> top = { keypoints[0], keypoints[1] };
        std::vector<cv::Point2f> bottom = { keypoints[2], keypoints[3] };
        std::sort(top.begin(), top.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });
        std::sort(bottom.begin(), bottom.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });
        keypoints[0] = top[0];     // left top
        keypoints[1] = top[1];     // right top
        keypoints[2] = bottom[1];  // right bottom
        keypoints[3] = bottom[0];  // left bottom
    }

    std::vector<Armor> detect(const cv::Mat& bgr_img, const cv::Rect& roi) {
        if (bgr_img.empty()) {
            return {};
        }
        cv::Rect crop = roi & cv::Rect(0, 0, bgr_img.cols, bgr_img.rows);
        if (crop.width <= 1 || crop.height <= 1) {
            return {};
        }
        const cv::Mat src = bgr_img(crop);

        const double x_scale = static_cast<double>(params.input_h) / src.rows;
        const double y_scale = static_cast<double>(params.input_w) / src.cols;
        const double scale = std::min(x_scale, y_scale);
        int resized_h = static_cast<int>(std::lround(src.rows * scale));
        int resized_w = static_cast<int>(std::lround(src.cols * scale));
        resized_h = std::max(1, std::min(resized_h, params.input_h));
        resized_w = std::max(1, std::min(resized_w, params.input_w));

        cv::Mat input_img(params.input_h, params.input_w, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::resize(src, input_img(cv::Rect(0, 0, resized_w, resized_h)), cv::Size(resized_w, resized_h));

        ov::Tensor input_tensor(
            ov::element::u8,
            { 1, static_cast<size_t>(params.input_h), static_cast<size_t>(params.input_w), 3 },
            input_img.data
        );
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        auto output_tensor = infer_request.get_output_tensor(0);        const auto shape = output_tensor.get_shape();
        if (shape.size() < 3) {
            return {};
        }
        const int output_rows = static_cast<int>(shape[shape.size() - 2]);
        const int output_cols = static_cast<int>(shape[shape.size() - 1]);
        if (output_rows < 4 + params.class_num || output_cols <= 0) {
            return {};
        }

        cv::Mat output(output_rows, output_cols, CV_32F, output_tensor.data<float>());
        cv::Mat transposed;
        cv::transpose(output, transposed);
        if (transposed.cols < 4 + params.class_num + 8) {
            return {};
        }

        std::vector<int> ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<cv::Point2f>> all_keypoints;
        const cv::Point2f offset(static_cast<float>(crop.x), static_cast<float>(crop.y));

        for (int r = 0; r < transposed.rows; ++r) {
            cv::Mat row = transposed.row(r);
            cv::Mat scores = row.colRange(4, 4 + params.class_num);
            double score = 0;
            cv::Point class_id;
            cv::minMaxLoc(scores, nullptr, &score, nullptr, &class_id);
            if (score < params.conf_threshold) {
                continue;
            }

            const float cx = row.at<float>(0, 0) / static_cast<float>(scale) + offset.x;
            const float cy = row.at<float>(0, 1) / static_cast<float>(scale) + offset.y;
            const float w = row.at<float>(0, 2) / static_cast<float>(scale);
            const float h = row.at<float>(0, 3) / static_cast<float>(scale);
            const int left = static_cast<int>(std::lround(cx - 0.5f * w));
            const int top = static_cast<int>(std::lround(cy - 0.5f * h));
            const int width = std::max(1, static_cast<int>(std::lround(w)));
            const int height = std::max(1, static_cast<int>(std::lround(h)));

            std::vector<cv::Point2f> keypoints;
            keypoints.reserve(4);
            for (int i = 0; i < 4; ++i) {
                const float px = row.at<float>(0, 4 + params.class_num + i * 2)
                    / static_cast<float>(scale) + offset.x;
                const float py = row.at<float>(0, 4 + params.class_num + i * 2 + 1)
                    / static_cast<float>(scale) + offset.y;
                keypoints.emplace_back(px, py);
            }
            sort_keypoints(keypoints);

            ids.push_back(class_id.x);
            confidences.push_back(static_cast<float>(score));
            boxes.emplace_back(left, top, width, height);
            all_keypoints.emplace_back(std::move(keypoints));
        }

        if (confidences.empty()) {
            return {};
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, confidences, params.conf_threshold, params.nms_threshold, keep);

        std::vector<Armor> armors;
        armors.reserve(keep.size());
        for (int index: keep) {
            const auto& points = all_keypoints[index];
            if (points.size() != 4) {
                continue;
            }
            Armor armor;
            armor.net.emplace(Armor::NetCtx{});
            armor.net->confidence = confidences[index];
            armor.net->color = (ids[index] == 0) ? ArmorColor::BLUE : ArmorColor::RED;
            armor.net->number = ArmorClass::UNKNOWN;

            // srm 顺序 [LT, RT, RB, LB] -> WUST [LT, LB, RB, RT]
            auto& wust_points = armor.net->key_points.points;
            wust_points[ArmorKeyPointsIndex::LEFT_TOP] = points[0];
            wust_points[ArmorKeyPointsIndex::LEFT_BOTTOM] = points[3];
            wust_points[ArmorKeyPointsIndex::RIGHT_BOTTOM] = points[2];
            wust_points[ArmorKeyPointsIndex::RIGHT_TOP] = points[1];
            armors.emplace_back(std::move(armor));
        }
        return armors;
    }
};
#else
struct YoloDetector::Impl {
    int input_w = 416;
    int input_h = 416;
    Impl(const YAML::Node&) {
        throw std::runtime_error("YoloDetector requires USE_OPENVINO");
    }
    std::vector<Armor> detect(const cv::Mat&, const cv::Rect&) {
        return {};
    }
};
#endif

YoloDetector::YoloDetector(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
YoloDetector::~YoloDetector() noexcept {
    _impl.reset();
}
std::vector<Armor> YoloDetector::detect(const cv::Mat& bgr_img, const cv::Rect& roi) {
    return _impl->detect(bgr_img, roi);
}
int YoloDetector::input_w() const noexcept {
    return _impl->input_w;
}
int YoloDetector::input_h() const noexcept {
    return _impl->input_h;
}

} // namespace awakening::auto_aim
