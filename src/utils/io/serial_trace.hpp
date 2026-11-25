#pragma once

#include "utils/drivers/telemetry_protocol.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace awakening {

// 串口双向 trace + 录像帧时间戳。
// 所有事件使用同一个 steady_clock 时间基准；wall_time 只用于人读。
// TX/RX 由 SerialDriver 的 IO 线程调用，video 由录像任务调用，内部加锁。
class SerialTrace {
public:
    SerialTrace(std::string trace_path, std::string video_path)
        : trace_path_(std::move(trace_path)), video_path_(std::move(video_path)) {
        namespace fs = std::filesystem;
        if (const auto parent = fs::path(trace_path_).parent_path(); !parent.empty()) {
            fs::create_directories(parent);
        }
        out_.open(trace_path_, std::ios::out | std::ios::trunc);
        if (!out_) {
            throw std::runtime_error("cannot open serial trace: " + trace_path_);
        }
        out_ << "mono_ns,wall_time,event,seq,yaw_deg,pitch_deg,roll_deg,mode,robot_id,"
                "bullet_speed,fire,video_frame_index,capture_mono_ns,raw_hex,note\n";
        meta("trace=" + trace_path_ + ";video=" + video_path_);
        out_.flush();
    }

    ~SerialTrace() {
        std::lock_guard<std::mutex> lock(mutex_);
        out_.flush();
    }

    void log_tx(const std::vector<uint8_t>& bytes) {
        const auto mono = mono_ns();
        const auto wall = wall_iso();
        float yaw = 0.0f, pitch = 0.0f;
        int32_t fire = 0;
        if (bytes.size() == 38) {
            std::memcpy(&yaw, bytes.data() + 4, sizeof(float));
            std::memcpy(&pitch, bytes.data() + 8, sizeof(float));
            std::memcpy(&fire, bytes.data() + 14, sizeof(int32_t));
        }
        std::vector<std::string> fields(15);
        fields[0] = std::to_string(mono);
        fields[1] = wall;
        fields[2] = "tx";
        fields[3] = std::to_string(tx_seq_++);
        fields[4] = std::to_string(yaw);
        fields[5] = std::to_string(pitch);
        fields[10] = std::to_string(fire);
        fields[13] = hex(bytes);
        write_row(fields);
    }

    void log_rx(
        const TongjiTelemetry& telemetry,
        const uint8_t* raw,
        std::size_t size
    ) {
        const auto mono = mono_ns();
        const auto wall = wall_iso();
        std::vector<std::string> fields(15);
        fields[0] = std::to_string(mono);
        fields[1] = wall;
        fields[2] = "rx";
        fields[3] = std::to_string(rx_seq_++);
        fields[4] = std::to_string(telemetry.yaw);
        fields[5] = std::to_string(telemetry.pitch);
        fields[6] = std::to_string(telemetry.roll);
        fields[7] = std::to_string(telemetry.mode);
        fields[8] = std::to_string(telemetry.robot_id);
        fields[9] = std::to_string(telemetry.bullet_speed);
        fields[13] = hex_bytes(raw, size);
        write_row(fields);
    }

    template <typename TimePoint>
    void log_video(const TimePoint& capture_time, uint64_t frame_index) {
        const auto mono = mono_ns();
        const auto wall = wall_iso();
        const auto capture_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                capture_time.time_since_epoch()
            ).count();
        std::vector<std::string> fields(15);
        fields[0] = std::to_string(mono);
        fields[1] = wall;
        fields[2] = "video";
        fields[3] = std::to_string(video_seq_++);
        fields[11] = std::to_string(frame_index);
        fields[12] = std::to_string(capture_ns);
        fields[14] = video_path_;
        write_row(fields);
    }

    // 给录像画面叠加人读时间；与 trace 的 wall_time 使用同一格式。
    static std::string wall_now() { return wall_iso(); }

private:
    static int64_t mono_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    static std::string wall_iso() {
        const auto now = std::chrono::system_clock::now();
        const auto t = std::chrono::system_clock::to_time_t(now);
        std::tm tm {};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch())
                .count() % 1000000;
        std::ostringstream out;
        out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << '.' << std::setw(6)
            << std::setfill('0') << us;
        return out.str();
    }

    static std::string hex(const std::vector<uint8_t>& bytes) {
        return hex_bytes(bytes.data(), bytes.size());
    }

    static std::string hex_bytes(const uint8_t* data, std::size_t size) {
        if (data == nullptr || size == 0) return {};
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (std::size_t i = 0; i < size; ++i) {
            if (i) out << ' ';
            out << std::setw(2) << static_cast<int>(data[i]);
        }
        return out.str();
    }

    void meta(const std::string& note) {
        std::vector<std::string> fields(15);
        fields[0] = std::to_string(mono_ns());
        fields[1] = wall_iso();
        fields[2] = "meta";
        fields[3] = "0";
        fields[14] = note;
        write_row(fields);
    }

    void write_row(const std::vector<std::string>& fields) {
        std::string row;
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i) row += ',';
            row += fields[i];
        }
        write(row);
    }

    void write(const std::string& row) {
        std::lock_guard<std::mutex> lock(mutex_);
        out_ << row << '\n';
        // 1 kHz RX 不逐行 flush；按 100 ms 刷新，崩溃时最多丢 100 ms。
        const auto now = std::chrono::steady_clock::now();
        if (now - last_flush_ >= std::chrono::milliseconds(100)) {
            out_.flush();
            last_flush_ = now;
        }
    }

    std::string trace_path_;
    std::string video_path_;
    std::ofstream out_;
    std::mutex mutex_;
    std::chrono::steady_clock::time_point last_flush_ {};
    uint64_t tx_seq_ = 0;
    uint64_t rx_seq_ = 0;
    uint64_t video_seq_ = 0;
};
} // namespace awakening
