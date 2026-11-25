#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace awakening::runtime {

// Dedicated latest-only frame-driven planning worker running outside Scheduler's
// TBB task_group.
// - Producers (tracker / armor_omni) call notify() without blocking TBB workers.
// - Multiple notifications arriving while a solve is in progress coalesce into a
//   single subsequent solve on the latest SWMR target (O(1) memory, no backlog).
// - Keeps MPC planning (and its internal TBB parallel_invoke) completely outside
//   Scheduler's task arena so TBB work-stealing can never re-enter task mutexes.
class FramePlanLoop {
public:
    struct Signal {
        std::mutex mutex;
        std::condition_variable cv;
        uint64_t seq = 0;
        bool stop = false;

        void notify() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++seq;
            }
            cv.notify_one();
        }

        void request_stop() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = true;
            }
            cv.notify_all();
        }
    };

    explicit FramePlanLoop(std::shared_ptr<Signal> signal = std::make_shared<Signal>()):
        signal_(signal ? std::move(signal) : std::make_shared<Signal>()) {}

    ~FramePlanLoop() {
        stop();
    }

    FramePlanLoop(const FramePlanLoop&) = delete;
    FramePlanLoop& operator=(const FramePlanLoop&) = delete;

    [[nodiscard]] std::shared_ptr<Signal> signal() const noexcept {
        return signal_;
    }

    void start(std::function<void()> step_fn) {
        stop();
        {
            std::lock_guard<std::mutex> lock(signal_->mutex);
            signal_->stop = false;
        }
        worker_ = std::thread([sig = signal_, fn = std::move(step_fn)]() {
            uint64_t seen_seq = 0;
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(sig->mutex);
                    sig->cv.wait(lock, [&]() { return sig->stop || sig->seq != seen_seq; });
                    if (sig->stop) {
                        break;
                    }
                    seen_seq = sig->seq;
                }
                if (fn) {
                    fn();
                }
            }
        });
    }

    void stop() {
        signal_->request_stop();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    std::shared_ptr<Signal> signal_;
    std::thread worker_;
};

} // namespace awakening::runtime
