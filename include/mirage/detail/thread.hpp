#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace mirage::detail {

class ThreadPool {
  public:
  explicit ThreadPool(int n) {
    workers_.reserve(n);
    for (int i = 0; i < n; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock lk(mtx_);
            cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
          {
            std::unique_lock lk(mtx_);
            if (--pending_ == 0) done_cv_.notify_all();
          }
        }
      });
    }
  }

  void run(std::function<void(int)> fn, int n) {
    if (n <= 1) {
      fn(0);
      return;
    }
    {
      std::unique_lock lk(mtx_);
      pending_ = n;
      for (int i = 0; i < n; ++i) tasks_.push([fn, i] { fn(i); });
    }
    cv_.notify_all();
    std::unique_lock lk(mtx_);
    done_cv_.wait(lk, [this] { return pending_ == 0; });
  }

  ~ThreadPool() {
    {
      std::unique_lock lk(mtx_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

  private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mtx_;
  std::condition_variable cv_, done_cv_;
  int pending_ = 0;
  bool stop_ = false;
};
}  // namespace mirage::detail
