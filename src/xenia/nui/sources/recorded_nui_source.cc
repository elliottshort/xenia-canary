/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/sources/recorded_nui_source.h"

#include <algorithm>

#include "xenia/base/logging.h"
#include "xenia/base/string.h"
#include "xenia/nui/nui_flags.h"

namespace xe {
namespace nui {

namespace {

constexpr size_t kMaskPixels = static_cast<size_t>(kDepthWidth) * kDepthHeight;
constexpr size_t kColorPixels = static_cast<size_t>(kColorWidth) * kColorHeight;
// Person key for a recorded body that carries no tracking id.
constexpr uint32_t kFallbackKeyBase = 0x80000000u;

}  // namespace

RecordedNuiSource::RecordedNuiSource() = default;

RecordedNuiSource::~RecordedNuiSource() { Stop(); }

bool RecordedNuiSource::Start(const DeviceState& initial_state) {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  if (started_) {
    return true;
  }
  device_state_ = initial_state;
  path_ = cvars::nui_playback_path;
  loop_ = cvars::nui_playback_loop;
  realtime_ = cvars::nui_playback_realtime;
  if (path_.empty()) {
    XELOGE("NUI playback: nui_playback_path is not set");
    SetStatus("no recording (nui_playback_path is empty)");
    return false;
  }
  std::string error;
  if (!reader_.Open(path_, &error)) {
    XELOGE("NUI playback: {}", error);
    SetStatus(error);
    return false;
  }
  const NuirecHeader& header = reader_.header();
  frame_period_us_ = 1000000ull / std::max<uint32_t>(1, header.fps);
  XELOGI(
      "NUI playback: '{}' ({} fps, mask={} depth={} colour={}, height {:.2f} "
      "m, hfov {:.0f}), {} {}",
      xe::path_to_utf8(path_), header.fps, header.has_mask(),
      header.has_depth(), header.has_color(), header.camera_height_m,
      header.hfov_degrees, realtime_ ? "realtime" : "frame-stepped",
      loop_ ? "looping" : "once");
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = false;
    eof_ = false;
    queue_.clear();
    latest_.reset();
    sequence_ = 0;
    clock_started_ = false;
    frames_produced_ = 0;
    frames_skipped_ = 0;
    loops_ = 0;
  }
  SetStatus(fmt::format("playing '{}'", xe::path_to_utf8(path_.filename())));
  xe::threading::Thread::CreationParameters params;
  params.stack_size = 512 * 1024;
  thread_ = xe::threading::Thread::Create(params, [this]() {
    xe::threading::set_name("NUI Playback");
    ReaderThreadMain();
  });
  started_ = true;
  return true;
}

void RecordedNuiSource::Stop() {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  if (!started_) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
    producer_cv_.notify_all();
    consumer_cv_.notify_all();
  }
  if (thread_) {
    xe::threading::Wait(thread_.get(), false);
    thread_.reset();
  }
  reader_.Close();
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.clear();
    latest_.reset();
  }
  pool_.clear();
  started_ = false;
  SetStatus("stopped");
}

void RecordedNuiSource::SetDeviceState(const DeviceState& state) {
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  device_state_ = state;
}

bool RecordedNuiSource::finished() const {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return eof_ && queue_.empty();
}

void RecordedNuiSource::SetStatus(const std::string& status) {
  std::lock_guard<std::mutex> lock(status_mutex_);
  status_ = status;
}

std::shared_ptr<SourceFrame> RecordedNuiSource::AllocateFrame() {
  // Reader thread only. A frame is free when neither the queue, latest_ nor
  // the consumer references it.
  for (auto& frame : pool_) {
    if (frame.use_count() == 1) {
      frame->Reset();
      return frame;
    }
  }
  auto frame = std::make_shared<SourceFrame>();
  pool_.push_back(frame);
  return frame;
}

void RecordedNuiSource::Decode(const NuirecFrame& in, uint64_t play_time_us,
                               SourceFrame* out) {
  out->capture_time_us = play_time_us;
  out->body_count = std::min(in.body_count, kMaxSkeletons);
  for (uint32_t i = 0; i < out->body_count; ++i) {
    out->bodies[i] = in.bodies[i];
    if (out->bodies[i].tracking_id == kInvalidTrackingId) {
      out->bodies[i].tracking_id = kFallbackKeyBase + i;
    }
  }
  out->floor_clip_plane = in.floor_clip_plane;
  out->normal_to_gravity = in.normal_to_gravity;
  if (in.has_player_mask && in.player_mask.size() >= kMaskPixels) {
    out->player_mask.assign(in.player_mask.begin(),
                            in.player_mask.begin() + kMaskPixels);
    out->has_player_mask = true;
  }
  if (in.has_depth && in.depth_mm.size() >= kMaskPixels) {
    out->depth_mm.assign(in.depth_mm.begin(),
                         in.depth_mm.begin() + kMaskPixels);
    out->has_depth = true;
  }
  if (in.has_color && in.color_rgb.size() >= kColorPixels * 3) {
    out->color_argb.resize(kColorPixels);
    const uint8_t* src = in.color_rgb.data();
    uint32_t* dst = out->color_argb.data();
    for (size_t i = 0; i < kColorPixels; ++i) {
      dst[i] = 0xFF000000u | (static_cast<uint32_t>(src[i * 3 + 0]) << 16) |
               (static_cast<uint32_t>(src[i * 3 + 1]) << 8) |
               static_cast<uint32_t>(src[i * 3 + 2]);
    }
    out->has_color = true;
  }
}

void RecordedNuiSource::ReaderThreadMain() {
  uint64_t loop_offset_us = 0;
  uint64_t pass_first_ts = 0;
  uint64_t pass_last_ts = 0;
  uint64_t frames_in_pass = 0;
  std::string error;
  while (true) {
    if (!reader_.ReadFrame(&scratch_, &error)) {
      if (!error.empty()) {
        XELOGE("NUI playback: {}", error);
        SetStatus(error);
        std::lock_guard<std::mutex> lock(queue_mutex_);
        eof_ = true;
        consumer_cv_.notify_all();
        return;
      }
      if (frames_in_pass == 0) {
        XELOGW("NUI playback: '{}' contains no frames",
               xe::path_to_utf8(path_));
        SetStatus("recording is empty");
        std::lock_guard<std::mutex> lock(queue_mutex_);
        eof_ = true;
        consumer_cv_.notify_all();
        return;
      }
      if (!loop_) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        eof_ = true;
        consumer_cv_.notify_all();
        return;
      }
      if (!reader_.Rewind(&error)) {
        XELOGE("NUI playback: {}", error);
        SetStatus(error);
        std::lock_guard<std::mutex> lock(queue_mutex_);
        eof_ = true;
        consumer_cv_.notify_all();
        return;
      }
      loop_offset_us += (pass_last_ts - pass_first_ts) + frame_period_us_;
      frames_in_pass = 0;
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        ++loops_;
      }
      continue;
    }
    uint64_t ts =
        static_cast<uint64_t>(std::max<int64_t>(0, scratch_.timestamp_us));
    if (frames_in_pass == 0) {
      pass_first_ts = ts;
      pass_last_ts = ts;
    }
    // Tolerate a non-monotonic timestamp: never go backwards.
    ts = std::max(ts, pass_first_ts);
    pass_last_ts = std::max(pass_last_ts, ts);
    ++frames_in_pass;
    uint64_t play_time_us = loop_offset_us + (ts - pass_first_ts);

    auto frame = AllocateFrame();
    Decode(scratch_, play_time_us, frame.get());

    std::unique_lock<std::mutex> lock(queue_mutex_);
    producer_cv_.wait(lock,
                      [&] { return stop_ || queue_.size() < kPrefetchDepth; });
    if (stop_) {
      return;
    }
    queue_.push_back({std::move(frame), play_time_us});
    consumer_cv_.notify_one();
  }
}

std::shared_ptr<const SourceFrame> RecordedNuiSource::AcquireLatest(
    uint64_t last_sequence) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  if (stop_) {
    return nullptr;
  }
  std::shared_ptr<SourceFrame> picked;
  if (realtime_) {
    auto now = std::chrono::steady_clock::now();
    if (!clock_started_) {
      clock_started_ = true;
      clock_start_ = now;
    }
    uint64_t elapsed_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now -
                                                              clock_start_)
            .count());
    while (!queue_.empty() && queue_.front().play_time_us <= elapsed_us) {
      if (picked) {
        ++frames_skipped_;
      }
      picked = std::move(queue_.front().frame);
      queue_.pop_front();
    }
  } else {
    if (queue_.empty() && !eof_) {
      // The reader is normally several frames ahead; give it a moment so
      // that every tick advances the recording by exactly one frame.
      consumer_cv_.wait_for(lock, std::chrono::milliseconds(10),
                            [&] { return stop_ || eof_ || !queue_.empty(); });
      if (stop_) {
        return nullptr;
      }
    }
    if (!queue_.empty()) {
      picked = std::move(queue_.front().frame);
      queue_.pop_front();
    }
  }
  if (picked) {
    producer_cv_.notify_one();
    picked->sequence = ++sequence_;
    latest_ = std::move(picked);
    ++frames_produced_;
  }
  if (latest_ && latest_->sequence > last_sequence) {
    return latest_;
  }
  return nullptr;
}

void RecordedNuiSource::GetStats(SourceStats* out_stats) const {
  *out_stats = SourceStats();
  out_stats->capture_fps =
      1000000.0 / static_cast<double>(std::max<uint64_t>(1, frame_period_us_));
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    out_stats->frames_produced = frames_produced_;
    out_stats->frames_dropped = frames_skipped_;
    std::string status;
    {
      std::lock_guard<std::mutex> status_lock(status_mutex_);
      status = status_;
    }
    if (eof_ && queue_.empty()) {
      status += " (end of recording)";
    } else if (loops_ > 0) {
      status += fmt::format(" (loop {})", loops_ + 1);
    }
    out_stats->status = std::move(status);
  }
}

}  // namespace nui
}  // namespace xe
