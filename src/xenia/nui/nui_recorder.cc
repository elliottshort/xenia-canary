/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/nui_recorder.h"

#include <algorithm>
#include <cstring>

#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/nui/nui_flags.h"
#include "xenia/nui/nui_system.h"

// NuiRecorder: the live frame sink + writer thread. The .nuirec file format
// itself (NuirecWriter / NuirecReader) lives in nuirec_file.cc.

namespace xe {
namespace nui {

namespace {

constexpr size_t kMaskPixels = static_cast<size_t>(kDepthWidth) * kDepthHeight;
constexpr size_t kColorBytes =
    static_cast<size_t>(kColorWidth) * kColorHeight * 3;

}  // namespace

// NuiRecorder ----------------------------------------------------------------

NuiRecorder* NuiRecorder::Get() {
  static NuiRecorder instance;
  return &instance;
}

NuiRecorder::NuiRecorder() = default;

NuiRecorder::~NuiRecorder() {
  // The NuiSystem may already be gone at static destruction time: only stop
  // the writer thread, never touch the system.
  system_ = nullptr;
  sink_id_ = 0;
  Stop();
}

void NuiRecorder::Start(NuiSystem* system) {
  if (!system || !system->is_enabled()) {
    return;
  }
  if (thread_) {
    return;
  }
  system_ = system;
  active_path_ = cvars::nui_record_path;
  last_poll_ = std::chrono::steady_clock::now();
  frames_written_.store(0, std::memory_order_relaxed);
  frames_dropped_.store(0, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(ring_mutex_);
    stop_ = false;
    ring_head_ = 0;
    ring_count_ = 0;
  }
  xe::threading::Thread::CreationParameters params;
  params.stack_size = 512 * 1024;
  params.initial_priority = xe::threading::ThreadPriority::kLowest;
  thread_ = xe::threading::Thread::Create(params, [this]() {
    xe::threading::set_name("NUI Recorder");
    WriterThreadMain();
  });
  if (thread_) {
    // initial_priority is only honoured on POSIX; set it explicitly.
    thread_->set_priority(xe::threading::ThreadPriority::kLowest);
  }
  sink_id_ = system_->AddFrameSink(
      [this](uint32_t frame_number, int64_t timestamp_us,
             const SkeletonFrame& skeleton,
             const std::shared_ptr<const SourceFrame>& source) {
        OnFrame(frame_number, timestamp_us, skeleton, source);
      });
  if (!active_path_.empty()) {
    XELOGI("NUI recorder: recording to '{}'", xe::path_to_utf8(active_path_));
  }
}

void NuiRecorder::Stop() {
  if (system_ && sink_id_) {
    system_->RemoveFrameSink(sink_id_);
  }
  sink_id_ = 0;
  system_ = nullptr;
  std::unique_ptr<xe::threading::Thread> thread;
  {
    std::lock_guard<std::mutex> lock(ring_mutex_);
    stop_ = true;
    thread = std::move(thread_);
    ring_cv_.notify_all();
  }
  if (thread) {
    xe::threading::Wait(thread.get(), false);
  }
  // The thread drains the ring before exiting; anything still queued was
  // added after the sink was removed and is discarded.
  {
    std::lock_guard<std::mutex> lock(ring_mutex_);
    ring_head_ = 0;
    ring_count_ = 0;
  }
  CloseFile();
  active_path_.clear();
  failed_path_.clear();
}

bool NuiRecorder::is_recording() const {
  std::lock_guard<std::mutex> lock(status_mutex_);
  return !open_path_.empty();
}

std::filesystem::path NuiRecorder::current_path() const {
  std::lock_guard<std::mutex> lock(status_mutex_);
  return open_path_;
}

std::string NuiRecorder::last_error() const {
  std::lock_guard<std::mutex> lock(status_mutex_);
  return last_error_;
}

void NuiRecorder::SetError(const std::string& error) {
  std::lock_guard<std::mutex> lock(status_mutex_);
  last_error_ = error;
}

void NuiRecorder::PollRecordPath() {
  auto now = std::chrono::steady_clock::now();
  if (now - last_poll_ < std::chrono::seconds(1)) {
    return;
  }
  last_poll_ = now;
  std::filesystem::path path = cvars::nui_record_path;
  if (path == active_path_) {
    return;
  }
  active_path_ = path;
  if (path.empty()) {
    // Ask the writer thread to close the current file after the frames
    // already queued.
    std::lock_guard<std::mutex> lock(ring_mutex_);
    if (ring_count_ < kRingSize) {
      Entry& entry = ring_[(ring_head_ + ring_count_) % kRingSize];
      entry.kind = Entry::Kind::kClose;
      entry.path.clear();
      ++ring_count_;
      ring_cv_.notify_one();
    }
    XELOGI("NUI recorder: stopping");
  } else {
    XELOGI("NUI recorder: recording to '{}'", xe::path_to_utf8(path));
  }
}

void NuiRecorder::OnFrame(uint32_t frame_number, int64_t timestamp_us,
                          const SkeletonFrame& skeleton,
                          const std::shared_ptr<const SourceFrame>& source) {
  PollRecordPath();
  if (active_path_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(ring_mutex_);
  if (stop_) {
    return;
  }
  if (ring_count_ >= kRingSize) {
    frames_dropped_.fetch_add(1, std::memory_order_relaxed);
    uint64_t dropped = frames_dropped_.load(std::memory_order_relaxed);
    if (dropped == 1 || (dropped % 300) == 0) {
      XELOGW("NUI recorder: ring full, {} frames dropped so far (frame {})",
             dropped, frame_number);
    }
    return;
  }
  Entry& entry = ring_[(ring_head_ + ring_count_) % kRingSize];
  entry.kind = Entry::Kind::kFrame;
  entry.path = active_path_;
  NuirecFrame& frame = entry.frame;
  frame.Reset();
  frame.timestamp_us = timestamp_us;
  if (source) {
    // The source frame carries person keys and every body the source saw;
    // that is what playback needs.
    frame.body_count = std::min(source->body_count, kMaxSkeletons);
    for (uint32_t i = 0; i < frame.body_count; ++i) {
      frame.bodies[i] = source->bodies[i];
    }
    frame.floor_clip_plane = source->floor_clip_plane;
    frame.normal_to_gravity = source->normal_to_gravity;
    if (source->has_player_mask && source->player_mask.size() >= kMaskPixels) {
      frame.player_mask.assign(source->player_mask.begin(),
                               source->player_mask.begin() + kMaskPixels);
      frame.has_player_mask = true;
    }
    if (source->has_depth && source->depth_mm.size() >= kMaskPixels) {
      frame.depth_mm.assign(source->depth_mm.begin(),
                            source->depth_mm.begin() + kMaskPixels);
      frame.has_depth = true;
    }
    if (source->has_color && source->color_argb.size() >= kColorBytes / 3) {
      frame.color_rgb.resize(kColorBytes);
      const uint32_t* src = source->color_argb.data();
      uint8_t* dst = frame.color_rgb.data();
      for (size_t i = 0; i < kColorBytes / 3; ++i) {
        uint32_t argb = src[i];
        dst[i * 3 + 0] = static_cast<uint8_t>((argb >> 16) & 0xFF);
        dst[i * 3 + 1] = static_cast<uint8_t>((argb >> 8) & 0xFF);
        dst[i * 3 + 2] = static_cast<uint8_t>(argb & 0xFF);
      }
      frame.has_color = true;
    }
  } else {
    // No source frame (a "none" source or nothing new): keep the published
    // skeletons, with their tracking ids as person keys.
    for (const auto& body : skeleton.skeletons) {
      if (body.state == SkeletonState::kNotTracked ||
          frame.body_count >= kMaxSkeletons) {
        continue;
      }
      frame.bodies[frame.body_count++] = body;
    }
    frame.floor_clip_plane = skeleton.floor_clip_plane;
    frame.normal_to_gravity = skeleton.normal_to_gravity;
  }
  ++ring_count_;
  ring_cv_.notify_one();
}

void NuiRecorder::WriterThreadMain() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(ring_mutex_);
      ring_cv_.wait(lock, [&] { return ring_count_ > 0 || stop_; });
      if (ring_count_ == 0) {
        break;
      }
      // Swap keeps the ring entry's buffers for reuse.
      std::swap(ring_[ring_head_], scratch_);
      ring_head_ = (ring_head_ + 1) % kRingSize;
      --ring_count_;
    }
    ProcessEntry(&scratch_);
  }
  CloseFile();
}

bool NuiRecorder::OpenFileFor(const Entry& entry) {
  NuirecHeader header;
  header.flags = 0;
  if (entry.frame.has_player_mask) {
    header.flags |= kNuirecFlagMask;
  }
  if (entry.frame.has_depth) {
    header.flags |= kNuirecFlagDepth;
  }
  if (entry.frame.has_color) {
    header.flags |= kNuirecFlagColor;
  }
  header.fps = kFrameRateHz;
  header.camera_height_m = static_cast<float>(cvars::nui_camera_height);
  header.hfov_degrees = static_cast<float>(cvars::nui_camera_hfov);
  std::string error;
  if (!writer_.Open(entry.path, header, &error)) {
    XELOGE("NUI recorder: {}", error);
    SetError(error);
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    open_path_ = entry.path;
    last_error_.clear();
  }
  XELOGI("NUI recorder: opened '{}' (mask={} depth={} colour={})",
         xe::path_to_utf8(entry.path), header.has_mask(), header.has_depth(),
         header.has_color());
  return true;
}

void NuiRecorder::CloseFile() {
  if (writer_.is_open()) {
    XELOGI("NUI recorder: closed '{}' after {} frames",
           xe::path_to_utf8(writer_.path()), writer_.frames_written());
    writer_.Close();
  }
  std::lock_guard<std::mutex> lock(status_mutex_);
  open_path_.clear();
}

void NuiRecorder::ProcessEntry(Entry* entry) {
  if (entry->kind == Entry::Kind::kClose) {
    CloseFile();
    return;
  }
  if (writer_.is_open() && writer_.path() != entry->path) {
    CloseFile();
  }
  if (!writer_.is_open()) {
    if (!failed_path_.empty() && failed_path_ == entry->path) {
      // Already reported; drop frames until the cvar names another file.
      frames_dropped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    if (!OpenFileFor(*entry)) {
      failed_path_ = entry->path;
      frames_dropped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    failed_path_.clear();
  }
  std::string error;
  if (!writer_.WriteFrame(entry->frame, &error)) {
    XELOGE("NUI recorder: {}; stopping the recording", error);
    SetError(error);
    CloseFile();
    return;
  }
  frames_written_.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace nui
}  // namespace xe
