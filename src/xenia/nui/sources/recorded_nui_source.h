/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_SOURCES_RECORDED_NUI_SOURCE_H_
#define XENIA_NUI_SOURCES_RECORDED_NUI_SOURCE_H_

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "xenia/base/threading.h"
#include "xenia/nui/nui_recorder.h"
#include "xenia/nui/nui_source.h"

namespace xe {
namespace nui {

// Plays a .nuirec recording (made with nui_record_path) back as if it were a
// live sensor. A reader thread decodes frames ahead of time into a short
// queue; AcquireLatest hands them out either by their recorded timestamps
// (nui_playback_realtime) or one per call (deterministic). The recording is
// looped when nui_playback_loop is set. Person keys are the tracking ids
// stored in the file.
class RecordedNuiSource : public NuiSource {
 public:
  RecordedNuiSource();
  ~RecordedNuiSource() override;

  std::string_view name() const override { return "playback"; }
  bool Start(const DeviceState& initial_state) override;
  void Stop() override;
  void SetDeviceState(const DeviceState& state) override;
  std::shared_ptr<const SourceFrame> AcquireLatest(
      uint64_t last_sequence) override;
  void GetStats(SourceStats* out_stats) const override;

  // For tools: the recording being played and how many frames were handed
  // out (including repeats of a loop).
  const std::filesystem::path& path() const { return path_; }
  bool finished() const;

 private:
  struct Queued {
    std::shared_ptr<SourceFrame> frame;
    uint64_t play_time_us = 0;
  };
  static constexpr size_t kPrefetchDepth = 8;

  std::shared_ptr<SourceFrame> AllocateFrame();
  void ReaderThreadMain();
  static void Decode(const NuirecFrame& in, uint64_t play_time_us,
                     SourceFrame* out);
  void SetStatus(const std::string& status);

  std::filesystem::path path_;
  bool loop_ = true;
  bool realtime_ = true;
  uint64_t frame_period_us_ = kFramePeriodMicroseconds;

  std::mutex state_mutex_;
  DeviceState device_state_;
  bool started_ = false;

  // Reader thread state.
  NuirecReader reader_;
  NuirecFrame scratch_;
  std::vector<std::shared_ptr<SourceFrame>> pool_;
  std::unique_ptr<xe::threading::Thread> thread_;

  // Queue between the reader thread and AcquireLatest.
  mutable std::mutex queue_mutex_;
  std::condition_variable producer_cv_;
  std::condition_variable consumer_cv_;
  std::deque<Queued> queue_;
  bool stop_ = false;
  bool eof_ = false;

  // Consumer state (under queue_mutex_).
  std::shared_ptr<SourceFrame> latest_;
  uint64_t sequence_ = 0;
  bool clock_started_ = false;
  std::chrono::steady_clock::time_point clock_start_;
  uint64_t frames_produced_ = 0;
  uint64_t frames_skipped_ = 0;
  uint64_t loops_ = 0;

  mutable std::mutex status_mutex_;
  std::string status_;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_SOURCES_RECORDED_NUI_SOURCE_H_
