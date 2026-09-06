/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_NUI_SYSTEM_H_
#define XENIA_NUI_NUI_SYSTEM_H_

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "xenia/base/threading.h"
#include "xenia/nui/holt_smoother.h"
#include "xenia/nui/nui_source.h"
#include "xenia/nui/nui_types.h"

namespace xe {
namespace hid {
class InputSystem;
}  // namespace hid

namespace nui {

// HRESULTs of the NUI API (Kinect SDK v1 values; the guest-facing layer maps
// them to the values of the NUI library version linked into the title).
constexpr uint32_t kNuiOk = 0;
constexpr uint32_t kNuiErrorInvalidArg = 0x80070057;
constexpr uint32_t kNuiErrorPointer = 0x80004003;
constexpr uint32_t kNuiErrorNotImplemented = 0x80004001;
constexpr uint32_t kNuiErrorPending = 0x8000000A;
constexpr uint32_t kNuiErrorDeviceNotConnected = 0x8007048F;
constexpr uint32_t kNuiErrorDeviceNotReady = 0x80070015;
constexpr uint32_t kNuiErrorAlreadyInitialized = 0x800704DF;
constexpr uint32_t kNuiErrorFrameNoData = 0x83010001;
constexpr uint32_t kNuiErrorStreamNotEnabled = 0x83010003;
constexpr uint32_t kNuiErrorImageStreamInUse = 0x83010004;
constexpr uint32_t kNuiErrorFrameLimitExceeded = 0x83010005;
constexpr uint32_t kNuiErrorFeatureNotInitialized = 0x83010006;

constexpr int32_t kElevationMinimumDegrees = -27;
constexpr int32_t kElevationMaximumDegrees = 27;

// The emulated sensor: owns the data source and a 30 Hz pacer thread, and
// exposes the device model that the guest-facing NUI layer implements the
// title's API against. Kernel code never runs on the pacer thread; it only
// waits on published frames (from guest threads) and gets notified through
// listeners that run on the pacer thread and must not block.
class NuiSystem {
 public:
  using FrameListener = std::function<void()>;
  // Recording sink: runs on the pacer thread after every publish with the
  // published frame number, its timestamp, the skeleton frame and the source
  // frame it was built from (may be null). Must not block.
  using FrameSink =
      std::function<void(uint32_t frame_number, int64_t timestamp_us,
                         const SkeletonFrame& skeleton,
                         const std::shared_ptr<const SourceFrame>& source)>;

  struct Stats {
    uint64_t frames_published = 0;
    uint64_t source_frames_consumed = 0;
    uint64_t frames_repeated = 0;
    uint64_t skeleton_reads = 0;
    uint64_t skeleton_timeouts = 0;
    uint64_t image_reads = 0;
    uint64_t image_timeouts = 0;
    uint32_t tracked_bodies = 0;
    SourceStats source;
  };

  explicit NuiSystem(hid::InputSystem* input_system);
  ~NuiSystem();

  // Reads configuration; cheap. The source and the pacer thread start on the
  // first Initialize().
  bool Setup();
  void Shutdown();
  void Pause();
  void Resume();
  // The title was terminated: drop everything it set up.
  void ResetGuestState();

  // nui cvar.
  bool is_enabled() const { return enabled_; }
  // Whether titles should see a connected sensor.
  bool is_device_present() const {
    return enabled_ && device_present_ && !source_failed_;
  }
  // The guest-facing layer clears this when it cannot emulate the title's
  // NUI library version, so the title takes its "no sensor" path.
  void SetDevicePresent(bool present) { device_present_ = present; }

  // Declares that code outside the guest-facing NUI layer (a title-specific
  // hook layer that owns part of the runtime, see kernel/nui/nui_hle.cc)
  // reads skeletons or images from this system directly, without calling
  // Initialize / EnableSkeletonTracking / OpenImageStream. The source is
  // then asked for those planes unconditionally, so a foreign handler never
  // finds them empty. Cleared by ResetGuestState.
  void SetExternalConsumers(bool skeletons, bool images);

  // Title lifecycle (NuiInitialize / NuiShutdown).
  uint32_t Initialize(uint32_t init_flags);
  void Uninitialize();
  bool initialized() const { return initialized_; }
  uint32_t init_flags() const { return init_flags_; }

  // Skeleton tracking.
  uint32_t EnableSkeletonTracking(uint32_t tracking_flags);
  void DisableSkeletonTracking();
  bool skeleton_tracking_enabled() const { return skeleton_enabled_; }
  uint32_t skeleton_tracking_flags() const { return skeleton_flags_; }
  void SetTrackedSkeletons(uint32_t first_id, uint32_t second_id);
  // Blocks (bounded by |timeout_ms|) until a skeleton frame newer than
  // |last_frame_number| has been published, then copies it. Returns false on
  // timeout. Pass 0 as |last_frame_number| to accept the current frame.
  bool GetNextSkeletonFrame(uint32_t last_frame_number, uint32_t timeout_ms,
                            SkeletonFrame* out_frame);
  // NuiTransformSmooth: nullptr uses the SDK default parameters.
  void TransformSmooth(SkeletonFrame* frame, const SmoothParameters* params);

  // Image streams. Stream ids are host handles starting at 1.
  uint32_t OpenImageStream(ImageType type, ImageResolution resolution,
                           uint32_t stream_flags, uint32_t frame_limit,
                           uint32_t* out_stream_id);
  void CloseImageStream(uint32_t stream_id);
  bool SetImageStreamFlags(uint32_t stream_id, uint32_t stream_flags);
  bool GetImageStreamFlags(uint32_t stream_id, uint32_t* out_flags) const;
  bool GetNextImageFrame(uint32_t stream_id, uint32_t last_frame_number,
                         uint32_t timeout_ms, ImageFrame* out_frame);

  // Camera tilt (simulated motor).
  uint32_t SetElevationAngle(int32_t degrees);
  int32_t elevation_angle() const;
  int32_t elevation_target() const;
  bool elevation_moving() const;
  Vec4 normal_to_gravity() const;

  // Tracking id <-> user index bindings (XamUserNui*).
  void BindUser(uint32_t tracking_id, uint32_t user_index);
  void UnbindUser(uint32_t user_index);
  uint32_t GetUserIndexForTrackingId(uint32_t tracking_id) const;
  uint32_t GetTrackingIdForUserIndex(uint32_t user_index) const;
  // Tracking id of the tracked skeleton closest to the sensor centre line,
  // or kInvalidTrackingId.
  uint32_t GetBestTrackingId() const;

  // Listeners run on the pacer thread after every published frame.
  uint32_t AddFrameListener(FrameListener listener);
  void RemoveFrameListener(uint32_t listener_id);
  // Sinks run on the pacer thread after every published frame, after the
  // listeners, and receive the published data (recording).
  uint32_t AddFrameSink(FrameSink sink);
  void RemoveFrameSink(uint32_t sink_id);

  // Latest published skeleton frame (copy) and the source frame it was built
  // from (may be null). Returns false when nothing has been published yet.
  bool GetLatestFrames(SkeletonFrame* out_skeleton,
                       std::shared_ptr<const SourceFrame>* out_source) const;
  // The active source, or nullptr. Only for the UI thread and only for short,
  // thread-safe calls such as WebcamNuiSource::GetPreview: the pointer is
  // valid until the next RestartSource / Uninitialize / Shutdown, which the
  // UI thread itself triggers.
  NuiSource* source_for_ui();
  // Recreates the source from the current cvars (nui_source and the source
  // settings) and restarts it if it was running; used after settings
  // changes. Clears a previous source failure so the next attempt is made.
  void RestartSource();
  // Re-sends the device state (which includes the nui_camera_pitch,
  // nui_camera_height, nui_tilt_mode and nui_max_players cvars) to the
  // running source; cheap. Used after settings changes that do not need a
  // restart.
  void RefreshDeviceState();

  uint32_t current_frame_number() const {
    return frame_number_.load(std::memory_order_acquire);
  }
  int64_t current_timestamp_us() const {
    return timestamp_us_.load(std::memory_order_acquire);
  }
  Stats GetStats() const;
  std::string source_name() const;

 private:
  struct Slot {
    uint32_t person_key = 0;
    uint32_t tracking_id = kInvalidTrackingId;
  };
  struct ImageStream {
    uint32_t id = 0;
    ImageType type = ImageType::kDepthAndPlayerIndex;
    ImageResolution resolution = ImageResolution::k320x240;
    uint32_t flags = 0;
    uint32_t frame_limit = 2;
  };

  bool EnsureSourceStarted();
  void StartPacer();
  void StopPacer();
  void PacerThreadMain();
  void Tick();
  void BuildSkeletonFrame(const SourceFrame* source, SkeletonFrame* out,
                          std::array<uint8_t, kMaxSkeletons>* out_body_slots);
  void PushDeviceState();
  DeviceState BuildDeviceState() const;
  void RenderImageFrame(const std::shared_ptr<const SourceFrame>& source,
                        const std::array<uint8_t, kMaxSkeletons>& body_slots,
                        const ImageStream& stream, ImageFrame* out);
  void LogStatsIfDue();

  hid::InputSystem* input_system_ = nullptr;
  bool enabled_ = false;
  std::atomic<bool> device_present_{true};
  std::atomic<bool> source_failed_{false};

  // Device state written by guest threads, read by the pacer.
  mutable std::mutex state_mutex_;
  std::unique_ptr<NuiSource> source_;
  bool source_started_ = false;
  std::atomic<bool> initialized_{false};
  uint32_t init_flags_ = 0;
  std::atomic<bool> skeleton_enabled_{false};
  uint32_t skeleton_flags_ = 0;
  // Set by SetExternalConsumers: a foreign owner of part of the NUI runtime
  // may read skeletons/images without setting the state above.
  bool external_skeleton_consumers_ = false;
  bool external_image_consumers_ = false;
  uint32_t title_tracked_ids_[kMaxTrackedSkeletons] = {0, 0};
  std::vector<ImageStream> streams_;
  uint32_t next_stream_id_ = 1;
  float tilt_target_degrees_ = 0.0f;
  float tilt_current_degrees_ = 0.0f;
  std::array<Slot, kMaxSkeletons> slots_{};
  uint32_t next_tracking_id_ = 1;
  std::array<uint32_t, 8> user_bindings_{};  // user index -> tracking id

  // Published data.
  mutable std::mutex publish_mutex_;
  std::condition_variable publish_cv_;
  std::shared_ptr<const SourceFrame> latest_source_;
  SkeletonFrame latest_skeleton_;
  std::array<uint8_t, kMaxSkeletons> latest_body_slots_{};
  std::atomic<uint32_t> frame_number_{0};
  std::atomic<int64_t> timestamp_us_{0};
  uint64_t last_source_sequence_ = 0;
  std::chrono::steady_clock::time_point last_source_time_;

  // Pacer thread.
  std::unique_ptr<xe::threading::Thread> pacer_thread_;
  std::mutex control_mutex_;
  std::condition_variable control_cv_;
  bool pacer_stop_ = false;
  bool paused_ = false;

  // Listeners.
  mutable std::mutex listeners_mutex_;
  std::vector<std::pair<uint32_t, FrameListener>> listeners_;
  uint32_t next_listener_id_ = 1;
  std::vector<std::pair<uint32_t, FrameSink>> sinks_;
  uint32_t next_sink_id_ = 1;

  HoltSmoother smoother_;
  std::mutex smoother_mutex_;

  mutable std::mutex stats_mutex_;
  Stats stats_;
  std::chrono::steady_clock::time_point last_stats_log_;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_NUI_SYSTEM_H_
