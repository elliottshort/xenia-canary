/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/nui_system.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "xenia/base/clock.h"
#include "xenia/base/logging.h"
#include "xenia/nui/camera_model.h"
#include "xenia/nui/nui_flags.h"

namespace xe {
namespace nui {

namespace {

constexpr float kTiltSlewDegreesPerSecond = 10.0f;
constexpr auto kSourceStaleTimeout = std::chrono::milliseconds(500);

bool IsUpperBodyJoint(Joint joint) {
  switch (joint) {
    case Joint::kShoulderCenter:
    case Joint::kHead:
    case Joint::kShoulderLeft:
    case Joint::kElbowLeft:
    case Joint::kWristLeft:
    case Joint::kHandLeft:
    case Joint::kShoulderRight:
    case Joint::kElbowRight:
    case Joint::kWristRight:
    case Joint::kHandRight:
      return true;
    default:
      return false;
  }
}

}  // namespace

NuiSystem::NuiSystem(hid::InputSystem* input_system)
    : input_system_(input_system) {
  for (auto& binding : user_bindings_) {
    binding = kInvalidTrackingId;
  }
  last_source_time_ = std::chrono::steady_clock::now();
  last_stats_log_ = last_source_time_;
}

NuiSystem::~NuiSystem() { Shutdown(); }

bool NuiSystem::Setup() {
  enabled_ = cvars::nui;
  if (!enabled_) {
    XELOGI("NUI: Kinect emulation disabled (nui=false)");
    return true;
  }
  XELOGI("NUI: Kinect emulation enabled, source={}", cvars::nui_source);
  return true;
}

void NuiSystem::Shutdown() {
  StopPacer();
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (source_ && source_started_) {
    source_->Stop();
    source_started_ = false;
  }
  source_.reset();
  initialized_ = false;
}

void NuiSystem::Pause() {
  std::lock_guard<std::mutex> lock(control_mutex_);
  paused_ = true;
  control_cv_.notify_all();
}

void NuiSystem::Resume() {
  std::lock_guard<std::mutex> lock(control_mutex_);
  paused_ = false;
  control_cv_.notify_all();
}

void NuiSystem::ResetGuestState() {
  Uninitialize();
  std::lock_guard<std::mutex> lock(state_mutex_);
  for (auto& binding : user_bindings_) {
    binding = kInvalidTrackingId;
  }
  for (auto& slot : slots_) {
    slot = Slot();
  }
  next_tracking_id_ = 1;
  tilt_target_degrees_ = 0.0f;
  tilt_current_degrees_ = 0.0f;
  device_present_ = true;
}

DeviceState NuiSystem::BuildDeviceState() const {
  // Caller holds state_mutex_.
  DeviceState state;
  state.init_flags = init_flags_;
  state.skeleton_tracking = skeleton_enabled_;
  state.seated_mode =
      (skeleton_flags_ & kTrackingEnableSeatedSupport) != 0;
  state.near_mode = (skeleton_flags_ & kTrackingEnableInNearRange) != 0;
  for (const auto& stream : streams_) {
    if (ImageTypeIsDepth(stream.type)) {
      state.want_depth = true;
      state.want_player_mask = true;
      if (stream.flags & kStreamEnableNearMode) {
        state.near_mode = true;
      }
    } else {
      state.want_color = true;
    }
  }
  // Skeleton tracking itself is derived from depth in the real sensor; ask
  // for the mask so hand-cursor libraries that read depth get consistent
  // data.
  if (skeleton_enabled_) {
    state.want_player_mask = true;
  }
  state.tilt_degrees =
      cvars::nui_tilt_mode == "ignore" ? 0.0f : tilt_current_degrees_;
  state.camera_pitch_degrees = static_cast<float>(cvars::nui_camera_pitch);
  state.camera_height_m = static_cast<float>(cvars::nui_camera_height);
  state.preferred_person_keys[0] = 0;
  state.preferred_person_keys[1] = 0;
  if (skeleton_flags_ & kTrackingTitleSetsTrackedSkeletons) {
    for (uint32_t i = 0; i < kMaxTrackedSkeletons; ++i) {
      for (const auto& slot : slots_) {
        if (slot.tracking_id != kInvalidTrackingId &&
            slot.tracking_id == title_tracked_ids_[i]) {
          state.preferred_person_keys[i] = slot.person_key;
        }
      }
    }
  }
  state.max_tracked_players = static_cast<uint32_t>(
      std::clamp(cvars::nui_max_players, 1, int(kMaxTrackedSkeletons)));
  return state;
}

void NuiSystem::PushDeviceState() {
  // Caller holds state_mutex_.
  if (source_ && source_started_) {
    source_->SetDeviceState(BuildDeviceState());
  }
}

bool NuiSystem::EnsureSourceStarted() {
  // Caller holds state_mutex_.
  if (source_started_) {
    return true;
  }
  if (!source_) {
    source_ = CreateNuiSource(cvars::nui_source);
    if (!source_) {
      XELOGE("NUI: unknown source '{}'", cvars::nui_source);
      source_failed_ = true;
      return false;
    }
  }
  if (!source_->Start(BuildDeviceState())) {
    XELOGE("NUI: source '{}' failed to start", source_->name());
    source_failed_ = true;
    return false;
  }
  source_started_ = true;
  source_failed_ = false;
  last_source_sequence_ = 0;
  last_source_time_ = std::chrono::steady_clock::now();
  XELOGI("NUI: source '{}' started", source_->name());
  return true;
}

uint32_t NuiSystem::Initialize(uint32_t init_flags) {
  if (!is_device_present()) {
    XELOGW("NUI: NuiInitialize(flags={:08X}) while no sensor is present "
           "(enabled={}, present={}, source_failed={})",
           init_flags, enabled_, device_present_.load(),
           source_failed_.load());
    return kNuiErrorDeviceNotConnected;
  }
  if (init_flags & ~kInitKnownMask) {
    XELOGW("NUI: NuiInitialize(flags={:08X}) has unknown flag bits", init_flags);
    return kNuiErrorInvalidArg;
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (initialized_) {
      return kNuiErrorAlreadyInitialized;
    }
    init_flags_ = init_flags;
    if (!EnsureSourceStarted()) {
      return kNuiErrorDeviceNotConnected;
    }
    initialized_ = true;
    PushDeviceState();
  }
  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    latest_source_.reset();
    latest_skeleton_ = SkeletonFrame();
    latest_body_slots_.fill(0);
  }
  StartPacer();
  XELOGI("NUI: NuiInitialize(flags={:08X})", init_flags);
  return kNuiOk;
}

void NuiSystem::Uninitialize() {
  StopPacer();
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!initialized_) {
    return;
  }
  XELOGI("NUI: NuiShutdown");
  initialized_ = false;
  init_flags_ = 0;
  skeleton_enabled_ = false;
  skeleton_flags_ = 0;
  title_tracked_ids_[0] = title_tracked_ids_[1] = 0;
  streams_.clear();
  if (source_ && source_started_) {
    source_->Stop();
    source_started_ = false;
  }
  std::lock_guard<std::mutex> smoother_lock(smoother_mutex_);
  smoother_.Reset();
}

uint32_t NuiSystem::EnableSkeletonTracking(uint32_t tracking_flags) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!initialized_ || !(init_flags_ & kInitSkeleton)) {
    return kNuiErrorFeatureNotInitialized;
  }
  skeleton_flags_ = tracking_flags;
  skeleton_enabled_ = true;
  PushDeviceState();
  XELOGI("NUI: skeleton tracking enabled (flags={:08X})", tracking_flags);
  return kNuiOk;
}

void NuiSystem::DisableSkeletonTracking() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  skeleton_enabled_ = false;
  skeleton_flags_ = 0;
  PushDeviceState();
}

void NuiSystem::SetTrackedSkeletons(uint32_t first_id, uint32_t second_id) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  title_tracked_ids_[0] = first_id;
  title_tracked_ids_[1] = second_id;
  PushDeviceState();
}

bool NuiSystem::GetNextSkeletonFrame(uint32_t last_frame_number,
                                     uint32_t timeout_ms,
                                     SkeletonFrame* out_frame) {
  std::unique_lock<std::mutex> lock(publish_mutex_);
  auto ready = [&] {
    return frame_number_.load(std::memory_order_relaxed) > last_frame_number;
  };
  if (!ready()) {
    if (timeout_ms == 0) {
      // fallthrough to the timeout path
    } else if (timeout_ms == 0xFFFFFFFFu) {
      // Never block forever: the emulator may be shutting down.
      publish_cv_.wait_for(lock, std::chrono::seconds(1), ready);
    } else {
      publish_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                           ready);
    }
  }
  {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    stats_.skeleton_reads++;
    if (!ready()) {
      stats_.skeleton_timeouts++;
    }
  }
  if (!ready()) {
    return false;
  }
  *out_frame = latest_skeleton_;
  return true;
}

void NuiSystem::TransformSmooth(SkeletonFrame* frame,
                                const SmoothParameters* params) {
  SmoothParameters p;
  if (cvars::nui_smoothing_override) {
    p.smoothing = static_cast<float>(cvars::nui_smoothing);
    p.correction = static_cast<float>(cvars::nui_correction);
    p.prediction = static_cast<float>(cvars::nui_prediction);
    p.jitter_radius = static_cast<float>(cvars::nui_jitter_radius);
    p.max_deviation_radius =
        static_cast<float>(cvars::nui_max_deviation_radius);
  } else if (params) {
    p = *params;
  }
  std::lock_guard<std::mutex> lock(smoother_mutex_);
  smoother_.Apply(frame, p);
}

uint32_t NuiSystem::OpenImageStream(ImageType type, ImageResolution resolution,
                                    uint32_t stream_flags,
                                    uint32_t frame_limit,
                                    uint32_t* out_stream_id) {
  if (!out_stream_id) {
    return kNuiErrorPointer;
  }
  *out_stream_id = 0;
  const bool is_depth = ImageTypeIsDepth(type);
  switch (type) {
    case ImageType::kDepthAndPlayerIndex:
      if (resolution != ImageResolution::k80x60 &&
          resolution != ImageResolution::k320x240) {
        return kNuiErrorInvalidArg;
      }
      break;
    case ImageType::kDepth:
      if (resolution != ImageResolution::k80x60 &&
          resolution != ImageResolution::k320x240 &&
          resolution != ImageResolution::k640x480) {
        return kNuiErrorInvalidArg;
      }
      break;
    case ImageType::kColor:
    case ImageType::kColorYuv:
    case ImageType::kColorRawYuv:
      if (resolution != ImageResolution::k640x480 &&
          resolution != ImageResolution::k1280x960) {
        return kNuiErrorInvalidArg;
      }
      break;
    default:
      return kNuiErrorInvalidArg;
  }
  if (frame_limit < 1 || frame_limit > 4) {
    return kNuiErrorFrameLimitExceeded;
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!initialized_) {
    return kNuiErrorDeviceNotReady;
  }
  const uint32_t required_flag =
      type == ImageType::kDepthAndPlayerIndex ? kInitDepthAndPlayerIndex
      : type == ImageType::kDepth             ? kInitDepth
                                              : kInitColor;
  if (!(init_flags_ & required_flag) &&
      !(is_depth && (init_flags_ & (kInitDepth | kInitDepthAndPlayerIndex)))) {
    return kNuiErrorFeatureNotInitialized;
  }
  for (const auto& stream : streams_) {
    if (stream.type == type) {
      return kNuiErrorImageStreamInUse;
    }
  }
  ImageStream stream;
  stream.id = next_stream_id_++;
  stream.type = type;
  stream.resolution = resolution;
  stream.flags = stream_flags;
  stream.frame_limit = frame_limit;
  streams_.push_back(stream);
  PushDeviceState();
  *out_stream_id = stream.id;
  XELOGI("NUI: image stream {} opened: type={} resolution={} flags={:08X}",
         stream.id, static_cast<uint32_t>(type),
         static_cast<uint32_t>(resolution), stream_flags);
  return kNuiOk;
}

void NuiSystem::CloseImageStream(uint32_t stream_id) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  streams_.erase(std::remove_if(streams_.begin(), streams_.end(),
                                [&](const ImageStream& stream) {
                                  return stream.id == stream_id;
                                }),
                 streams_.end());
  PushDeviceState();
}

bool NuiSystem::SetImageStreamFlags(uint32_t stream_id, uint32_t stream_flags) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  for (auto& stream : streams_) {
    if (stream.id == stream_id) {
      stream.flags = stream_flags;
      PushDeviceState();
      return true;
    }
  }
  return false;
}

bool NuiSystem::GetImageStreamFlags(uint32_t stream_id,
                                    uint32_t* out_flags) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  for (const auto& stream : streams_) {
    if (stream.id == stream_id) {
      *out_flags = stream.flags;
      return true;
    }
  }
  return false;
}

void NuiSystem::RenderImageFrame(
    const std::shared_ptr<const SourceFrame>& source,
    const std::array<uint8_t, kMaxSkeletons>& body_slots,
    const ImageStream& stream, ImageFrame* out) {
  out->type = stream.type;
  out->resolution = stream.resolution;
  out->width = ImageResolutionWidth(stream.resolution);
  out->height = ImageResolutionHeight(stream.resolution);
  out->flags = stream.flags;
  const size_t pixel_count = static_cast<size_t>(out->width) * out->height;
  if (ImageTypeIsDepth(stream.type)) {
    out->color_argb.clear();
    out->depth_mm.resize(pixel_count);
    out->player_index.resize(pixel_count);
    const bool have = source && source->has_depth &&
                      source->depth_mm.size() >= kDepthWidth * kDepthHeight;
    const bool have_mask =
        source && source->has_player_mask &&
        source->player_mask.size() >= kDepthWidth * kDepthHeight;
    // Player indices in the source mask are body indices; the title sees
    // skeleton slot numbers.
    uint8_t remap[kMaxSkeletons + 1] = {0};
    for (uint32_t i = 0; i < kMaxSkeletons; ++i) {
      remap[i + 1] = body_slots[i] ? body_slots[i] : 0;
    }
    for (uint32_t y = 0; y < out->height; ++y) {
      const uint32_t sy = y * kDepthHeight / out->height;
      for (uint32_t x = 0; x < out->width; ++x) {
        const uint32_t sx = x * kDepthWidth / out->width;
        const size_t si = static_cast<size_t>(sy) * kDepthWidth + sx;
        const size_t di = static_cast<size_t>(y) * out->width + x;
        uint16_t depth = have ? source->depth_mm[si] : 0;
        uint8_t player = have_mask ? source->player_mask[si] : 0;
        if (player > kMaxSkeletons) {
          player = 0;
        }
        out->depth_mm[di] = depth;
        out->player_index[di] =
            stream.type == ImageType::kDepthAndPlayerIndex ? remap[player] : 0;
      }
    }
  } else {
    out->depth_mm.clear();
    out->player_index.clear();
    out->color_argb.resize(pixel_count);
    const bool have = source && source->has_color &&
                      source->color_argb.size() >= kColorWidth * kColorHeight;
    if (!have) {
      std::fill(out->color_argb.begin(), out->color_argb.end(), 0xFF000000u);
    } else {
      for (uint32_t y = 0; y < out->height; ++y) {
        const uint32_t sy = y * kColorHeight / out->height;
        for (uint32_t x = 0; x < out->width; ++x) {
          const uint32_t sx = x * kColorWidth / out->width;
          out->color_argb[static_cast<size_t>(y) * out->width + x] =
              source->color_argb[static_cast<size_t>(sy) * kColorWidth + sx];
        }
      }
    }
  }
}

bool NuiSystem::GetNextImageFrame(uint32_t stream_id,
                                  uint32_t last_frame_number,
                                  uint32_t timeout_ms, ImageFrame* out_frame) {
  ImageStream stream;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    bool found = false;
    for (const auto& s : streams_) {
      if (s.id == stream_id) {
        stream = s;
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  std::shared_ptr<const SourceFrame> source;
  std::array<uint8_t, kMaxSkeletons> body_slots{};
  {
    std::unique_lock<std::mutex> lock(publish_mutex_);
    auto ready = [&] {
      return frame_number_.load(std::memory_order_relaxed) >
             last_frame_number;
    };
    if (!ready() && timeout_ms != 0) {
      auto wait = timeout_ms == 0xFFFFFFFFu
                      ? std::chrono::milliseconds(1000)
                      : std::chrono::milliseconds(timeout_ms);
      publish_cv_.wait_for(lock, wait, ready);
    }
    {
      std::lock_guard<std::mutex> stats_lock(stats_mutex_);
      stats_.image_reads++;
      if (!ready()) {
        stats_.image_timeouts++;
      }
    }
    if (!ready()) {
      return false;
    }
    source = latest_source_;
    body_slots = latest_body_slots_;
    out_frame->frame_number = frame_number_.load(std::memory_order_relaxed);
    out_frame->timestamp_us = timestamp_us_.load(std::memory_order_relaxed);
  }
  RenderImageFrame(source, body_slots, stream, out_frame);
  return true;
}

uint32_t NuiSystem::SetElevationAngle(int32_t degrees) {
  if (degrees < kElevationMinimumDegrees ||
      degrees > kElevationMaximumDegrees) {
    return kNuiErrorInvalidArg;
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!initialized_) {
    return kNuiErrorDeviceNotReady;
  }
  tilt_target_degrees_ = static_cast<float>(degrees);
  XELOGI("NUI: elevation angle requested: {} degrees", degrees);
  return kNuiOk;
}

int32_t NuiSystem::elevation_angle() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return static_cast<int32_t>(std::lround(tilt_current_degrees_));
}

int32_t NuiSystem::elevation_target() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return static_cast<int32_t>(std::lround(tilt_target_degrees_));
}

bool NuiSystem::elevation_moving() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return std::fabs(tilt_current_degrees_ - tilt_target_degrees_) > 0.01f;
}

Vec4 NuiSystem::normal_to_gravity() const {
  float tilt;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    tilt = tilt_current_degrees_;
  }
  Vec4 up;
  ComputeFloorPlane(0.0f, tilt, nullptr, &up);
  return up;
}

void NuiSystem::BindUser(uint32_t tracking_id, uint32_t user_index) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (user_index >= user_bindings_.size()) {
    return;
  }
  for (auto& binding : user_bindings_) {
    if (binding == tracking_id) {
      binding = kInvalidTrackingId;
    }
  }
  user_bindings_[user_index] = tracking_id;
}

void NuiSystem::UnbindUser(uint32_t user_index) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (user_index < user_bindings_.size()) {
    user_bindings_[user_index] = kInvalidTrackingId;
  }
}

uint32_t NuiSystem::GetUserIndexForTrackingId(uint32_t tracking_id) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (tracking_id == kInvalidTrackingId) {
    return kInvalidUserIndex;
  }
  for (uint32_t i = 0; i < user_bindings_.size(); ++i) {
    if (user_bindings_[i] == tracking_id) {
      return i;
    }
  }
  return kInvalidUserIndex;
}

uint32_t NuiSystem::GetTrackingIdForUserIndex(uint32_t user_index) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (user_index >= user_bindings_.size()) {
    return kInvalidTrackingId;
  }
  return user_bindings_[user_index];
}

uint32_t NuiSystem::GetBestTrackingId() const {
  std::lock_guard<std::mutex> lock(publish_mutex_);
  uint32_t best = kInvalidTrackingId;
  float best_score = 1e9f;
  for (const auto& skeleton : latest_skeleton_.skeletons) {
    if (skeleton.state != SkeletonState::kTracked) {
      continue;
    }
    float score = std::fabs(skeleton.position.x);
    if (score < best_score) {
      best_score = score;
      best = skeleton.tracking_id;
    }
  }
  return best;
}

uint32_t NuiSystem::AddFrameListener(FrameListener listener) {
  std::lock_guard<std::mutex> lock(listeners_mutex_);
  uint32_t id = next_listener_id_++;
  listeners_.emplace_back(id, std::move(listener));
  return id;
}

void NuiSystem::RemoveFrameListener(uint32_t listener_id) {
  std::lock_guard<std::mutex> lock(listeners_mutex_);
  listeners_.erase(std::remove_if(listeners_.begin(), listeners_.end(),
                                  [&](const auto& entry) {
                                    return entry.first == listener_id;
                                  }),
                   listeners_.end());
}

uint32_t NuiSystem::AddFrameSink(FrameSink sink) {
  std::lock_guard<std::mutex> lock(listeners_mutex_);
  uint32_t id = next_sink_id_++;
  sinks_.emplace_back(id, std::move(sink));
  return id;
}

void NuiSystem::RemoveFrameSink(uint32_t sink_id) {
  std::lock_guard<std::mutex> lock(listeners_mutex_);
  sinks_.erase(
      std::remove_if(sinks_.begin(), sinks_.end(),
                     [&](const auto& entry) { return entry.first == sink_id; }),
      sinks_.end());
}

bool NuiSystem::GetLatestFrames(
    SkeletonFrame* out_skeleton,
    std::shared_ptr<const SourceFrame>* out_source) const {
  std::lock_guard<std::mutex> lock(publish_mutex_);
  if (frame_number_.load(std::memory_order_relaxed) == 0) {
    return false;
  }
  if (out_skeleton) {
    *out_skeleton = latest_skeleton_;
  }
  if (out_source) {
    *out_source = latest_source_;
  }
  return true;
}

NuiSource* NuiSystem::source_for_ui() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return source_.get();
}

void NuiSystem::RestartSource() {
  bool was_started = false;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    // The nui cvar may have been changed by the settings UI; a title that is
    // already running keeps its hooks either way.
    enabled_ = cvars::nui;
    was_started = source_started_;
    if (source_ && source_started_) {
      source_->Stop();
      source_started_ = false;
    }
    source_.reset();
    source_failed_ = false;
    // Person keys of the new source instance may collide with the old ones:
    // treat everybody as having left the play space.
    for (auto& slot : slots_) {
      if (slot.person_key != 0) {
        for (auto& binding : user_bindings_) {
          if (binding == slot.tracking_id) {
            binding = kInvalidTrackingId;
          }
        }
      }
      slot = Slot();
    }
    last_source_sequence_ = 0;
    if (was_started) {
      if (EnsureSourceStarted()) {
        PushDeviceState();
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    latest_source_.reset();
  }
  XELOGI("NUI: source restarted (nui_source={}, running={})", cvars::nui_source,
         was_started);
}

void NuiSystem::RefreshDeviceState() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  PushDeviceState();
}

NuiSystem::Stats NuiSystem::GetStats() const {
  Stats stats;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats = stats_;
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (source_) {
    source_->GetStats(&stats.source);
  }
  return stats;
}

std::string NuiSystem::source_name() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return source_ ? std::string(source_->name()) : std::string("none");
}

void NuiSystem::StartPacer() {
  std::lock_guard<std::mutex> lock(control_mutex_);
  if (pacer_thread_) {
    return;
  }
  pacer_stop_ = false;
  xe::threading::Thread::CreationParameters params;
  params.stack_size = 256 * 1024;
  pacer_thread_ = xe::threading::Thread::Create(params, [this]() {
    xe::threading::set_name("NUI Pacer");
    PacerThreadMain();
  });
}

void NuiSystem::StopPacer() {
  std::unique_ptr<xe::threading::Thread> thread;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!pacer_thread_) {
      return;
    }
    pacer_stop_ = true;
    control_cv_.notify_all();
    thread = std::move(pacer_thread_);
  }
  xe::threading::Wait(thread.get(), false);
}

void NuiSystem::PacerThreadMain() {
  auto next_tick = std::chrono::steady_clock::now();
  while (true) {
    {
      std::unique_lock<std::mutex> lock(control_mutex_);
      control_cv_.wait_until(lock, next_tick,
                             [&] { return pacer_stop_ || paused_; });
      if (pacer_stop_) {
        return;
      }
      if (paused_) {
        control_cv_.wait(lock, [&] { return pacer_stop_ || !paused_; });
        if (pacer_stop_) {
          return;
        }
        next_tick = std::chrono::steady_clock::now();
        continue;
      }
    }
    double scalar = xe::Clock::guest_time_scalar();
    if (scalar <= 0.0) {
      scalar = 1.0;
    }
    auto period = std::chrono::microseconds(
        static_cast<int64_t>(kFramePeriodMicroseconds / scalar));
    period = std::clamp(period, std::chrono::microseconds(1000),
                        std::chrono::microseconds(1000000));
    next_tick += period;
    auto now = std::chrono::steady_clock::now();
    if (next_tick < now - period * 4) {
      // Fell far behind (debugger, heavy load): resynchronize.
      next_tick = now + period;
    }
    Tick();
  }
}

void NuiSystem::Tick() {
  std::shared_ptr<const SourceFrame> source;
  bool have_new_source = false;
  float tilt_before = 0.0f;
  float tilt_after = 0.0f;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!initialized_ || !source_ || !source_started_) {
      return;
    }
    // Simulated tilt motor.
    tilt_before = tilt_current_degrees_;
    if (tilt_current_degrees_ != tilt_target_degrees_) {
      float step = kTiltSlewDegreesPerSecond * kFramePeriodMicroseconds /
                   1000000.0f;
      float delta = tilt_target_degrees_ - tilt_current_degrees_;
      if (std::fabs(delta) <= step) {
        tilt_current_degrees_ = tilt_target_degrees_;
      } else {
        tilt_current_degrees_ += delta > 0.0f ? step : -step;
      }
    }
    tilt_after = tilt_current_degrees_;
    if (tilt_after != tilt_before) {
      PushDeviceState();
    }
    source = source_->AcquireLatest(last_source_sequence_);
    if (source) {
      have_new_source = true;
      last_source_sequence_ = source->sequence;
      last_source_time_ = std::chrono::steady_clock::now();
    }
  }

  SkeletonFrame frame;
  std::array<uint8_t, kMaxSkeletons> body_slots{};
  const auto now = std::chrono::steady_clock::now();
  const bool stale = !have_new_source &&
                     (now - last_source_time_) > kSourceStaleTimeout;
  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    if (!have_new_source) {
      source = latest_source_;
    }
  }
  if (stale) {
    // The source stopped delivering: everybody left the play space.
    source.reset();
  }
  BuildSkeletonFrame(source.get(), &frame, &body_slots);

  const uint32_t frame_number = frame_number_.load() + 1;
  const int64_t timestamp = timestamp_us_.load() + kFramePeriodMicroseconds;
  frame.frame_number = frame_number;
  frame.timestamp_us = timestamp;
  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    latest_source_ = source;
    latest_skeleton_ = frame;
    latest_body_slots_ = body_slots;
    timestamp_us_.store(timestamp, std::memory_order_release);
    frame_number_.store(frame_number, std::memory_order_release);
  }
  publish_cv_.notify_all();

  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.frames_published++;
    if (have_new_source) {
      stats_.source_frames_consumed++;
    } else {
      stats_.frames_repeated++;
    }
    uint32_t tracked = 0;
    for (const auto& skeleton : frame.skeletons) {
      if (skeleton.state == SkeletonState::kTracked) {
        ++tracked;
      }
    }
    stats_.tracked_bodies = tracked;
  }

  std::vector<std::pair<uint32_t, FrameListener>> listeners;
  std::vector<std::pair<uint32_t, FrameSink>> sinks;
  {
    std::lock_guard<std::mutex> lock(listeners_mutex_);
    listeners = listeners_;
    sinks = sinks_;
  }
  for (auto& entry : listeners) {
    entry.second();
  }
  for (auto& entry : sinks) {
    entry.second(frame_number, timestamp, frame, source);
  }
  LogStatsIfDue();
}

void NuiSystem::BuildSkeletonFrame(
    const SourceFrame* source, SkeletonFrame* out,
    std::array<uint8_t, kMaxSkeletons>* out_body_slots) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  *out = SkeletonFrame();
  out_body_slots->fill(0);

  const bool seated = (skeleton_flags_ & kTrackingEnableSeatedSupport) != 0;
  ComputeFloorPlane(static_cast<float>(cvars::nui_camera_height),
                    tilt_current_degrees_, &out->floor_clip_plane,
                    &out->normal_to_gravity);
  if (seated) {
    out->flags |= 0x8;  // NUI_SKELETON_FRAME_FLAG_SEATED_SUPPORT_ENABLED
  }

  // Release slots of bodies that are gone, keep slots of bodies present.
  std::array<bool, kMaxSkeletons> slot_present{};
  const uint32_t body_count =
      source ? std::min(source->body_count, kMaxSkeletons) : 0;
  for (uint32_t b = 0; b < body_count; ++b) {
    const Skeleton& body = source->bodies[b];
    if (body.state == SkeletonState::kNotTracked ||
        body.tracking_id == kInvalidTrackingId) {
      continue;
    }
    for (uint32_t s = 0; s < kMaxSkeletons; ++s) {
      if (slots_[s].person_key == body.tracking_id) {
        slot_present[s] = true;
      }
    }
  }
  for (uint32_t s = 0; s < kMaxSkeletons; ++s) {
    if (!slot_present[s] && slots_[s].person_key != 0) {
      // Person left: any user binding to it is void.
      for (auto& binding : user_bindings_) {
        if (binding == slots_[s].tracking_id) {
          binding = kInvalidTrackingId;
        }
      }
      slots_[s] = Slot();
    }
  }

  // Assign slots to new bodies (lowest free slot) and decide which are fully
  // tracked.
  const uint32_t max_tracked = static_cast<uint32_t>(
      std::clamp(cvars::nui_max_players, 1, int(kMaxTrackedSkeletons)));
  const bool title_selects =
      (skeleton_flags_ & kTrackingTitleSetsTrackedSkeletons) != 0;
  uint32_t tracked_count = 0;
  for (uint32_t b = 0; b < body_count; ++b) {
    const Skeleton& body = source->bodies[b];
    if (body.state == SkeletonState::kNotTracked ||
        body.tracking_id == kInvalidTrackingId) {
      continue;
    }
    int slot_index = -1;
    for (uint32_t s = 0; s < kMaxSkeletons; ++s) {
      if (slots_[s].person_key == body.tracking_id) {
        slot_index = static_cast<int>(s);
        break;
      }
    }
    if (slot_index < 0) {
      for (uint32_t s = 0; s < kMaxSkeletons; ++s) {
        if (slots_[s].person_key == 0) {
          slots_[s].person_key = body.tracking_id;
          slots_[s].tracking_id = next_tracking_id_++;
          slot_index = static_cast<int>(s);
          break;
        }
      }
    }
    if (slot_index < 0) {
      continue;  // more than six bodies
    }
    (*out_body_slots)[b] = static_cast<uint8_t>(slot_index + 1);
    Skeleton& dst = out->skeletons[slot_index];
    dst = body;
    dst.tracking_id = slots_[slot_index].tracking_id;
    dst.enrollment_index = kInvalidUserIndex;
    dst.user_index = kInvalidUserIndex;
    for (uint32_t u = 0; u < user_bindings_.size(); ++u) {
      if (user_bindings_[u] == dst.tracking_id) {
        dst.user_index = u;
      }
    }
    bool fully_tracked = body.state == SkeletonState::kTracked;
    if (fully_tracked && skeleton_enabled_) {
      if (title_selects) {
        fully_tracked = dst.tracking_id == title_tracked_ids_[0] ||
                        dst.tracking_id == title_tracked_ids_[1];
      } else {
        fully_tracked = tracked_count < max_tracked;
      }
    } else {
      fully_tracked = false;
    }
    if (fully_tracked) {
      ++tracked_count;
      dst.state = SkeletonState::kTracked;
      if (seated) {
        for (uint32_t j = 0; j < kJointCount; ++j) {
          if (!IsUpperBodyJoint(static_cast<Joint>(j))) {
            dst.joints[j] = Vec4();
            dst.joint_states[j] = JointState::kNotTracked;
          }
        }
        dst.position =
            dst.joints[static_cast<size_t>(Joint::kShoulderCenter)];
      }
    } else {
      dst.state = SkeletonState::kPositionOnly;
      for (uint32_t j = 0; j < kJointCount; ++j) {
        dst.joints[j] = Vec4();
        dst.joint_states[j] = JointState::kNotTracked;
      }
      dst.quality_flags = 0;
    }
  }

  // Auto-bind the first tracked player to user 0.
  if (cvars::nui_auto_bind_user0) {
    bool user0_bound = user_bindings_[0] != kInvalidTrackingId;
    if (!user0_bound) {
      for (auto& skeleton : out->skeletons) {
        if (skeleton.state == SkeletonState::kTracked) {
          user_bindings_[0] = skeleton.tracking_id;
          skeleton.user_index = 0;
          break;
        }
      }
    }
  }
}

void NuiSystem::LogStatsIfDue() {
  if (!cvars::nui_log_stats) {
    return;
  }
  auto now = std::chrono::steady_clock::now();
  if (now - last_stats_log_ < std::chrono::seconds(5)) {
    return;
  }
  last_stats_log_ = now;
  Stats stats = GetStats();
  XELOGI(
      "NUI stats: published={} source={} repeated={} skeleton reads={} "
      "(timeouts {}) image reads={} (timeouts {}) tracked={} source_fps={:.1f} "
      "inference={:.1f}ms {}",
      stats.frames_published, stats.source_frames_consumed,
      stats.frames_repeated, stats.skeleton_reads, stats.skeleton_timeouts,
      stats.image_reads, stats.image_timeouts, stats.tracked_bodies,
      stats.source.capture_fps, stats.source.inference_ms,
      stats.source.status);
}

}  // namespace nui
}  // namespace xe
