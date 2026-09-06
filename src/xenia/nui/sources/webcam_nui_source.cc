/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/sources/webcam_nui_source.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <utility>

#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/nui/camera_model.h"
#include "xenia/nui/nui_flags.h"
#include "xenia/nui/sources/webcam_remap_lut.h"

namespace xe {
namespace nui {

namespace {

constexpr uint32_t kCaptureTimeoutMs = 100;
// Capture-rate statistics window.
constexpr uint64_t kCaptureStatsWindowUs = 500000;
constexpr uint32_t kMaxLandmarkPersons = 2;
// Pause between attempts to open a camera that is missing or in use.
constexpr auto kCameraRetryInterval = std::chrono::seconds(3);
// How often Stop() re-closes the camera while waiting for the capture
// thread, in case an Open began after the previous Close.
constexpr auto kStopPollInterval = std::chrono::milliseconds(100);

uint64_t NowUs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Mirrors an RGBA frame in place (Kinect images are mirror views).
void FlipHorizontal(CameraFrame* frame) {
  if (!frame->width || !frame->height) {
    return;
  }
  const size_t stride = frame->stride ? frame->stride : frame->width * 4;
  for (uint32_t y = 0; y < frame->height; ++y) {
    uint32_t* row =
        reinterpret_cast<uint32_t*>(frame->rgba.data() + y * stride);
    std::reverse(row, row + frame->width);
  }
}

// Dimensions of a segmentation buffer: the working resolution, or a square
// model output. Returns false for unknown layouts.
bool SegmentationDimensions(size_t size, uint32_t* out_w, uint32_t* out_h) {
  if (size == static_cast<size_t>(kDepthWidth) * kDepthHeight) {
    *out_w = kDepthWidth;
    *out_h = kDepthHeight;
    return true;
  }
  const uint32_t side =
      static_cast<uint32_t>(std::sqrt(static_cast<double>(size)));
  if (side && static_cast<size_t>(side) * side == size) {
    *out_w = side;
    *out_h = side;
    return true;
  }
  return false;
}

std::string CameraSelectorForLog() {
  return cvars::nui_camera.empty() ? std::string("<first>") : cvars::nui_camera;
}

}  // namespace

WebcamNuiSource::WebcamNuiSource()
    : mask_lut_(std::make_unique<WebcamRemapLut>()),
      color_lut_(std::make_unique<WebcamRemapLut>()) {}

WebcamNuiSource::~WebcamNuiSource() { Stop(); }

bool WebcamNuiSource::Start(const DeviceState& initial_state) {
  if (running_) {
    return true;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    device_state_ = initial_state;
  }
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = SourceStats();
    camera_ready_ = false;
    camera_error_.clear();
    camera_name_.clear();
    estimator_ready_ = false;
    estimator_error_.clear();
    estimator_description_.clear();
    capture_count_ = 0;
    capture_window_start_us_ = 0;
    capture_window_count_ = 0;
    UpdateStatusLocked();
  }
  tracker_.Reset();
  skeleton_synthesizer_.Reset();
  last_synthesis_timestamp_us_ = 0;

  // The capture object is created here (cheap) so that Stop() can always
  // reach it; the device itself is opened on the capture thread.
  camera_ = CameraCapture::Create();
  if (!camera_) {
    XELOGE("NUI webcam: camera capture is not available on this platform");
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.status = "camera capture not available on this platform";
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    latest_.reset();
    sequence_ = 0;
  }
  {
    std::lock_guard<std::mutex> lock(preview_mutex_);
    preview_valid_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    captured_fresh_ = false;
  }

  running_ = true;
  xe::threading::Thread::CreationParameters params;
  params.stack_size = 256 * 1024;
  capture_thread_ =
      xe::threading::Thread::Create(params, [this]() { CaptureThreadMain(); });
  if (!capture_thread_) {
    XELOGE("NUI webcam: failed to create the capture thread");
    running_ = false;
    camera_.reset();
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.status = "failed to create the capture thread";
    return false;
  }
  capture_thread_->set_name("NUI Capture");
  params.stack_size = 4 * 1024 * 1024;
  inference_thread_ = xe::threading::Thread::Create(
      params, [this]() { InferenceThreadMain(); });
  if (!inference_thread_) {
    XELOGE("NUI webcam: failed to create the inference thread");
    Stop();
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.status = "failed to create the inference thread";
    return false;
  }
  inference_thread_->set_name("NUI Inference");
  return true;
}

void WebcamNuiSource::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
  }
  stop_cv_.notify_all();
  {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    captured_fresh_ = false;
  }
  capture_cv_.notify_all();
  if (capture_thread_) {
    // The capture thread may be blocked in CameraCapture::Open (up to its
    // first-frame timeout) or ReadFrame; Close() wakes both. It is repeated
    // while waiting in case an Open began just after the previous Close.
    while (true) {
      if (camera_) {
        camera_->Close();
      }
      if (xe::threading::Wait(capture_thread_.get(), false,
                              kStopPollInterval) ==
          xe::threading::WaitResult::kSuccess) {
        break;
      }
    }
    capture_thread_.reset();
  }
  if (inference_thread_) {
    // An estimator creation in progress cannot be interrupted; it finishes
    // and the thread exits.
    xe::threading::Wait(inference_thread_.get(), false);
    inference_thread_.reset();
  }
  if (camera_) {
    camera_->Close();
    camera_.reset();
  }
  estimator_.reset();
  {
    std::lock_guard<std::mutex> lock(preview_mutex_);
    preview_valid_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    camera_ready_ = false;
    estimator_ready_ = false;
    stats_.capture_fps = 0.0;
    stats_.status = "stopped";
  }
}

void WebcamNuiSource::SetDeviceState(const DeviceState& state) {
  std::lock_guard<std::mutex> lock(mutex_);
  device_state_ = state;
}

std::shared_ptr<const SourceFrame> WebcamNuiSource::AcquireLatest(
    uint64_t last_sequence) {
  std::lock_guard<std::mutex> lock(publish_mutex_);
  if (!latest_ || latest_->sequence <= last_sequence) {
    return nullptr;
  }
  return latest_;
}

void WebcamNuiSource::GetStats(SourceStats* out_stats) const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  *out_stats = stats_;
}

WebcamNuiSource::State WebcamNuiSource::state() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return StateLocked();
}

WebcamNuiSource::State WebcamNuiSource::StateLocked() const {
  if (!running_) {
    return State::kStopped;
  }
  if (!camera_ready_) {
    return State::kStartingCamera;
  }
  if (!estimator_ready_) {
    return State::kLoadingModels;
  }
  if (!estimator_error_.empty()) {
    return State::kCameraOnly;
  }
  return State::kRunning;
}

void WebcamNuiSource::UpdateStatusLocked() {
  if (!camera_ready_) {
    stats_.status = camera_error_.empty()
                        ? std::string("starting camera")
                        : "camera error: " + camera_error_ + " (retrying)";
  } else if (!estimator_ready_) {
    stats_.status = "loading models";
  } else if (!estimator_error_.empty()) {
    stats_.status =
        "no pose estimation (" + estimator_error_ + "); camera only";
  } else if (estimator_health_ != PoseEstimator::BackendHealth::kOk &&
             !estimator_status_.empty()) {
    // The estimator lost its device and is rebuilding, or came back on the
    // CPU; it says so itself and goes back to kOk once it has recovered.
    stats_.status = estimator_status_ + " (" + camera_name_ + ")";
  } else {
    stats_.status = estimator_description_ + " / " + camera_name_;
  }
}

bool WebcamNuiSource::GetPreview(Preview* out_preview) {
  preview_requested_.store(true, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(preview_mutex_);
  if (!preview_valid_) {
    return false;
  }
  *out_preview = preview_;
  return true;
}

std::shared_ptr<SourceFrame> WebcamNuiSource::AllocateFrame() {
  // Called from the inference thread with publish_mutex_ held: a frame is
  // free when only the pool references it (neither latest_ nor a consumer).
  for (auto& frame : pool_) {
    if (frame.use_count() == 1) {
      frame->Reset();
      return frame;
    }
  }
  auto frame = std::make_shared<SourceFrame>();
  frame->player_mask.resize(static_cast<size_t>(kDepthWidth) * kDepthHeight);
  frame->depth_mm.resize(static_cast<size_t>(kDepthWidth) * kDepthHeight);
  frame->color_argb.resize(static_cast<size_t>(kColorWidth) * kColorHeight);
  pool_.push_back(frame);
  return frame;
}

bool WebcamNuiSource::OpenCamera(std::string* out_error) {
  CameraCapture::Options camera_options;
  camera_options.device = cvars::nui_camera;
  camera_options.width =
      static_cast<uint32_t>(std::max(cvars::nui_capture_width, 160));
  camera_options.height =
      static_cast<uint32_t>(std::max(cvars::nui_capture_height, 120));
  camera_options.fps =
      static_cast<uint32_t>(std::max(cvars::nui_capture_fps, 1));
  out_error->clear();
  // Blocks until the first frame or the capture's own open timeout; Stop()
  // interrupts it through CameraCapture::Close.
  return camera_->Open(camera_options, out_error);
}

void WebcamNuiSource::WaitForRetry() {
  std::unique_lock<std::mutex> lock(mutex_);
  stop_cv_.wait_for(lock, kCameraRetryInterval,
                    [this]() { return !running_.load(); });
}

void WebcamNuiSource::CaptureThreadMain() {
  CameraFrame frame;
  std::string error;
  // The reason of the last logged failure; identical repeats stay quiet.
  std::string logged_error;
  bool camera_open = false;
  while (running_) {
    if (!camera_open) {
      if (!OpenCamera(&error)) {
        if (!running_) {
          break;
        }
        if (error != logged_error) {
          XELOGW("NUI webcam: cannot open camera '{}': {}; retrying every {} s",
                 CameraSelectorForLog(), error,
                 std::chrono::duration_cast<std::chrono::seconds>(
                     kCameraRetryInterval)
                     .count());
          logged_error = error;
        }
        {
          std::lock_guard<std::mutex> lock(stats_mutex_);
          camera_ready_ = false;
          camera_error_ = error;
          UpdateStatusLocked();
        }
        WaitForRetry();
        continue;
      }
      camera_open = true;
      logged_error.clear();
      XELOGI("NUI webcam: opened '{}' at {}x{} @ {:.1f} fps",
             camera_->device_name(), camera_->width(), camera_->height(),
             camera_->fps());
      {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        camera_ready_ = true;
        camera_error_.clear();
        camera_name_ = camera_->device_name();
        capture_window_start_us_ = 0;
        capture_window_count_ = 0;
        UpdateStatusLocked();
      }
    }

    error.clear();
    if (!camera_->ReadFrame(&frame, kCaptureTimeoutMs, &error)) {
      if (error.empty()) {
        // Timeout.
        continue;
      }
      if (!running_) {
        break;
      }
      // Device error (unplugged, taken over by another application): drop
      // the device and go back to opening it.
      XELOGW("NUI webcam: capture error: {}; reopening the camera", error);
      camera_->Close();
      camera_open = false;
      logged_error = error;
      {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        camera_ready_ = false;
        camera_error_ = error;
        stats_.capture_fps = 0.0;
        stats_.frames_dropped++;
        UpdateStatusLocked();
      }
      WaitForRetry();
      continue;
    }
    if (!frame.width || !frame.height || frame.rgba.empty()) {
      continue;
    }
    if (!frame.timestamp_us) {
      frame.timestamp_us = NowUs();
    }
    {
      std::lock_guard<std::mutex> lock(capture_mutex_);
      std::swap(captured_, frame);
      if (captured_fresh_) {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.frames_dropped++;
      }
      captured_fresh_ = true;
    }
    capture_cv_.notify_one();

    // Capture rate over half-second windows.
    const uint64_t now = NowUs();
    std::lock_guard<std::mutex> lock(stats_mutex_);
    capture_count_++;
    capture_window_count_++;
    if (!capture_window_start_us_) {
      capture_window_start_us_ = now;
    } else if (now - capture_window_start_us_ >= kCaptureStatsWindowUs) {
      stats_.capture_fps = capture_window_count_ * 1000000.0 /
                           static_cast<double>(now - capture_window_start_us_);
      capture_window_start_us_ = now;
      capture_window_count_ = 0;
    }
  }
}

void WebcamNuiSource::CreateEstimator() {
  // Failure leaves a camera-only source (colour stream works, nobody is
  // ever tracked) so the user can still see the preview.
  PoseEstimator::Options estimator_options;
  if (!cvars::nui_model_path.empty()) {
    estimator_options.model_dir = cvars::nui_model_path;
  } else {
    estimator_options.model_dir =
        xe::filesystem::GetExecutableFolder() / "nui" / "models";
  }
  estimator_options.quality = cvars::nui_model_quality;
  estimator_options.execution_provider = cvars::nui_execution_provider;
  estimator_options.threads = cvars::nui_inference_threads;
  estimator_options.max_persons = static_cast<uint32_t>(std::clamp(
      cvars::nui_max_players, 1, static_cast<int32_t>(kMaxLandmarkPersons)));
  estimator_options.want_segmentation = cvars::nui_segmentation;
  std::string error;
  estimator_ = PoseEstimator::Create(estimator_options, &error);
  std::string description;
  if (estimator_) {
    description = fmt::format("{} / {}", estimator_->backend_name(),
                              estimator_->model_name());
    XELOGI("NUI webcam: pose estimation on {} with {} (models in {})",
           estimator_->backend_name(), estimator_->model_name(),
           xe::path_to_utf8(estimator_options.model_dir));
  } else {
    XELOGE(
        "NUI webcam: pose estimation unavailable: {}. Models are expected in "
        "{} and onnxruntime.dll in {} (run tools/nui/setup_nui.py). Running "
        "camera-only: the colour stream works but nobody will be tracked.",
        error, xe::path_to_utf8(estimator_options.model_dir),
        cvars::nui_runtime_path.empty()
            ? xe::path_to_utf8(xe::filesystem::GetExecutableFolder() / "nui" /
                               "runtime")
            : xe::path_to_utf8(cvars::nui_runtime_path));
  }
  std::lock_guard<std::mutex> lock(stats_mutex_);
  estimator_ready_ = true;
  estimator_error_ = estimator_ ? std::string() : error;
  estimator_description_ = description;
  UpdateStatusLocked();
}

void WebcamNuiSource::InferenceThreadMain() {
  CreateEstimator();

  CameraFrame work;
  std::vector<PoseResult> poses;
  std::vector<PersonTracker::Assignment> assignments;
  std::string error;
  uint64_t last_error_log_us = 0;
  PoseEstimator::BackendHealth last_health = PoseEstimator::BackendHealth::kOk;
  while (running_) {
    {
      std::unique_lock<std::mutex> lock(capture_mutex_);
      capture_cv_.wait_for(lock, std::chrono::milliseconds(kCaptureTimeoutMs),
                           [this]() { return captured_fresh_ || !running_; });
      if (!running_) {
        break;
      }
      if (!captured_fresh_) {
        continue;
      }
      std::swap(work, captured_);
      captured_fresh_ = false;
    }
    if (work.stride == 0) {
      work.stride = work.width * 4;
    }
    if (!cvars::nui_camera_mirrored) {
      FlipHorizontal(&work);
    }

    const uint64_t start_us = NowUs();
    poses.clear();
    if (estimator_) {
      error.clear();
      if (!estimator_->Process(work.rgba.data(), work.width, work.height,
                               work.stride, &poses, &error)) {
        poses.clear();
        if (start_us - last_error_log_us > 2000000) {
          XELOGW("NUI webcam: pose estimation failed: {}", error);
          last_error_log_us = start_us;
        }
      }
      // The estimator logs its own line per transition; the status line
      // follows it so the UI shows a lost GPU (and the CPU fallback)
      // without anybody having to read the log.
      const PoseEstimator::BackendHealth health = estimator_->backend_health();
      if (health != last_health) {
        last_health = health;
        std::lock_guard<std::mutex> lock(stats_mutex_);
        estimator_health_ = health;
        estimator_status_ = estimator_->status();
        estimator_description_ = fmt::format(
            "{} / {}", estimator_->backend_name(), estimator_->model_name());
        UpdateStatusLocked();
      }
    }
    PersonTracker::Options tracker_options;
    assignments = tracker_.Update(poses, work.timestamp_us, tracker_options);
    for (uint32_t key : tracker_.expired_keys()) {
      skeleton_synthesizer_.Forget(key);
    }

    std::shared_ptr<SourceFrame> frame;
    {
      std::lock_guard<std::mutex> lock(publish_mutex_);
      frame = AllocateFrame();
    }
    Synthesize(work, poses, assignments, frame.get());
    const uint64_t end_us = NowUs();

    {
      std::lock_guard<std::mutex> lock(publish_mutex_);
      frame->sequence = ++sequence_;
      latest_ = frame;
    }
    // The preview (a full camera frame plus segmentations) is only copied
    // once somebody has asked for it.
    if (preview_requested_.load(std::memory_order_relaxed)) {
      std::lock_guard<std::mutex> lock(preview_mutex_);
      preview_.frame = work;
      preview_.poses = poses;
      preview_.person_keys.resize(assignments.size());
      for (size_t i = 0; i < assignments.size(); ++i) {
        preview_.person_keys[i] = assignments[i].person_key;
      }
      preview_valid_ = true;
    }
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      stats_.frames_produced++;
      const double total_ms = (end_us - start_us) / 1000.0;
      stats_.inference_ms = stats_.inference_ms > 0.0
                                ? stats_.inference_ms * 0.9 + total_ms * 0.1
                                : total_ms;
    }
  }
}

void WebcamNuiSource::Synthesize(
    const CameraFrame& camera_frame, const std::vector<PoseResult>& poses,
    const std::vector<PersonTracker::Assignment>& assignments,
    SourceFrame* out_frame) {
  DeviceState state;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    state = device_state_;
  }
  out_frame->capture_time_us = camera_frame.timestamp_us;

  // Skeletons: tracked bodies first, then position-only ones.
  SkeletonSynthesizer::Options skeleton_options;
  skeleton_options.image_width = camera_frame.width;
  skeleton_options.image_height = camera_frame.height;
  skeleton_options.hfov_degrees = static_cast<float>(cvars::nui_camera_hfov);
  skeleton_options.mirrored = true;
  skeleton_options.camera_pitch_degrees = state.camera_pitch_degrees;
  skeleton_options.tilt_degrees = state.tilt_degrees;
  skeleton_options.user_scale = static_cast<float>(cvars::nui_user_scale);
  // Frame interval for the temporal filters (the synthesizer clamps it).
  if (last_synthesis_timestamp_us_ &&
      camera_frame.timestamp_us > last_synthesis_timestamp_us_) {
    skeleton_options.frame_dt_seconds =
        static_cast<float>(camera_frame.timestamp_us -
                           last_synthesis_timestamp_us_) *
        1e-6f;
  }
  last_synthesis_timestamp_us_ = camera_frame.timestamp_us;

  struct Candidate {
    Skeleton skeleton;
    const PoseResult* pose;
  };
  std::vector<Candidate> tracked;
  std::vector<Candidate> position_only;
  const size_t count = std::min(poses.size(), assignments.size());
  for (size_t i = 0; i < count; ++i) {
    const PoseResult& pose = poses[i];
    if (!pose.valid) {
      continue;
    }
    const uint32_t key = assignments[i].person_key;
    if (!key) {
      continue;
    }
    Candidate candidate;
    candidate.pose = &pose;
    const bool full = skeleton_synthesizer_.Synthesize(
        pose, key, skeleton_options, &candidate.skeleton);
    candidate.skeleton.tracking_id = key;
    candidate.skeleton.enrollment_index = kInvalidUserIndex;
    candidate.skeleton.user_index = kInvalidUserIndex;
    if (full && candidate.skeleton.state == SkeletonState::kTracked) {
      tracked.push_back(std::move(candidate));
    } else {
      if (candidate.skeleton.state == SkeletonState::kNotTracked) {
        candidate.skeleton.state = SkeletonState::kPositionOnly;
      }
      position_only.push_back(std::move(candidate));
    }
  }
  std::vector<Candidate> ordered;
  ordered.reserve(tracked.size() + position_only.size());
  for (auto& c : tracked) {
    ordered.push_back(std::move(c));
  }
  for (auto& c : position_only) {
    ordered.push_back(std::move(c));
  }
  if (ordered.size() > kMaxSkeletons) {
    ordered.resize(kMaxSkeletons);
  }
  out_frame->body_count = static_cast<uint32_t>(ordered.size());
  for (size_t i = 0; i < ordered.size(); ++i) {
    out_frame->bodies[i] = ordered[i].skeleton;
  }

  ComputeFloorPlane(state.camera_height_m, state.tilt_degrees,
                    &out_frame->floor_clip_plane,
                    &out_frame->normal_to_gravity);

  // Depth and player mask.
  const bool want_depth = state.want_depth || state.want_player_mask;
  if (want_depth) {
    const size_t pixel_count = static_cast<size_t>(kDepthWidth) * kDepthHeight;
    if (out_frame->depth_mm.size() != pixel_count) {
      out_frame->depth_mm.resize(pixel_count);
    }
    if (out_frame->player_mask.size() != pixel_count) {
      out_frame->player_mask.resize(pixel_count);
    }
    DepthSynthesizer::Options depth_options;
    depth_options.camera_height_m = state.camera_height_m;
    depth_options.tilt_degrees = state.tilt_degrees;
    depth_options.render_position_only = true;
    uint16_t* depth = out_frame->depth_mm.data();
    uint8_t* mask = out_frame->player_mask.data();
    depth_synthesizer_.Render(out_frame->bodies, out_frame->body_count,
                              depth_options, depth, mask);

    // Person segmentation, when the model provides one: the silhouette
    // replaces the capsule outline of that person, keeping the synthesized
    // depth inside it and the background outside.
    bool any_segmentation = false;
    for (const auto& c : ordered) {
      if (c.pose->has_segmentation && !c.pose->segmentation.empty()) {
        any_segmentation = true;
        break;
      }
    }
    if (any_segmentation) {
      if (segmentation_mask_.size() != pixel_count) {
        segmentation_mask_.resize(pixel_count);
      }
      if (background_depth_.size() != pixel_count) {
        background_depth_.resize(pixel_count);
      }
      if (scratch_mask_.size() != pixel_count) {
        scratch_mask_.resize(pixel_count);
      }
      std::memset(segmentation_mask_.data(), 0, pixel_count);
      depth_synthesizer_.Render(out_frame->bodies, 0, depth_options,
                                background_depth_.data(), scratch_mask_.data());

      std::array<float, kMaxSkeletons> root_depth_m{};
      std::array<bool, kMaxSkeletons> has_segmentation{};
      for (size_t i = 0; i < ordered.size(); ++i) {
        const Skeleton& body = ordered[i].skeleton;
        float z = skeleton_synthesizer_.last_root_depth(body.tracking_id);
        if (z <= 0.0f) {
          z = body.position.z;
        }
        root_depth_m[i] = z;
        has_segmentation[i] = ordered[i].pose->has_segmentation &&
                              !ordered[i].pose->segmentation.empty();
      }
      // Nearer person wins: paint far to near.
      std::array<uint32_t, kMaxSkeletons> order{};
      for (uint32_t i = 0; i < ordered.size(); ++i) {
        order[i] = i;
      }
      std::sort(order.begin(), order.begin() + ordered.size(),
                [&](uint32_t a, uint32_t b) {
                  return root_depth_m[a] > root_depth_m[b];
                });
      const float angle = state.camera_pitch_degrees - state.tilt_degrees;
      for (uint32_t n = 0; n < ordered.size(); ++n) {
        const uint32_t i = order[n];
        if (!has_segmentation[i]) {
          continue;
        }
        const std::vector<uint8_t>& seg = ordered[i].pose->segmentation;
        uint32_t seg_w = 0, seg_h = 0;
        if (!SegmentationDimensions(seg.size(), &seg_w, &seg_h)) {
          has_segmentation[i] = false;
          continue;
        }
        // The segmentation covers the whole camera frame resampled to
        // seg_w x seg_h, so the projection uses the camera's aspect.
        mask_lut_->Update(kDepthWidth, kDepthHeight, kDepthNominalFocalLengthPx,
                          seg_w, seg_h, camera_frame.width, camera_frame.height,
                          static_cast<float>(cvars::nui_camera_hfov), angle);
        const uint8_t player = static_cast<uint8_t>(i + 1);
        const uint32_t* lut = mask_lut_->entries.data();
        uint8_t* seg_mask = segmentation_mask_.data();
        for (size_t p = 0; p < pixel_count; ++p) {
          const uint32_t src = lut[p];
          if (src != kInvalidLutEntry && seg[src] >= 128) {
            seg_mask[p] = player;
          }
        }
      }
      // Merge: segmentation silhouettes for people who have one, capsule
      // silhouettes for the others, nearer surface in front.
      for (size_t p = 0; p < pixel_count; ++p) {
        const uint8_t seg_player = segmentation_mask_[p];
        const uint8_t cap_player = mask[p];
        uint16_t cap_depth = depth[p];
        if (seg_player) {
          uint16_t seg_depth;
          if (cap_player == seg_player && cap_depth) {
            seg_depth = cap_depth;
          } else {
            seg_depth = static_cast<uint16_t>(std::clamp(
                root_depth_m[seg_player - 1] * 1000.0f, 1.0f, 65535.0f));
          }
          if (cap_player && !has_segmentation[cap_player - 1] &&
              cap_player != seg_player && cap_depth && cap_depth < seg_depth) {
            // The capsule-only person is in front.
            continue;
          }
          depth[p] = seg_depth;
          mask[p] = seg_player;
        } else if (cap_player && has_segmentation[cap_player - 1]) {
          // Capsule pixel of a person whose silhouette comes from the
          // segmentation: outside the silhouette, so background.
          depth[p] = background_depth_[p];
          mask[p] = 0;
        }
      }
    }
    out_frame->has_depth = true;
    out_frame->has_player_mask = true;
  }

  // Colour: the webcam image resampled into the Kinect colour camera's
  // field of view, 0xAARRGGBB.
  if (state.want_color && camera_frame.width && camera_frame.height) {
    const size_t pixel_count = static_cast<size_t>(kColorWidth) * kColorHeight;
    if (out_frame->color_argb.size() != pixel_count) {
      out_frame->color_argb.resize(pixel_count);
    }
    const float angle = state.camera_pitch_degrees - state.tilt_degrees;
    color_lut_->Update(kColorWidth, kColorHeight, kColorNominalFocalLengthPx,
                       camera_frame.width, camera_frame.height,
                       camera_frame.width, camera_frame.height,
                       static_cast<float>(cvars::nui_camera_hfov), angle);
    const uint32_t* lut = color_lut_->entries.data();
    const uint8_t* rgba = camera_frame.rgba.data();
    const uint32_t stride = camera_frame.stride;
    const uint32_t src_w = camera_frame.width;
    uint32_t* out = out_frame->color_argb.data();
    for (size_t p = 0; p < pixel_count; ++p) {
      const uint32_t src = lut[p];
      if (src == kInvalidLutEntry) {
        out[p] = 0xFF000000u;
        continue;
      }
      const uint8_t* px =
          rgba + static_cast<size_t>(src / src_w) * stride + (src % src_w) * 4;
      out[p] = 0xFF000000u | (static_cast<uint32_t>(px[0]) << 16) |
               (static_cast<uint32_t>(px[1]) << 8) | px[2];
    }
    out_frame->has_color = true;
  }
}

}  // namespace nui
}  // namespace xe
