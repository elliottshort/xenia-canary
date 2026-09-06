/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_SOURCES_WEBCAM_NUI_SOURCE_H_
#define XENIA_NUI_SOURCES_WEBCAM_NUI_SOURCE_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "xenia/base/threading.h"
#include "xenia/nui/camera_capture.h"
#include "xenia/nui/depth_synthesizer.h"
#include "xenia/nui/nui_source.h"
#include "xenia/nui/person_tracker.h"
#include "xenia/nui/pose_estimator.h"
#include "xenia/nui/skeleton_synthesizer.h"

namespace xe {
namespace nui {

struct WebcamRemapLut;

// Kinect data from a webcam: a capture thread pulls RGBA frames, an
// inference thread runs pose estimation and synthesizes the skeletons,
// player mask, depth and colour images of a SourceFrame.
//
// Start() only creates the two threads and returns: the camera is opened on
// the capture thread (retried every few seconds while it is unplugged or in
// use) and the pose estimator is created on the inference thread, so a
// title's NuiInitialize is never stalled by device or DirectML start-up.
// Until both are done AcquireLatest returns nullptr and GetStats().status
// says what is happening (see State). Stop() is valid in every state and
// interrupts a camera open in progress; an estimator creation in progress
// is waited for (at most a couple of seconds).
//
// Configuration comes from the nui_camera*, nui_model_*, nui_runtime_path,
// nui_execution_provider, nui_user_scale and nui_max_players cvars.
class WebcamNuiSource : public NuiSource {
 public:
  // Start-up progress. GetStats().status carries the matching text:
  //   kStartingCamera  "starting camera" or "camera error: <why> (retrying)"
  //   kLoadingModels   "loading models"
  //   kCameraOnly      "no pose estimation (<why>); camera only"
  //   kRunning         "<backend> / <model> / <camera name>", or, while
  //                    the estimator is recovering from a lost GPU device,
  //                    whatever PoseEstimator::status() reports
  enum class State {
    kStopped,
    kStartingCamera,
    kLoadingModels,
    kCameraOnly,
    kRunning,
  };

  WebcamNuiSource();
  ~WebcamNuiSource() override;

  std::string_view name() const override { return "webcam"; }
  bool Start(const DeviceState& initial_state) override;
  void Stop() override;
  void SetDeviceState(const DeviceState& state) override;
  std::shared_ptr<const SourceFrame> AcquireLatest(
      uint64_t last_sequence) override;
  void GetStats(SourceStats* out_stats) const override;

  State state() const;

  // Latest camera frame with the 2D landmarks of every tracked person, for
  // the preview window. Returns false if nothing has been captured yet.
  struct Preview {
    CameraFrame frame;
    std::vector<PoseResult> poses;
    std::vector<uint32_t> person_keys;
  };
  bool GetPreview(Preview* out_preview);

 private:
  void CaptureThreadMain();
  bool OpenCamera(std::string* out_error);
  // Sleeps until the camera retry interval passes or Stop() is called.
  void WaitForRetry();
  void InferenceThreadMain();
  void CreateEstimator();
  std::shared_ptr<SourceFrame> AllocateFrame();
  void Synthesize(const CameraFrame& camera_frame,
                  const std::vector<PoseResult>& poses,
                  const std::vector<PersonTracker::Assignment>& assignments,
                  SourceFrame* out_frame);
  // Recomputes stats_.status from the camera/estimator fields; call with
  // stats_mutex_ held.
  void UpdateStatusLocked();
  State StateLocked() const;

  std::mutex mutex_;
  std::condition_variable stop_cv_;  // with mutex_; signalled by Stop()
  DeviceState device_state_;
  std::atomic<bool> running_{false};

  // Created in Start(), opened/read by the capture thread only, closed by
  // the capture thread on errors and by Stop() (which is safe against an
  // Open in progress on the capture thread).
  std::unique_ptr<CameraCapture> camera_;
  // Created and used by the inference thread only.
  std::unique_ptr<PoseEstimator> estimator_;
  PersonTracker tracker_;
  SkeletonSynthesizer skeleton_synthesizer_;
  DepthSynthesizer depth_synthesizer_;

  // Capture -> inference hand-off (latest frame wins).
  std::mutex capture_mutex_;
  std::condition_variable capture_cv_;
  CameraFrame captured_;
  bool captured_fresh_ = false;

  std::unique_ptr<xe::threading::Thread> capture_thread_;
  std::unique_ptr<xe::threading::Thread> inference_thread_;

  // Published frames.
  std::mutex publish_mutex_;
  std::shared_ptr<SourceFrame> latest_;
  std::vector<std::shared_ptr<SourceFrame>> pool_;
  uint64_t sequence_ = 0;

  // Preview data; only maintained once GetPreview has been called.
  std::atomic<bool> preview_requested_{false};
  std::mutex preview_mutex_;
  Preview preview_;
  bool preview_valid_ = false;

  // Capture timestamp of the last synthesized frame (inference thread).
  uint64_t last_synthesis_timestamp_us_ = 0;

  // Synthesis scratch (inference thread only): Kinect-to-webcam pixel
  // lookup tables and the buffers used to merge segmentation silhouettes
  // with the capsule render.
  std::unique_ptr<WebcamRemapLut> mask_lut_;
  std::unique_ptr<WebcamRemapLut> color_lut_;
  std::vector<uint8_t> segmentation_mask_;
  std::vector<uint8_t> scratch_mask_;
  std::vector<uint16_t> background_depth_;

  // Stats and start-up progress (all under stats_mutex_).
  mutable std::mutex stats_mutex_;
  SourceStats stats_;
  bool camera_ready_ = false;
  std::string camera_error_;  // last open/read failure; empty when none
  std::string camera_name_;
  bool estimator_ready_ = false;       // creation finished (either way)
  std::string estimator_error_;        // empty when pose estimation works
  std::string estimator_description_;  // "<backend> / <model>"
  // Backend health of a working estimator (a lost GPU device and the
  // recovery that follows); the status text is the estimator's own, empty
  // while it is healthy.
  PoseEstimator::BackendHealth estimator_health_ =
      PoseEstimator::BackendHealth::kOk;
  std::string estimator_status_;
  uint64_t capture_count_ = 0;
  uint64_t capture_window_start_us_ = 0;
  uint64_t capture_window_count_ = 0;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_SOURCES_WEBCAM_NUI_SOURCE_H_
