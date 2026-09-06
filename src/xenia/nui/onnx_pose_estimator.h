/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_ONNX_POSE_ESTIMATOR_H_
#define XENIA_NUI_ONNX_POSE_ESTIMATOR_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "xenia/nui/onnx_runtime.h"
#include "xenia/nui/pose_estimator.h"

namespace xe {
namespace nui {

// Decides what to do after inference fails, so a driver TDR (which takes the
// DirectML device down and makes every later Run() fail the same way) does
// not end tracking for the rest of the run.
//
// Pure logic on a caller-supplied clock: no sessions, no threads, no GPU, so
// it is unit-tested directly. The owner feeds it RecordSuccess() /
// RecordFailure() per frame and, while it is not healthy, calls Poll() once
// per frame and does what it says:
//
//   kContinue    run the frame normally
//   kWait        the backoff deadline has not passed; fail the frame fast
//   kRebuildGpu  destroy and recreate the sessions on the configured
//                provider, then report the outcome with RecordRebuildResult
//   kRebuildCpu  same, but forced onto the CPU provider (last resort)
//   kGiveUp      inference is not coming back this run
//
// Backoff is 2 s, then 5 s, then 15 s, for at most kMaxGpuAttempts DirectML
// attempts; after that one CPU rebuild is tried and kept for the rest of the
// run.
class InferenceRecovery {
 public:
  enum class Action { kContinue, kWait, kRebuildGpu, kRebuildCpu, kGiveUp };
  enum class State {
    kHealthy,
    kRecovering,   // sessions dropped, waiting to rebuild
    kDegradedCpu,  // rebuilt on the CPU provider
    kFailed,       // gave up
  };

  // Consecutive plain failures (no device-lost signal) that also trigger a
  // rebuild: a wedged provider reports garbage rather than a DXGI error.
  static constexpr uint32_t kFailureThreshold = 8;
  static constexpr uint32_t kMaxGpuAttempts = 3;
  static constexpr uint64_t kBackoffUs[3] = {2000000, 5000000, 15000000};

  State state() const { return state_; }
  uint32_t consecutive_failures() const { return consecutive_failures_; }
  uint32_t gpu_attempts() const { return gpu_attempts_; }
  // Absolute deadline of the current backoff; only meaningful while
  // state() == kRecovering.
  uint64_t retry_deadline_us() const { return deadline_us_; }

  // A frame succeeded.
  void RecordSuccess();
  // A frame failed. |device_lost| when the provider said so (see
  // OnnxErrorIsDeviceLost). Returns kWait when the caller must now drop its
  // sessions and start the backoff, kContinue while the failure still looks
  // transient, kGiveUp once the run is over.
  Action RecordFailure(uint64_t now_us, bool device_lost);
  // What to do this frame. Const: it never advances the state machine on its
  // own, RecordRebuildResult does.
  Action Poll(uint64_t now_us) const;
  // Outcome of a rebuild started because Poll() asked for one. |on_cpu| must
  // match the action (kRebuildCpu -> true).
  void RecordRebuildResult(bool success, bool on_cpu, uint64_t now_us);
  // Ends the run: every later Poll() returns kGiveUp. Called when the owner
  // acts on a kGiveUp that did not come out of a rebuild.
  void MarkFailed();

 private:
  static uint64_t BackoffUs(uint32_t attempt);

  State state_ = State::kHealthy;
  uint32_t consecutive_failures_ = 0;
  uint32_t gpu_attempts_ = 0;
  bool cpu_attempted_ = false;
  uint64_t deadline_us_ = 0;
};

// MediaPipe BlazePose on ONNX Runtime.
//
// Per frame: the person detector (pose_detection.onnx, 224x224, SSD anchors
// + weighted NMS) runs every kDetectorInterval frames or whenever fewer than
// max_persons regions are tracked; every region (from a detection or from
// the previous frame's landmarks) is cropped as a rotated square, fed to the
// landmark model (pose_landmark_{lite,full,heavy}.onnx, 256x256) and decoded
// into 33 image landmarks, 33 world landmarks, a pose score and a
// segmentation mask. The next frame's region comes from the auxiliary
// landmarks (hip centre and size/rotation point), exactly like MediaPipe's
// pose_landmark_cpu graph. A region is dropped when its pose score falls
// below kMinPoseScore.
//
// All model constants (tensor names, anchor layout, scales) follow the
// MediaPipe graph files quoted next to the code in onnx_pose_estimator.cc.
class OnnxPoseEstimator : public PoseEstimator {
 public:
  static constexpr uint32_t kDetectorInputSize = 224;
  static constexpr uint32_t kDetectorAnchorCount = 2254;
  static constexpr uint32_t kDetectorValuesPerAnchor = 12;
  static constexpr uint32_t kDetectorKeypointCount = 4;
  static constexpr uint32_t kLandmarkInputSize = 256;
  static constexpr uint32_t kLandmarkModelCount = 39;  // 33 + 2 aux + 4 unused
  static constexpr uint32_t kLandmarkValues = 5;       // x y z vis presence
  static constexpr uint32_t kHeatmapSize = 64;
  static constexpr uint32_t kHeatmapKernel = 7;
  static constexpr uint32_t kDetectorInterval = 5;
  static constexpr float kMinDetectionScore = 0.5f;
  static constexpr float kMinPoseScore = 0.5f;
  static constexpr float kNmsIouThreshold = 0.3f;
  static constexpr float kRegionScale = 1.25f;

  // One decoded detection in full-image normalized coordinates.
  struct Detection {
    float score = 0.0f;
    // Face box (only used for NMS): x_min, y_min, x_max, y_max.
    std::array<float, 4> box{};
    // 0 = hip centre, 1 = full-body size/rotation point, 2 = shoulder
    // centre, 3 = upper-body size/rotation point.
    std::array<std::array<float, 2>, kDetectorKeypointCount> keypoints{};
  };

  static std::unique_ptr<OnnxPoseEstimator> Create(const Options& options,
                                                   std::string* out_error);
  ~OnnxPoseEstimator() override;

  bool Process(const uint8_t* rgba, uint32_t width, uint32_t height,
               uint32_t stride, std::vector<PoseResult>* out_results,
               std::string* out_error) override;
  void CancelRecovery() override;
  BackendHealth backend_health() const override;
  std::string status() const override;
  std::string backend_name() const override;
  std::string model_name() const override;
  double last_inference_ms() const override;

  // Pure helpers, exposed for tests and tools.
  //
  // SSD anchor centres for the detector, kDetectorAnchorCount pairs of
  // (x, y) in [0, 1] (anchor width/height are 1: fixed_anchor_size).
  static void GenerateAnchors(std::vector<float>* out_xy);
  // AlignmentPointsRectsCalculator (start 0, end 1, target 90 deg) followed
  // by RectTransformationCalculator (scale 1.25, square_long): the rotated
  // square crop centred on |x0,y0| (normalized) whose "up" is the direction
  // to |x1,y1|. |width| and |height| are the image size in pixels.
  static PoseRegion RegionFromAlignmentPoints(float x0, float y0, float x1,
                                              float y1, uint32_t width,
                                              uint32_t height);
  // Decodes raw detector tensors (|boxes| = anchors x 12, |scores| =
  // anchors x 1) into weighted-NMS'd detections in letterboxed-input
  // coordinates. |anchors_xy| from GenerateAnchors.
  static void DecodeDetections(const float* boxes, const float* scores,
                               const float* anchors_xy,
                               std::vector<Detection>* scratch,
                               std::vector<Detection>* out_detections);

 private:
  OnnxPoseEstimator() = default;
  bool Initialize(const Options& options, std::string* out_error);
  // Loads the runtime and both models on |provider| ("auto", "dml" or
  // "cpu") and binds their tensors. Used by Initialize and, after a lost
  // device, by MaybeRecover on the inference thread.
  bool BuildSessions(const std::string& provider, std::string* out_error);
  void DestroySessions();
  bool BindDetectorTensors(std::string* out_error);
  bool BindLandmarkTensors(std::string* out_error);

  // Recovery, all on the inference thread. MaybeRecover runs the state
  // machine at the top of Process (returning false without blocking while a
  // backoff is pending); NoteInferenceFailure classifies a failed frame.
  bool MaybeRecover(std::string* out_error);
  void NoteInferenceFailure(const std::string& error, std::string* out_error);
  void PublishHealth(BackendHealth health);

  bool RunDetector(const uint8_t* rgba, uint32_t width, uint32_t height,
                   uint32_t stride, std::string* out_error);
  bool RunLandmarks(const uint8_t* rgba, uint32_t width, uint32_t height,
                    uint32_t stride, const PoseRegion& region,
                    PoseResult* out_result, PoseRegion* out_next_region,
                    std::string* out_error);
  void ResampleSegmentation(const PoseRegion& region, uint32_t width,
                            uint32_t height, std::vector<uint8_t>* out_mask);
  // Takes a result's segmentation buffer back into |segmentation_pool_| so
  // the next frame reuses it instead of allocating 75 KB per person.
  void RecycleSegmentation(PoseResult* result);

  OnnxRuntime* runtime_ = nullptr;
  std::unique_ptr<OnnxSession> detector_;
  std::unique_ptr<OnnxSession> landmark_;
  uint32_t max_persons_ = 2;
  bool want_segmentation_ = true;
  // Kept so the sessions can be rebuilt after a lost device.
  Options options_;
  InferenceRecovery recovery_;
  // Set from another thread by CancelRecovery(); read by MaybeRecover on the
  // inference thread, which then neither waits nor rebuilds.
  std::atomic<bool> cancel_recovery_{false};

  // Written by the inference thread on a build or a health change, read by
  // any thread through the accessors.
  mutable std::mutex state_mutex_;
  std::string landmark_model_name_;
  std::string backend_name_;
  std::atomic<BackendHealth> health_{BackendHealth::kOk};

  // Tensor bindings (indices into the sessions' inputs()/outputs()).
  bool detector_nchw_ = false;
  int detector_boxes_index_ = -1;
  int detector_scores_index_ = -1;
  bool landmark_nchw_ = false;
  int landmark_points_index_ = -1;
  int landmark_flag_index_ = -1;
  int landmark_segmentation_index_ = -1;
  int landmark_heatmap_index_ = -1;
  int landmark_world_index_ = -1;

  // Preallocated buffers.
  std::vector<float> anchors_xy_;
  std::vector<float> detector_input_;
  std::vector<float> landmark_input_;
  std::vector<std::vector<float>> detector_outputs_;
  std::vector<std::vector<int64_t>> detector_shapes_;
  std::vector<std::vector<float>> landmark_outputs_;
  std::vector<std::vector<int64_t>> landmark_shapes_;
  std::vector<Detection> detection_scratch_;
  std::vector<Detection> detections_;
  std::vector<float> mask_;  // 256x256 sigmoid'd segmentation of the crop
  // Run() arguments, refilled (never reallocated) per frame so that inference
  // does not allocate.
  std::vector<const float*> detector_inputs_;
  std::vector<std::vector<int64_t>> detector_input_shapes_;
  std::vector<const float*> landmark_inputs_;
  std::vector<std::vector<int64_t>> landmark_input_shapes_;
  // Segmentation buffers reclaimed from the caller's results at the top of
  // Process and handed back out in ResampleSegmentation.
  std::vector<std::vector<uint8_t>> segmentation_pool_;

  // Regions being tracked, in result order; each is the crop for the next
  // frame (from the detector or from the previous landmarks).
  std::vector<PoseRegion> regions_;
  uint64_t frame_index_ = 0;
  // Written by the inference thread, read by any thread through the
  // accessor.
  std::atomic<double> last_inference_ms_{0.0};
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_ONNX_POSE_ESTIMATOR_H_
