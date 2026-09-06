/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_POSE_ESTIMATOR_H_
#define XENIA_NUI_POSE_ESTIMATOR_H_

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "xenia/nui/nui_types.h"

namespace xe {
namespace nui {

// BlazePose / MediaPipe Pose landmark indices (33 landmarks).
enum class PoseLandmarkIndex : uint8_t {
  kNose = 0,
  kLeftEyeInner = 1,
  kLeftEye = 2,
  kLeftEyeOuter = 3,
  kRightEyeInner = 4,
  kRightEye = 5,
  kRightEyeOuter = 6,
  kLeftEar = 7,
  kRightEar = 8,
  kMouthLeft = 9,
  kMouthRight = 10,
  kLeftShoulder = 11,
  kRightShoulder = 12,
  kLeftElbow = 13,
  kRightElbow = 14,
  kLeftWrist = 15,
  kRightWrist = 16,
  kLeftPinky = 17,
  kRightPinky = 18,
  kLeftIndex = 19,
  kRightIndex = 20,
  kLeftThumb = 21,
  kRightThumb = 22,
  kLeftHip = 23,
  kRightHip = 24,
  kLeftKnee = 25,
  kRightKnee = 26,
  kLeftAnkle = 27,
  kRightAnkle = 28,
  kLeftHeel = 29,
  kRightHeel = 30,
  kLeftFootIndex = 31,
  kRightFootIndex = 32,
  kCount = 33,
};
constexpr uint32_t kPoseLandmarkCount = 33;

struct PoseLandmark {
  // Normalized coordinates in the full camera image: x to the right, y down,
  // both in [0, 1] (may exceed the range for landmarks outside the frame).
  float x = 0.0f;
  float y = 0.0f;
  // Depth relative to the hips, in the same scale as x (image width units);
  // smaller is closer to the camera.
  float z = 0.0f;
  float visibility = 0.0f;  // [0, 1]
  float presence = 0.0f;    // [0, 1]
};

struct PoseWorldLandmark {
  // Metres, origin at the hip centre; x right, y down, z away from camera
  // (MediaPipe convention).
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct PoseRegion {
  // Person box in normalized full-image coordinates (centre and size) and
  // its rotation in radians, as used for the landmark model's input crop.
  float center_x = 0.5f;
  float center_y = 0.5f;
  float width = 1.0f;
  float height = 1.0f;
  float rotation = 0.0f;
  float score = 0.0f;
};

struct PoseResult {
  bool valid = false;
  // Confidence that a person is present in the region.
  float pose_score = 0.0f;
  PoseRegion region;
  std::array<PoseLandmark, kPoseLandmarkCount> landmarks{};
  std::array<PoseWorldLandmark, kPoseLandmarkCount> world_landmarks{};
  // Person segmentation over the full camera image, resampled to
  // kDepthWidth x kDepthHeight, 0 = background .. 255 = person. Empty when
  // the model/quality setting does not produce one.
  bool has_segmentation = false;
  std::vector<uint8_t> segmentation;
};

// Multi-person 3D pose estimation on RGB frames. The default implementation
// runs MediaPipe BlazePose (detector + landmark model) on ONNX Runtime; the
// detector runs periodically or when a tracked region is lost, and regions
// are carried from frame to frame so result order is stable while people
// stay in view.
class PoseEstimator {
 public:
  struct Options {
    std::filesystem::path model_dir;
    // "lite", "full" or "heavy" (BlazePose landmark model variants); "auto"
    // picks "full" on GPU providers and "lite" on CPU.
    std::string quality = "auto";
    // "auto", "dml" or "cpu".
    std::string execution_provider = "auto";
    int threads = 0;  // 0 = auto
    uint32_t max_persons = 2;
    bool want_segmentation = true;
  };

  // Creates the ONNX Runtime backed estimator. Returns nullptr with a
  // reason when the runtime or models are missing.
  static std::unique_ptr<PoseEstimator> Create(const Options& options,
                                               std::string* out_error);

  virtual ~PoseEstimator() = default;

  // Processes one RGBA frame (top-down, |stride| bytes per row) and returns
  // up to max_persons results, tracked ones first. Results for people who
  // left the frame are dropped, so the vector size varies.
  virtual bool Process(const uint8_t* rgba, uint32_t width, uint32_t height,
                       uint32_t stride, std::vector<PoseResult>* out_results,
                       std::string* out_error) = 0;

  // Backend health, for the stats line and the preview UI. Anything but kOk
  // means Process() is failing or running slower than it should; the
  // implementation recovers on its own where it can.
  enum class BackendHealth {
    kOk,
    kRecovering,   // the inference device was lost; sessions are rebuilding
    kDegradedCpu,  // fell back to the CPU provider after a lost device
    kFailed,       // inference will not come back for the rest of the run
  };
  virtual BackendHealth backend_health() const { return BackendHealth::kOk; }
  // One line describing a non-kOk health, empty while healthy.
  virtual std::string status() const { return std::string(); }
  bool degraded() const { return backend_health() != BackendHealth::kOk; }

  // The provider actually in use (it changes after a fallback).
  virtual std::string backend_name() const = 0;  // e.g. "DirectML", "CPU"
  virtual std::string model_name() const = 0;    // e.g. "pose_landmark_full"
  virtual double last_inference_ms() const = 0;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_POSE_ESTIMATOR_H_
