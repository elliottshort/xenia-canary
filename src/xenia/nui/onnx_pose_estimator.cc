/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/onnx_pose_estimator.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/nui/nui_flags.h"

// Model semantics (tensor layouts, anchors, decoding) follow the MediaPipe
// graphs the ONNX files were converted from:
//   mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
//   mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
//   mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt
//   mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt
//   mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt
// and the calculators they instantiate (SsdAnchorsCalculator,
// TensorsToDetectionsCalculator, NonMaxSuppressionCalculator,
// DetectionLetterboxRemovalCalculator, AlignmentPointsRectsCalculator,
// RectTransformationCalculator, ImageToTensorCalculator,
// TensorsToLandmarksCalculator, RefineLandmarksFromHeatmapCalculator,
// LandmarkProjectionCalculator, WorldLandmarkProjectionCalculator,
// TensorsToSegmentationCalculator).

namespace xe {
namespace nui {

namespace {

constexpr float kPi = 3.14159265358979f;

// Detector tensor names (tf2onnx conversion of pose_detection.tflite).
constexpr char kDetectorBoxesName[] = "Identity";     // [1, 2254, 12]
constexpr char kDetectorScoresName[] = "Identity_1";  // [1, 2254, 1]
// Landmark model tensor names (tf2onnx conversion of
// pose_landmark_{lite,full,heavy}.tflite).
constexpr char kLandmarkPointsName[] = "Identity";          // [1, 195]
constexpr char kLandmarkFlagName[] = "Identity_1";          // [1, 1]
constexpr char kLandmarkSegmentationName[] = "Identity_2";  // [1,256,256,1]
constexpr char kLandmarkHeatmapName[] = "Identity_3";       // [1,64,64,39]
constexpr char kLandmarkWorldName[] = "Identity_4";         // [1, 117]

// TensorsToDetectionsCalculator options in pose_detection_cpu.pbtxt.
constexpr float kDetectorScoreClip = 100.0f;  // score_clipping_thresh
constexpr float kDetectorBoxScale = 224.0f;   // x/y/w/h_scale
// RefineLandmarksFromHeatmapCalculator default min_confidence_to_refine.
constexpr float kHeatmapMinConfidence = 0.5f;

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// NormalizeRadians from detections_to_rects_calculator.cc.
inline float NormalizeRadians(float angle) {
  return angle - 2.0f * kPi * std::floor((angle + kPi) / (2.0f * kPi));
}

uint64_t NowUs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

int64_t ElementCount(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      return -1;
    }
    count *= dim;
  }
  return count;
}

std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) {
      out += ",";
    }
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

int FindTensorByName(const std::vector<OnnxTensorInfo>& tensors,
                     const char* name) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].name == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int FindTensorByElementCount(const std::vector<OnnxTensorInfo>& tensors,
                             int64_t count) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (ElementCount(tensors[i].shape) == count) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Accepts [1, size, size, 3] (NHWC) or [1, 3, size, size] (NCHW), dynamic
// dims allowed; sets |out_nchw|.
bool CheckImageInput(const OnnxTensorInfo& input, int64_t size, bool* out_nchw,
                     std::string* out_error) {
  const auto& shape = input.shape;
  auto dim_ok = [](int64_t dim, int64_t expected) {
    return dim < 0 || dim == expected;
  };
  if (shape.size() == 4 && dim_ok(shape[0], 1) && dim_ok(shape[1], size) &&
      dim_ok(shape[2], size) && dim_ok(shape[3], 3)) {
    *out_nchw = false;
    return true;
  }
  if (shape.size() == 4 && dim_ok(shape[0], 1) && shape[1] == 3 &&
      dim_ok(shape[2], size) && dim_ok(shape[3], size)) {
    *out_nchw = true;
    return true;
  }
  if (out_error) {
    *out_error = fmt::format(
        "input '{}' has shape {}, expected [1,{},{},3] or [1,3,{},{}]",
        input.name, ShapeToString(shape), size, size, size, size);
  }
  return false;
}

// Resamples the rotated rectangle (centre |cx_px,cy_px|, size |w_px,h_px|,
// |rotation| radians) of an RGBA image into a |size|x|size| RGB float
// tensor, bilinear. This is ImageToTensorCalculator with NORM_RECT:
// crop-local (u, v) in [0,1]^2 maps to
//   x = w*cos(r)*(u-0.5) - h*sin(r)*(v-0.5) + cx
//   y = w*sin(r)*(u-0.5) + h*cos(r)*(v-0.5) + cy
// where output pixel (i, j) samples u = (i+0.5)/size, v = (j+0.5)/size, and
// image pixel centres sit at half-integer continuous coordinates. Output
// value = pixel * scale + bias. Outside the image, |replicate_border| selects
// BORDER_REPLICATE (the nearest edge pixel) or BORDER_ZERO (black, i.e. the
// bias value). Which one each graph gets is settled by MediaPipe:
// mediapipe/calculators/tensor/image_to_tensor_calculator.proto documents
// "BORDER_REPLICATE is used by default" and GetBorderMode() in
// mediapipe/calculators/tensor/image_to_tensor_utils.cc maps
// BORDER_UNSPECIFIED to BorderMode::kReplicate; the landmark graph
// (mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt) leaves
// border_mode unset, the detector graph
// (mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt) sets
// border_mode: BORDER_ZERO.
void WarpCrop(const uint8_t* rgba, uint32_t width, uint32_t height,
              uint32_t stride, float cx_px, float cy_px, float w_px, float h_px,
              float rotation, uint32_t size, float scale, float bias, bool nchw,
              bool replicate_border, float* out) {
  const float c = std::cos(rotation);
  const float s = std::sin(rotation);
  const float step = 1.0f / static_cast<float>(size);
  const float dx_di = w_px * c * step;
  const float dy_di = w_px * s * step;
  const float dx_dj = -h_px * s * step;
  const float dy_dj = h_px * c * step;
  const float u0 = 0.5f * step - 0.5f;
  const float x00 = cx_px + w_px * c * u0 - h_px * s * u0 - 0.5f;
  const float y00 = cy_px + w_px * s * u0 + h_px * c * u0 - 0.5f;
  const int max_x = static_cast<int>(width) - 1;
  const int max_y = static_cast<int>(height) - 1;
  const size_t plane = static_cast<size_t>(size) * size;

  for (uint32_t j = 0; j < size; ++j) {
    float x = x00 + dx_dj * static_cast<float>(j);
    float y = y00 + dy_dj * static_cast<float>(j);
    float* row_out = out + (nchw ? static_cast<size_t>(j) * size
                                 : static_cast<size_t>(j) * size * 3);
    for (uint32_t i = 0; i < size; ++i, x += dx_di, y += dy_di) {
      float r = 0.0f;
      float g = 0.0f;
      float b = 0.0f;
      float sx = x;
      float sy = y;
      if (replicate_border) {
        // Clamping the sample position replicates the edge pixels.
        sx = std::min(std::max(sx, 0.0f), static_cast<float>(max_x));
        sy = std::min(std::max(sy, 0.0f), static_cast<float>(max_y));
      }
      const float fx = std::floor(sx);
      const float fy = std::floor(sy);
      const int x0 = static_cast<int>(fx);
      const int y0 = static_cast<int>(fy);
      if (x0 >= 0 && x0 < max_x && y0 >= 0 && y0 < max_y) {
        // All four taps inside the image.
        const float tx = sx - fx;
        const float ty = sy - fy;
        const uint8_t* p0 = rgba + static_cast<size_t>(y0) * stride +
                            static_cast<size_t>(x0) * 4;
        const uint8_t* p1 = p0 + stride;
        const float w00 = (1.0f - tx) * (1.0f - ty);
        const float w10 = tx * (1.0f - ty);
        const float w01 = (1.0f - tx) * ty;
        const float w11 = tx * ty;
        r = p0[0] * w00 + p0[4] * w10 + p1[0] * w01 + p1[4] * w11;
        g = p0[1] * w00 + p0[5] * w10 + p1[1] * w01 + p1[5] * w11;
        b = p0[2] * w00 + p0[6] * w10 + p1[2] * w01 + p1[6] * w11;
      } else if (x0 >= -1 && x0 <= max_x && y0 >= -1 && y0 <= max_y) {
        // Straddling the border: missing taps are black, or the nearest
        // edge pixel when replicating (only ever a zero-weight tap then,
        // since the position itself was clamped).
        const float tx = sx - fx;
        const float ty = sy - fy;
        auto tap = [&](int px, int py, float weight) {
          if (px < 0 || px > max_x || py < 0 || py > max_y) {
            if (!replicate_border) {
              return;
            }
            px = std::min(std::max(px, 0), max_x);
            py = std::min(std::max(py, 0), max_y);
          }
          const uint8_t* p = rgba + static_cast<size_t>(py) * stride +
                             static_cast<size_t>(px) * 4;
          r += p[0] * weight;
          g += p[1] * weight;
          b += p[2] * weight;
        };
        tap(x0, y0, (1.0f - tx) * (1.0f - ty));
        tap(x0 + 1, y0, tx * (1.0f - ty));
        tap(x0, y0 + 1, (1.0f - tx) * ty);
        tap(x0 + 1, y0 + 1, tx * ty);
      }
      if (nchw) {
        row_out[i] = r * scale + bias;
        row_out[plane + i] = g * scale + bias;
        row_out[2 * plane + i] = b * scale + bias;
      } else {
        row_out[i * 3 + 0] = r * scale + bias;
        row_out[i * 3 + 1] = g * scale + bias;
        row_out[i * 3 + 2] = b * scale + bias;
      }
    }
  }
}

float BoxIou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  const float ix = std::max(0.0f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
  const float iy = std::max(0.0f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  const float inter = ix * iy;
  const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
  const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
  const float uni = area_a + area_b - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

// SsdAnchorsCalculator with the pose_detection_cpu.pbtxt options:
//   num_layers 5, min_scale 0.1484375, max_scale 0.75, input 224x224,
//   anchor_offset_x/y 0.5, strides {8, 16, 32, 32, 32}, aspect_ratios {1.0},
//   fixed_anchor_size true, interpolated_scale_aspect_ratio 1.0 (default),
//   reduce_boxes_in_lowest_layer false (default).
// Layers sharing a stride are merged; each merged layer contributes
// (layers x 2) anchors per cell (the 1.0 aspect ratio plus the interpolated
// scale). Order: layer, row, column, anchor. With fixed_anchor_size every
// anchor has width = height = 1, so only the centres matter:
//   stride 8  -> 28x28 cells x 2 = 1568
//   stride 16 -> 14x14 cells x 2 =  392
//   stride 32 ->  7x7  cells x 6 =  294   (three layers)
//   total 2254.
void OnnxPoseEstimator::GenerateAnchors(std::vector<float>* out_xy) {
  static constexpr int kStrides[] = {8, 16, 32, 32, 32};
  static constexpr int kNumLayers = 5;
  static constexpr float kAnchorOffset = 0.5f;
  out_xy->clear();
  out_xy->reserve(kDetectorAnchorCount * 2);
  int layer = 0;
  while (layer < kNumLayers) {
    int anchors_per_cell = 0;
    int last = layer;
    while (last < kNumLayers && kStrides[last] == kStrides[layer]) {
      // One anchor for aspect_ratios[0] and one for the interpolated scale.
      anchors_per_cell += 2;
      ++last;
    }
    const int feature_map =
        static_cast<int>(std::ceil(static_cast<float>(kDetectorInputSize) /
                                   static_cast<float>(kStrides[layer])));
    for (int y = 0; y < feature_map; ++y) {
      for (int x = 0; x < feature_map; ++x) {
        for (int a = 0; a < anchors_per_cell; ++a) {
          out_xy->push_back((static_cast<float>(x) + kAnchorOffset) /
                            static_cast<float>(feature_map));
          out_xy->push_back((static_cast<float>(y) + kAnchorOffset) /
                            static_cast<float>(feature_map));
        }
      }
    }
    layer = last;
  }
}

PoseRegion OnnxPoseEstimator::RegionFromAlignmentPoints(float x0, float y0,
                                                        float x1, float y1,
                                                        uint32_t width,
                                                        uint32_t height) {
  // AlignmentPointsRectsCalculator (rotation_vector_start 0, end 1,
  // target_angle 90 deg) works in pixels.
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  const float px0 = x0 * w;
  const float py0 = y0 * h;
  const float px1 = x1 * w;
  const float py1 = y1 * h;
  const float box_size = 2.0f * std::hypot(px1 - px0, py1 - py0);
  const float rotation =
      NormalizeRadians(kPi * 0.5f - std::atan2(-(py1 - py0), px1 - px0));
  // RectTransformationCalculator: scale_x = scale_y = 1.25, square_long
  // (both sides already equal in pixels), no shift.
  const float side = kRegionScale * box_size;
  PoseRegion region;
  region.center_x = x0;
  region.center_y = y0;
  region.width = side / w;
  region.height = side / h;
  region.rotation = rotation;
  region.score = 0.0f;
  return region;
}

// TensorsToDetectionsCalculator (num_classes 1, num_boxes 2254,
// num_coords 12, box_coord_offset 0, keypoint_coord_offset 4,
// num_keypoints 4, num_values_per_keypoint 2, sigmoid_score true,
// score_clipping_thresh 100, reverse_output_order true => box format XYWH,
// x/y/w/h_scale 224, min_score_thresh 0.5) followed by
// NonMaxSuppressionCalculator (min_suppression_threshold 0.3, IoU, WEIGHTED).
void OnnxPoseEstimator::DecodeDetections(const float* boxes,
                                         const float* scores,
                                         const float* anchors_xy,
                                         std::vector<Detection>* scratch,
                                         std::vector<Detection>* out) {
  scratch->clear();
  for (uint32_t i = 0; i < kDetectorAnchorCount; ++i) {
    const float raw_score =
        std::min(kDetectorScoreClip, std::max(-kDetectorScoreClip, scores[i]));
    const float score = Sigmoid(raw_score);
    if (score < kMinDetectionScore) {
      continue;
    }
    const float* r = boxes + static_cast<size_t>(i) * kDetectorValuesPerAnchor;
    const float ax = anchors_xy[i * 2 + 0];
    const float ay = anchors_xy[i * 2 + 1];
    // anchor.w = anchor.h = 1 (fixed_anchor_size).
    const float x_center = r[0] / kDetectorBoxScale + ax;
    const float y_center = r[1] / kDetectorBoxScale + ay;
    const float w = r[2] / kDetectorBoxScale;
    const float h = r[3] / kDetectorBoxScale;
    Detection det;
    det.score = score;
    det.box = {x_center - w * 0.5f, y_center - h * 0.5f, x_center + w * 0.5f,
               y_center + h * 0.5f};
    for (uint32_t k = 0; k < kDetectorKeypointCount; ++k) {
      det.keypoints[k][0] = r[4 + 2 * k] / kDetectorBoxScale + ax;
      det.keypoints[k][1] = r[5 + 2 * k] / kDetectorBoxScale + ay;
    }
    scratch->push_back(det);
  }
  std::stable_sort(
      scratch->begin(), scratch->end(),
      [](const Detection& a, const Detection& b) { return a.score > b.score; });

  // Weighted NMS: the highest remaining detection absorbs every remaining
  // detection whose IoU with it exceeds the threshold (itself included);
  // box corners and keypoints become the score-weighted average, the score
  // stays the top score. Consumed entries are marked with a negative score.
  out->clear();
  for (size_t i = 0; i < scratch->size(); ++i) {
    if ((*scratch)[i].score < 0.0f) {
      continue;
    }
    const Detection top = (*scratch)[i];
    (*scratch)[i].score = -1.0f;
    float total = top.score;
    std::array<float, 4> box = top.box;
    std::array<std::array<float, 2>, kDetectorKeypointCount> kps =
        top.keypoints;
    for (auto& v : box) {
      v *= top.score;
    }
    for (auto& kp : kps) {
      kp[0] *= top.score;
      kp[1] *= top.score;
    }
    for (size_t j = i + 1; j < scratch->size(); ++j) {
      Detection& cand = (*scratch)[j];
      if (cand.score < 0.0f || BoxIou(top.box, cand.box) <= kNmsIouThreshold) {
        continue;
      }
      total += cand.score;
      for (int v = 0; v < 4; ++v) {
        box[v] += cand.box[v] * cand.score;
      }
      for (uint32_t k = 0; k < kDetectorKeypointCount; ++k) {
        kps[k][0] += cand.keypoints[k][0] * cand.score;
        kps[k][1] += cand.keypoints[k][1] * cand.score;
      }
      cand.score = -1.0f;
    }
    Detection merged;
    merged.score = top.score;
    for (int v = 0; v < 4; ++v) {
      merged.box[v] = box[v] / total;
    }
    for (uint32_t k = 0; k < kDetectorKeypointCount; ++k) {
      merged.keypoints[k][0] = kps[k][0] / total;
      merged.keypoints[k][1] = kps[k][1] / total;
    }
    out->push_back(merged);
  }
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Recovery state machine
// ---------------------------------------------------------------------------

uint64_t InferenceRecovery::BackoffUs(uint32_t attempt) {
  return kBackoffUs[std::min<uint32_t>(
      attempt, static_cast<uint32_t>(std::size(kBackoffUs)) - 1)];
}

void InferenceRecovery::RecordSuccess() { consecutive_failures_ = 0; }

InferenceRecovery::Action InferenceRecovery::RecordFailure(uint64_t now_us,
                                                           bool device_lost) {
  if (state_ == State::kFailed) {
    return Action::kGiveUp;
  }
  ++consecutive_failures_;
  if (state_ == State::kRecovering) {
    // Already waiting on a rebuild; Poll() decides when it happens.
    return Action::kWait;
  }
  if (state_ == State::kDegradedCpu) {
    // The CPU provider is the last resort, so a run of failures there ends
    // the run rather than starting another rebuild.
    if (consecutive_failures_ < kFailureThreshold) {
      return Action::kContinue;
    }
    state_ = State::kFailed;
    return Action::kGiveUp;
  }
  if (!device_lost && consecutive_failures_ < kFailureThreshold) {
    return Action::kContinue;
  }
  state_ = State::kRecovering;
  deadline_us_ = now_us + BackoffUs(gpu_attempts_);
  return Action::kWait;
}

InferenceRecovery::Action InferenceRecovery::Poll(uint64_t now_us) const {
  switch (state_) {
    case State::kHealthy:
    case State::kDegradedCpu:
      return Action::kContinue;
    case State::kFailed:
      return Action::kGiveUp;
    case State::kRecovering:
      break;
  }
  if (now_us < deadline_us_) {
    return Action::kWait;
  }
  if (gpu_attempts_ < kMaxGpuAttempts) {
    return Action::kRebuildGpu;
  }
  if (!cpu_attempted_) {
    return Action::kRebuildCpu;
  }
  return Action::kGiveUp;
}

void InferenceRecovery::RecordRebuildResult(bool success, bool on_cpu,
                                            uint64_t now_us) {
  consecutive_failures_ = 0;
  if (on_cpu) {
    cpu_attempted_ = true;
    state_ = success ? State::kDegradedCpu : State::kFailed;
    return;
  }
  ++gpu_attempts_;
  if (success) {
    // A later, unrelated device loss gets a full set of attempts again.
    state_ = State::kHealthy;
    gpu_attempts_ = 0;
    deadline_us_ = 0;
    return;
  }
  state_ = State::kRecovering;
  // With the GPU attempts used up the CPU rebuild happens on the next frame.
  deadline_us_ = gpu_attempts_ < kMaxGpuAttempts
                     ? now_us + BackoffUs(gpu_attempts_)
                     : now_us;
}

void InferenceRecovery::MarkFailed() {
  state_ = State::kFailed;
  consecutive_failures_ = 0;
  deadline_us_ = 0;
}

std::unique_ptr<PoseEstimator> PoseEstimator::Create(const Options& options,
                                                     std::string* out_error) {
  return OnnxPoseEstimator::Create(options, out_error);
}

std::unique_ptr<OnnxPoseEstimator> OnnxPoseEstimator::Create(
    const Options& options, std::string* out_error) {
  std::unique_ptr<OnnxPoseEstimator> estimator(new OnnxPoseEstimator());
  if (!estimator->Initialize(options, out_error)) {
    return nullptr;
  }
  return estimator;
}

OnnxPoseEstimator::~OnnxPoseEstimator() = default;

bool OnnxPoseEstimator::Initialize(const Options& options,
                                   std::string* out_error) {
  options_ = options;
  max_persons_ = std::max<uint32_t>(
      1, std::min<uint32_t>(options.max_persons, kMaxSkeletons));

  if (options.model_dir.empty()) {
    if (out_error) {
      *out_error = "no model folder configured (nui_model_path)";
    }
    return false;
  }

  if (!BuildSessions(options.execution_provider, out_error)) {
    return false;
  }

  // Buffers.
  GenerateAnchors(&anchors_xy_);
  detector_input_.assign(
      static_cast<size_t>(kDetectorInputSize) * kDetectorInputSize * 3, 0.0f);
  landmark_input_.assign(
      static_cast<size_t>(kLandmarkInputSize) * kLandmarkInputSize * 3, 0.0f);
  mask_.assign(static_cast<size_t>(kLandmarkInputSize) * kLandmarkInputSize,
               0.0f);
  detection_scratch_.reserve(256);
  detections_.reserve(16);
  regions_.reserve(max_persons_);
  frame_index_ = 0;

  XELOGI(
      "NUI pose: ONNX Runtime {} ({}), detector pose_detection.onnx, "
      "landmarks {} ({} persons, segmentation {})",
      runtime_->version(), landmark_->provider_name(), landmark_model_name_,
      max_persons_, want_segmentation_ ? "on" : "off");
  return true;
}

void OnnxPoseEstimator::DestroySessions() {
  detector_.reset();
  landmark_.reset();
  regions_.clear();
}

void OnnxPoseEstimator::CancelRecovery() {
  cancel_recovery_.store(true, std::memory_order_relaxed);
}

void OnnxPoseEstimator::RecycleSegmentation(PoseResult* result) {
  result->has_segmentation = false;
  if (result->segmentation.capacity() == 0 ||
      segmentation_pool_.size() >= max_persons_ + 2) {
    return;
  }
  result->segmentation.clear();
  segmentation_pool_.push_back(std::move(result->segmentation));
  result->segmentation = std::vector<uint8_t>();
}

bool OnnxPoseEstimator::BuildSessions(const std::string& provider,
                                      std::string* out_error) {
  DestroySessions();
  std::string error;
  const Options& options = options_;
  want_segmentation_ = options.want_segmentation;

  // The runtime search order is nui_runtime_path, <storage>/nui/runtime,
  // then the executable folder. Options carries only the model folder, so
  // the storage root is taken as the grandparent of <storage>/nui/models;
  // the loader falls back to <exe>/nui/runtime and <exe> anyway.
  std::filesystem::path storage_root;
  if (options.model_dir.has_parent_path() &&
      options.model_dir.parent_path().has_parent_path()) {
    storage_root = options.model_dir.parent_path().parent_path();
  }
  runtime_ = OnnxRuntime::Get(cvars::nui_runtime_path, storage_root);
  if (!runtime_) {
    if (out_error) {
      *out_error = OnnxRuntime::load_error();
    }
    return false;
  }

  OnnxSession::Options session_options;
  session_options.execution_provider = provider;
  session_options.intra_op_threads = options.threads;

  // Detector.
  std::filesystem::path detector_path =
      options.model_dir / "pose_detection.onnx";
  std::error_code ec;
  if (!std::filesystem::exists(detector_path, ec)) {
    if (out_error) {
      *out_error = fmt::format(
          "pose_detection.onnx not found in {}; run tools/nui/setup_nui.py "
          "or set nui_model_path",
          xe::path_to_utf8(options.model_dir));
    }
    return false;
  }
  detector_ =
      OnnxSession::Create(runtime_, detector_path, session_options, &error);
  if (!detector_) {
    if (out_error) {
      *out_error = fmt::format("pose_detection.onnx: {}", error);
    }
    return false;
  }
  if (!BindDetectorTensors(out_error)) {
    DestroySessions();
    return false;
  }

  // Landmark model: "auto" quality follows the provider the detector got.
  std::string quality = ToLower(options.quality);
  if (quality.empty() || quality == "auto") {
    quality = detector_->provider_name() == "DirectML" ? "full" : "lite";
  }
  if (quality != "lite" && quality != "full" && quality != "heavy") {
    XELOGW("NUI pose: unknown model quality '{}', using 'full'",
           options.quality);
    quality = "full";
  }
  std::vector<std::string> candidates;
  for (const char* variant : {quality.c_str(), "full", "lite"}) {
    if (std::find(candidates.begin(), candidates.end(), variant) ==
        candidates.end()) {
      candidates.push_back(variant);
    }
  }
  // Held here and published to the accessors only once the session is bound:
  // a half-built rebuild must not make backend_name() / model_name() name a
  // session that was torn down.
  std::string loaded_model_name;
  std::string loaded_backend_name;
  std::string tried;
  for (const auto& variant : candidates) {
    std::string file = "pose_landmark_" + variant + ".onnx";
    std::filesystem::path path = options.model_dir / file;
    if (!tried.empty()) {
      tried += ", ";
    }
    tried += file;
    if (!std::filesystem::exists(path, ec)) {
      if (variant == quality) {
        XELOGW("NUI pose: {} not found in {}; trying another variant", file,
               xe::path_to_utf8(options.model_dir));
      }
      continue;
    }
    landmark_ = OnnxSession::Create(runtime_, path, session_options, &error);
    if (!landmark_) {
      XELOGW("NUI pose: {} failed to load: {}", file, error);
      continue;
    }
    loaded_model_name = "pose_landmark_" + variant;
    loaded_backend_name = landmark_->provider_name();
    if (variant != quality) {
      XELOGW("NUI pose: using {} instead of pose_landmark_{}",
             loaded_model_name, quality);
    }
    break;
  }
  if (!landmark_) {
    DestroySessions();
    if (out_error) {
      *out_error = fmt::format(
          "no landmark model could be loaded from {} (tried {}); run "
          "tools/nui/setup_nui.py",
          xe::path_to_utf8(options.model_dir), tried);
    }
    return false;
  }
  if (!BindLandmarkTensors(out_error)) {
    DestroySessions();
    return false;
  }
  {
    // Published to the accessors, which any thread may call.
    std::lock_guard<std::mutex> lock(state_mutex_);
    landmark_model_name_ = std::move(loaded_model_name);
    backend_name_ = std::move(loaded_backend_name);
  }
  return true;
}

bool OnnxPoseEstimator::BindDetectorTensors(std::string* out_error) {
  const auto& inputs = detector_->inputs();
  const auto& outputs = detector_->outputs();
  if (inputs.size() != 1) {
    if (out_error) {
      *out_error = fmt::format("pose_detection.onnx has {} inputs, expected 1",
                               inputs.size());
    }
    return false;
  }
  if (!CheckImageInput(inputs[0], kDetectorInputSize, &detector_nchw_,
                       out_error)) {
    return false;
  }
  const int64_t boxes_count =
      static_cast<int64_t>(kDetectorAnchorCount) * kDetectorValuesPerAnchor;
  detector_boxes_index_ = FindTensorByElementCount(outputs, boxes_count);
  detector_scores_index_ =
      FindTensorByElementCount(outputs, kDetectorAnchorCount);
  if (detector_boxes_index_ < 0) {
    detector_boxes_index_ = FindTensorByName(outputs, kDetectorBoxesName);
  }
  if (detector_scores_index_ < 0) {
    detector_scores_index_ = FindTensorByName(outputs, kDetectorScoresName);
  }
  if (detector_boxes_index_ < 0 || detector_scores_index_ < 0 ||
      detector_boxes_index_ == detector_scores_index_) {
    if (out_error) {
      *out_error = fmt::format(
          "pose_detection.onnx outputs not recognized (expected {} "
          "[1,{},{}] and {} [1,{},1])",
          kDetectorBoxesName, kDetectorAnchorCount, kDetectorValuesPerAnchor,
          kDetectorScoresName, kDetectorAnchorCount);
    }
    return false;
  }
  return true;
}

bool OnnxPoseEstimator::BindLandmarkTensors(std::string* out_error) {
  const auto& inputs = landmark_->inputs();
  const auto& outputs = landmark_->outputs();
  if (inputs.size() != 1) {
    if (out_error) {
      *out_error = fmt::format("{} has {} inputs, expected 1",
                               landmark_model_name_, inputs.size());
    }
    return false;
  }
  if (!CheckImageInput(inputs[0], kLandmarkInputSize, &landmark_nchw_,
                       out_error)) {
    return false;
  }
  // Bind by shape first (ORT output order is not guaranteed to follow the
  // graph), then by the tf2onnx names.
  const int64_t points_count = kLandmarkModelCount * kLandmarkValues;  // 195
  const int64_t world_count = kLandmarkModelCount * 3;                 // 117
  const int64_t seg_count =
      static_cast<int64_t>(kLandmarkInputSize) * kLandmarkInputSize;
  const int64_t heatmap_count =
      static_cast<int64_t>(kHeatmapSize) * kHeatmapSize * kLandmarkModelCount;
  landmark_points_index_ = FindTensorByElementCount(outputs, points_count);
  landmark_flag_index_ = FindTensorByElementCount(outputs, 1);
  landmark_world_index_ = FindTensorByElementCount(outputs, world_count);
  landmark_segmentation_index_ = FindTensorByElementCount(outputs, seg_count);
  landmark_heatmap_index_ = FindTensorByElementCount(outputs, heatmap_count);
  if (landmark_points_index_ < 0) {
    landmark_points_index_ = FindTensorByName(outputs, kLandmarkPointsName);
  }
  if (landmark_flag_index_ < 0) {
    landmark_flag_index_ = FindTensorByName(outputs, kLandmarkFlagName);
  }
  if (landmark_world_index_ < 0) {
    landmark_world_index_ = FindTensorByName(outputs, kLandmarkWorldName);
  }
  if (landmark_segmentation_index_ < 0) {
    landmark_segmentation_index_ =
        FindTensorByName(outputs, kLandmarkSegmentationName);
  }
  if (landmark_heatmap_index_ < 0) {
    landmark_heatmap_index_ = FindTensorByName(outputs, kLandmarkHeatmapName);
  }
  if (landmark_points_index_ < 0 || landmark_flag_index_ < 0 ||
      landmark_world_index_ < 0) {
    if (out_error) {
      *out_error = fmt::format(
          "{} outputs not recognized (expected {} [1,195], {} [1,1] and {} "
          "[1,117])",
          landmark_model_name_, kLandmarkPointsName, kLandmarkFlagName,
          kLandmarkWorldName);
    }
    return false;
  }
  // The heatmap is only used when it is HWC (1x64x64x39); a channels-first
  // export would need a different indexing and is simply not refined.
  if (landmark_heatmap_index_ >= 0) {
    const auto& shape = outputs[landmark_heatmap_index_].shape;
    if (shape.size() != 4 || shape[1] != kHeatmapSize ||
        shape[2] != kHeatmapSize || shape[3] != kLandmarkModelCount) {
      XELOGW("NUI pose: {} heatmap has shape {}; skipping landmark refinement",
             landmark_model_name_, ShapeToString(shape));
      landmark_heatmap_index_ = -1;
    }
  }
  if (want_segmentation_ && landmark_segmentation_index_ < 0) {
    XELOGW("NUI pose: {} has no segmentation output", landmark_model_name_);
    want_segmentation_ = false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Per-frame processing
// ---------------------------------------------------------------------------

bool OnnxPoseEstimator::Process(const uint8_t* rgba, uint32_t width,
                                uint32_t height, uint32_t stride,
                                std::vector<PoseResult>* out_results,
                                std::string* out_error) {
  if (!rgba || width < 2 || height < 2 || stride < width * 4 || !out_results) {
    if (out_error) {
      *out_error = "invalid frame";
    }
    if (out_results) {
      out_results->clear();
    }
    return false;
  }
  // Take the previous frame's segmentation buffers back before anything can
  // clear or shrink |out_results|: they are kDepthWidth * kDepthHeight bytes
  // each, their size never changes, and this runs on the inference thread.
  for (auto& result : *out_results) {
    RecycleSegmentation(&result);
  }
  // A lost GPU device makes every later Run() fail the same way, so the
  // sessions are rebuilt here (never per frame: MaybeRecover honours the
  // backoff and returns immediately while one is pending).
  if (!MaybeRecover(out_error)) {
    out_results->clear();
    return false;
  }
  if (!detector_ || !landmark_) {
    if (out_error) {
      *out_error = "estimator is not initialized";
    }
    out_results->clear();
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  ++frame_index_;

  // The detector runs immediately when nobody is tracked (MediaPipe's
  // "no previous ROI" case) and every kDetectorInterval frames while there
  // is room for another person; with every slot taken it would only find
  // people already tracked.
  const bool run_detector =
      regions_.empty() ||
      (regions_.size() < max_persons_ && frame_index_ % kDetectorInterval == 0);
  std::string run_error;
  if (run_detector) {
    if (!RunDetector(rgba, width, height, stride, &run_error)) {
      regions_.clear();
      out_results->clear();
      NoteInferenceFailure(run_error, out_error);
      return false;
    }
    const float w = static_cast<float>(width);
    const float h = static_cast<float>(height);
    for (const Detection& det : detections_) {
      if (regions_.size() >= max_persons_) {
        break;
      }
      PoseRegion region = RegionFromAlignmentPoints(
          det.keypoints[0][0], det.keypoints[0][1], det.keypoints[1][0],
          det.keypoints[1][1], width, height);
      if (!(region.width * w >= 8.0f) || !std::isfinite(region.rotation)) {
        continue;
      }
      region.score = det.score;
      // Detections of people already tracked keep the (better) landmark
      // derived region: same person when the centres are closer than half
      // the larger crop.
      bool duplicate = false;
      for (const PoseRegion& tracked : regions_) {
        const float dist_px =
            std::hypot((tracked.center_x - region.center_x) * w,
                       (tracked.center_y - region.center_y) * h);
        const float size_px = std::max(tracked.width, region.width) * w;
        if (dist_px < 0.5f * size_px) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        regions_.push_back(region);
      }
    }
  }

  // Landmarks for every region; results keep the region order so a person's
  // index is stable while tracked (dropped regions close the gap).
  out_results->resize(regions_.size());
  size_t kept = 0;
  for (size_t i = 0; i < regions_.size();) {
    PoseResult& result = (*out_results)[kept];
    PoseRegion next_region;
    if (!RunLandmarks(rgba, width, height, stride, regions_[i], &result,
                      &next_region, &run_error)) {
      regions_.clear();
      out_results->clear();
      NoteInferenceFailure(run_error, out_error);
      return false;
    }
    if (result.pose_score < kMinPoseScore ||
        !(next_region.width * width >= 8.0f) ||
        !std::isfinite(next_region.rotation)) {
      regions_.erase(regions_.begin() + static_cast<std::ptrdiff_t>(i));
      continue;
    }
    next_region.score = result.pose_score;
    regions_[i] = next_region;
    ++kept;
    ++i;
  }
  // People who left keep their (never resized) mask buffers in the pool.
  for (size_t i = kept; i < out_results->size(); ++i) {
    RecycleSegmentation(&(*out_results)[i]);
  }
  out_results->resize(kept);

  const auto t1 = std::chrono::steady_clock::now();
  last_inference_ms_.store(
      std::chrono::duration<double, std::milli>(t1 - t0).count(),
      std::memory_order_relaxed);
  recovery_.RecordSuccess();
  return true;
}

// ---------------------------------------------------------------------------
// Recovery (inference thread)
// ---------------------------------------------------------------------------

bool OnnxPoseEstimator::MaybeRecover(std::string* out_error) {
  const InferenceRecovery::State state = recovery_.state();
  if (state == InferenceRecovery::State::kHealthy ||
      state == InferenceRecovery::State::kDegradedCpu) {
    return true;  // sessions are alive
  }
  if (cancel_recovery_.load(std::memory_order_relaxed)) {
    // Somebody is shutting us down. A rebuild calls into the provider (two
    // CreateSession calls plus a DXGI enumeration) and can block for tens of
    // seconds on an adapter that is still resetting, which the thread waiting
    // for us must not pay for. The state machine is left untouched.
    if (out_error) {
      *out_error = "pose estimation is shutting down";
    }
    return false;
  }
  const uint64_t now_us = NowUs();
  std::string error;
  switch (recovery_.Poll(now_us)) {
    case InferenceRecovery::Action::kContinue:
      return true;
    case InferenceRecovery::Action::kWait: {
      const uint64_t deadline_us = recovery_.retry_deadline_us();
      const double wait_s =
          deadline_us > now_us ? (deadline_us - now_us) / 1000000.0 : 0.0;
      if (out_error) {
        *out_error = fmt::format(
            "the inference device was lost; rebuilding in {:.1f} s", wait_s);
      }
      return false;
    }
    case InferenceRecovery::Action::kRebuildGpu: {
      const uint32_t attempt = recovery_.gpu_attempts() + 1;
      XELOGW("NUI pose: rebuilding the inference sessions (attempt {} of {})",
             attempt, InferenceRecovery::kMaxGpuAttempts);
      const bool ok = BuildSessions(options_.execution_provider, &error);
      recovery_.RecordRebuildResult(ok, /*on_cpu=*/false, NowUs());
      if (ok) {
        XELOGI("NUI pose: inference recovered on {} with {}", backend_name(),
               model_name());
        PublishHealth(BackendHealth::kOk);
        return true;
      }
      DestroySessions();
      if (out_error) {
        *out_error = fmt::format(
            "the inference device was lost; rebuild failed: {}", error);
      }
      return false;
    }
    case InferenceRecovery::Action::kRebuildCpu: {
      const bool ok = BuildSessions("cpu", &error);
      recovery_.RecordRebuildResult(ok, /*on_cpu=*/true, NowUs());
      if (ok) {
        XELOGW(
            "NUI pose: the GPU did not come back after {} attempts; pose "
            "estimation now runs on the CPU with {}",
            InferenceRecovery::kMaxGpuAttempts, model_name());
        PublishHealth(BackendHealth::kDegradedCpu);
        return true;
      }
      DestroySessions();
      XELOGE(
          "NUI pose: the CPU fallback failed too ({}); giving up on pose "
          "estimation for the rest of this run",
          error);
      PublishHealth(BackendHealth::kFailed);
      if (out_error) {
        *out_error = "pose estimation is unavailable after a lost device";
      }
      return false;
    }
    case InferenceRecovery::Action::kGiveUp:
      // Poll() reports this without a rebuild once every attempt has been
      // used; make it terminal here so backend_health() stops advertising a
      // recovery that will never happen.
      if (recovery_.state() != InferenceRecovery::State::kFailed) {
        XELOGE(
            "NUI pose: no inference device came back after {} GPU attempts "
            "and the CPU fallback; giving up for the rest of this run",
            InferenceRecovery::kMaxGpuAttempts);
        recovery_.MarkFailed();
        DestroySessions();
        PublishHealth(BackendHealth::kFailed);
      }
      break;
  }
  if (out_error) {
    *out_error = "pose estimation is unavailable after a lost device";
  }
  return false;
}

void OnnxPoseEstimator::NoteInferenceFailure(const std::string& error,
                                             std::string* out_error) {
  if (out_error) {
    *out_error = error;
  }
  const bool device_lost = (detector_ && detector_->device_lost()) ||
                           (landmark_ && landmark_->device_lost()) ||
                           OnnxErrorIsDeviceLost(error);
  const uint64_t now_us = NowUs();
  const InferenceRecovery::Action action =
      recovery_.RecordFailure(now_us, device_lost);
  const BackendHealth health = health_.load(std::memory_order_relaxed);
  if (action == InferenceRecovery::Action::kWait &&
      health != BackendHealth::kRecovering) {
    // One line per state change; the per-frame failures are the caller's to
    // throttle.
    const uint64_t deadline_us = recovery_.retry_deadline_us();
    const double wait_s =
        deadline_us > now_us ? (deadline_us - now_us) / 1000000.0 : 0.0;
    if (device_lost) {
      XELOGE(
          "NUI pose: the inference device was lost ({}); dropping the {} "
          "sessions and rebuilding in {:.0f} s",
          error, backend_name(), wait_s);
    } else {
      XELOGE(
          "NUI pose: {} inference failures in a row (last: {}); rebuilding "
          "the sessions in {:.0f} s",
          recovery_.consecutive_failures(), error, wait_s);
    }
    DestroySessions();
    PublishHealth(BackendHealth::kRecovering);
  } else if (action == InferenceRecovery::Action::kGiveUp &&
             health != BackendHealth::kFailed) {
    XELOGE(
        "NUI pose: inference keeps failing on the CPU ({}); giving up on pose "
        "estimation for the rest of this run",
        error);
    DestroySessions();
    PublishHealth(BackendHealth::kFailed);
  }
}

void OnnxPoseEstimator::PublishHealth(BackendHealth health) {
  health_.store(health, std::memory_order_relaxed);
}

// Detector pre-processing is ImageToTensorCalculator (224x224,
// keep_aspect_ratio, output range [-1, 1], border_mode: BORDER_ZERO as set
// in mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt) on the whole
// frame:
// PadRoi grows the frame to a square of side max(W, H) centred on it, so a
// 4:3 frame gets letterbox bands of (1 - H/W) / 2 top and bottom; the
// detections are mapped back with DetectionLetterboxRemovalCalculator.
bool OnnxPoseEstimator::RunDetector(const uint8_t* rgba, uint32_t width,
                                    uint32_t height, uint32_t stride,
                                    std::string* out_error) {
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  const float side = std::max(w, h);
  WarpCrop(rgba, width, height, stride, w * 0.5f, h * 0.5f, side, side, 0.0f,
           kDetectorInputSize, 2.0f / 255.0f, -1.0f, detector_nchw_,
           /*replicate_border=*/false, detector_input_.data());

  // Refilled rather than rebuilt: nothing here allocates after the first
  // frame, and the layout is re-read in case a rebuild changed the model.
  detector_inputs_.resize(1);
  detector_inputs_[0] = detector_input_.data();
  detector_input_shapes_.resize(1);
  std::vector<int64_t>& shape = detector_input_shapes_[0];
  shape.resize(4);
  shape[0] = 1;
  shape[1] = detector_nchw_ ? 3 : kDetectorInputSize;
  shape[2] = kDetectorInputSize;
  shape[3] = detector_nchw_ ? kDetectorInputSize : 3;
  std::string error;
  if (!detector_->Run(detector_inputs_, detector_input_shapes_,
                      &detector_outputs_, &detector_shapes_, &error)) {
    if (out_error) {
      *out_error = fmt::format("pose_detection.onnx: {}", error);
    }
    return false;
  }
  const auto& boxes = detector_outputs_[detector_boxes_index_];
  const auto& scores = detector_outputs_[detector_scores_index_];
  if (boxes.size() < static_cast<size_t>(kDetectorAnchorCount) *
                         kDetectorValuesPerAnchor ||
      scores.size() < kDetectorAnchorCount) {
    if (out_error) {
      *out_error =
          fmt::format("pose_detection.onnx returned {} box and {} score values",
                      boxes.size(), scores.size());
    }
    return false;
  }
  DecodeDetections(boxes.data(), scores.data(), anchors_xy_.data(),
                   &detection_scratch_, &detections_);

  // Letterbox removal: padding [left, top, right, bottom] as fractions of
  // the square input.
  const float pad_x = (side - w) * 0.5f / side;
  const float pad_y = (side - h) * 0.5f / side;
  const float scale_x = 1.0f - 2.0f * pad_x;
  const float scale_y = 1.0f - 2.0f * pad_y;
  for (Detection& det : detections_) {
    det.box[0] = (det.box[0] - pad_x) / scale_x;
    det.box[2] = (det.box[2] - pad_x) / scale_x;
    det.box[1] = (det.box[1] - pad_y) / scale_y;
    det.box[3] = (det.box[3] - pad_y) / scale_y;
    for (auto& kp : det.keypoints) {
      kp[0] = (kp[0] - pad_x) / scale_x;
      kp[1] = (kp[1] - pad_y) / scale_y;
    }
  }
  return true;
}

// Landmark model on one region:
//  - crop: ImageToTensorCalculator 256x256, [0, 1], rotated square ROI;
//  - TensorsToLandmarksCalculator (39 landmarks, input 256x256, SIGMOID
//    visibility/presence): x = raw/256, y = raw/256, z = raw/256;
//  - RefineLandmarksFromHeatmapCalculator (kernel 7, min confidence 0.5);
//  - LandmarkProjectionCalculator (square ROI): rotate about the crop centre
//    and scale by the region size; z scales with the region width;
//  - WorldLandmarkProjectionCalculator: rotate world x/y by the ROI angle;
//  - the next ROI comes from auxiliary landmarks 33 (hip centre) and 34
//    (size/rotation point) through the same alignment-points rule.
bool OnnxPoseEstimator::RunLandmarks(const uint8_t* rgba, uint32_t width,
                                     uint32_t height, uint32_t stride,
                                     const PoseRegion& region,
                                     PoseResult* out_result,
                                     PoseRegion* out_next_region,
                                     std::string* out_error) {
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  // mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt sets
  // output 256x256, keep_aspect_ratio and range [0, 1] but leaves
  // border_mode unset, which ImageToTensorCalculator treats as
  // BORDER_REPLICATE (see WarpCrop).
  WarpCrop(rgba, width, height, stride, region.center_x * w,
           region.center_y * h, region.width * w, region.height * h,
           region.rotation, kLandmarkInputSize, 1.0f / 255.0f, 0.0f,
           landmark_nchw_, /*replicate_border=*/true, landmark_input_.data());

  landmark_inputs_.resize(1);
  landmark_inputs_[0] = landmark_input_.data();
  landmark_input_shapes_.resize(1);
  std::vector<int64_t>& shape = landmark_input_shapes_[0];
  shape.resize(4);
  shape[0] = 1;
  shape[1] = landmark_nchw_ ? 3 : kLandmarkInputSize;
  shape[2] = kLandmarkInputSize;
  shape[3] = landmark_nchw_ ? kLandmarkInputSize : 3;
  std::string error;
  if (!landmark_->Run(landmark_inputs_, landmark_input_shapes_,
                      &landmark_outputs_, &landmark_shapes_, &error)) {
    if (out_error) {
      *out_error = fmt::format("{}: {}", landmark_model_name_, error);
    }
    return false;
  }
  const auto& points = landmark_outputs_[landmark_points_index_];
  const auto& flag = landmark_outputs_[landmark_flag_index_];
  const auto& world = landmark_outputs_[landmark_world_index_];
  if (points.size() < kLandmarkModelCount * kLandmarkValues || flag.empty() ||
      world.size() < kLandmarkModelCount * 3) {
    if (out_error) {
      *out_error = fmt::format(
          "{} returned {}/{}/{} landmark/flag/world values",
          landmark_model_name_, points.size(), flag.size(), world.size());
    }
    return false;
  }
  const float* heatmap = nullptr;
  if (landmark_heatmap_index_ >= 0) {
    const auto& hm = landmark_outputs_[landmark_heatmap_index_];
    if (hm.size() >= static_cast<size_t>(kHeatmapSize) * kHeatmapSize *
                         kLandmarkModelCount) {
      heatmap = hm.data();
    }
  }

  const float c = std::cos(region.rotation);
  const float s = std::sin(region.rotation);
  const float inv_size = 1.0f / static_cast<float>(kLandmarkInputSize);
  const int kernel_offset = static_cast<int>(kHeatmapKernel - 1) / 2;
  PoseLandmark aux[2];
  for (uint32_t i = 0; i < kLandmarkModelCount; ++i) {
    if (i >= kPoseLandmarkCount + 2) {
      break;  // 35..38 are unused
    }
    const float* raw = points.data() + static_cast<size_t>(i) * kLandmarkValues;
    float x = raw[0] * inv_size;
    float y = raw[1] * inv_size;
    const float z = raw[2] * inv_size;
    const float visibility = Sigmoid(raw[3]);
    const float presence = Sigmoid(raw[4]);

    if (heatmap) {
      const int col = static_cast<int>(x * kHeatmapSize);
      const int row = static_cast<int>(y * kHeatmapSize);
      if (col >= 0 && col < static_cast<int>(kHeatmapSize) && row >= 0 &&
          row < static_cast<int>(kHeatmapSize)) {
        const int r0 = std::max(0, row - kernel_offset);
        const int r1 =
            std::min(static_cast<int>(kHeatmapSize), row + kernel_offset + 1);
        const int c0 = std::max(0, col - kernel_offset);
        const int c1 =
            std::min(static_cast<int>(kHeatmapSize), col + kernel_offset + 1);
        float sum = 0.0f;
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float max_conf = 0.0f;
        for (int r = r0; r < r1; ++r) {
          for (int cc = c0; cc < c1; ++cc) {
            const float conf =
                Sigmoid(heatmap[(static_cast<size_t>(r) * kHeatmapSize + cc) *
                                    kLandmarkModelCount +
                                i]);
            sum += conf;
            sum_x += conf * static_cast<float>(cc);
            sum_y += conf * static_cast<float>(r);
            max_conf = std::max(max_conf, conf);
          }
        }
        if (max_conf >= kHeatmapMinConfidence && sum > 0.0f) {
          x = sum_x / sum / static_cast<float>(kHeatmapSize);
          y = sum_y / sum / static_cast<float>(kHeatmapSize);
        }
      }
    }

    // Projection into the full image.
    const float xr = x - 0.5f;
    const float yr = y - 0.5f;
    PoseLandmark landmark;
    landmark.x = (c * xr - s * yr) * region.width + region.center_x;
    landmark.y = (s * xr + c * yr) * region.height + region.center_y;
    landmark.z = z * region.width;
    landmark.visibility = visibility;
    landmark.presence = presence;
    if (i < kPoseLandmarkCount) {
      out_result->landmarks[i] = landmark;
      const float* wr = world.data() + static_cast<size_t>(i) * 3;
      PoseWorldLandmark& wl = out_result->world_landmarks[i];
      wl.x = c * wr[0] - s * wr[1];
      wl.y = s * wr[0] + c * wr[1];
      wl.z = wr[2];
    } else {
      aux[i - kPoseLandmarkCount] = landmark;
    }
  }

  out_result->pose_score = flag[0];
  out_result->valid = out_result->pose_score >= kMinPoseScore;
  out_result->region = region;
  out_result->region.score = out_result->pose_score;
  *out_next_region = RegionFromAlignmentPoints(aux[0].x, aux[0].y, aux[1].x,
                                               aux[1].y, width, height);

  out_result->has_segmentation = false;
  if (want_segmentation_ && out_result->valid &&
      landmark_segmentation_index_ >= 0) {
    const auto& seg = landmark_outputs_[landmark_segmentation_index_];
    if (seg.size() >= mask_.size()) {
      for (size_t i = 0; i < mask_.size(); ++i) {
        mask_[i] = Sigmoid(seg[i]);
      }
      ResampleSegmentation(region, width, height, &out_result->segmentation);
      out_result->has_segmentation = true;
    }
  }
  if (!out_result->has_segmentation) {
    out_result->segmentation.clear();
  }
  return true;
}

// The crop mask only exists inside the ROI square: for every depth-sized
// output pixel the full-image point is mapped through the inverse crop
// transform (u = (cos*(x-cx) + sin*(y-cy)) / w + 0.5,
// v = (-sin*(x-cx) + cos*(y-cy)) / h + 0.5) and the mask sampled bilinearly;
// points outside [0,1]^2 are background.
void OnnxPoseEstimator::ResampleSegmentation(const PoseRegion& region,
                                             uint32_t width, uint32_t height,
                                             std::vector<uint8_t>* out_mask) {
  const size_t pixel_count = static_cast<size_t>(kDepthWidth) * kDepthHeight;
  if (out_mask->capacity() < pixel_count && !segmentation_pool_.empty()) {
    // A buffer an earlier frame handed back.
    *out_mask = std::move(segmentation_pool_.back());
    segmentation_pool_.pop_back();
  }
  out_mask->resize(pixel_count);
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  const float c = std::cos(region.rotation);
  const float s = std::sin(region.rotation);
  const float cx = region.center_x * w;
  const float cy = region.center_y * h;
  const float w_px = std::max(region.width * w, 1e-3f);
  const float h_px = std::max(region.height * h, 1e-3f);
  const float dx_dpx = w / static_cast<float>(kDepthWidth);
  const float dy_dpy = h / static_cast<float>(kDepthHeight);
  // u and v are affine in the output pixel: step along a row.
  const float du_dpx = c * dx_dpx / w_px;
  const float dv_dpx = -s * dx_dpx / h_px;
  const float size = static_cast<float>(kLandmarkInputSize);
  const int max_index = static_cast<int>(kLandmarkInputSize) - 1;

  for (uint32_t py = 0; py < kDepthHeight; ++py) {
    const float x0 = 0.5f * dx_dpx - cx;
    const float y0 = (static_cast<float>(py) + 0.5f) * dy_dpy - cy;
    float u = (c * x0 + s * y0) / w_px + 0.5f;
    float v = (-s * x0 + c * y0) / h_px + 0.5f;
    uint8_t* row = out_mask->data() + static_cast<size_t>(py) * kDepthWidth;
    for (uint32_t px = 0; px < kDepthWidth; ++px, u += du_dpx, v += dv_dpx) {
      if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        row[px] = 0;
        continue;
      }
      const float mx = std::min(std::max(u * size - 0.5f, 0.0f),
                                static_cast<float>(max_index));
      const float my = std::min(std::max(v * size - 0.5f, 0.0f),
                                static_cast<float>(max_index));
      const int ix = static_cast<int>(mx);
      const int iy = static_cast<int>(my);
      const int ix1 = std::min(ix + 1, max_index);
      const int iy1 = std::min(iy + 1, max_index);
      const float tx = mx - static_cast<float>(ix);
      const float ty = my - static_cast<float>(iy);
      const float* m0 =
          mask_.data() + static_cast<size_t>(iy) * kLandmarkInputSize;
      const float* m1 =
          mask_.data() + static_cast<size_t>(iy1) * kLandmarkInputSize;
      const float value = (m0[ix] * (1.0f - tx) + m0[ix1] * tx) * (1.0f - ty) +
                          (m1[ix] * (1.0f - tx) + m1[ix1] * tx) * ty;
      row[px] = static_cast<uint8_t>(value * 255.0f + 0.5f);
    }
  }
}

PoseEstimator::BackendHealth OnnxPoseEstimator::backend_health() const {
  return health_.load(std::memory_order_relaxed);
}

std::string OnnxPoseEstimator::status() const {
  switch (health_.load(std::memory_order_relaxed)) {
    case BackendHealth::kRecovering:
      return "pose estimation lost the GPU, retrying";
    case BackendHealth::kDegradedCpu:
      return "pose estimation on CPU after a GPU reset";
    case BackendHealth::kFailed:
      return "pose estimation stopped after a GPU reset";
    case BackendHealth::kOk:
      break;
  }
  return std::string();
}

// The provider of the last session that loaded, so it still names the
// backend while the sessions are being rebuilt (and names the CPU after a
// fallback).
std::string OnnxPoseEstimator::backend_name() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return backend_name_;
}

std::string OnnxPoseEstimator::model_name() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return landmark_model_name_;
}

double OnnxPoseEstimator::last_inference_ms() const {
  return last_inference_ms_.load(std::memory_order_relaxed);
}

}  // namespace nui
}  // namespace xe
