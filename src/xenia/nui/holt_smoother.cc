/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/holt_smoother.h"

#include <algorithm>
#include <cmath>

namespace xe {
namespace nui {

namespace {

Vec4 Sub(const Vec4& a, const Vec4& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z, 0.0f};
}
Vec4 Add(const Vec4& a, const Vec4& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w};
}
Vec4 Scale(const Vec4& a, float s) { return {a.x * s, a.y * s, a.z * s, a.w}; }
float Length(const Vec4& a) {
  return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
// (1 - t) * a + t * b
Vec4 Lerp(const Vec4& a, const Vec4& b, float t) {
  return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t,
          a.w};
}

}  // namespace

HoltSmoother::HoltSmoother() { Reset(); }

void HoltSmoother::Reset() {
  for (auto& history : histories_) {
    history = SkeletonHistory();
  }
}

HoltSmoother::SkeletonHistory* HoltSmoother::FindOrAllocate(
    uint32_t tracking_id) {
  for (auto& history : histories_) {
    if (history.tracking_id == tracking_id) {
      return &history;
    }
  }
  // Reuse a slot whose skeleton is gone; the caller marks live ids first.
  for (auto& history : histories_) {
    if (history.tracking_id == kInvalidTrackingId) {
      history = SkeletonHistory();
      history.tracking_id = tracking_id;
      return &history;
    }
  }
  return nullptr;
}

void HoltSmoother::SmoothJoint(JointHistory& h, JointState state,
                               Vec4* position, const SmoothParameters& p) {
  const Vec4 raw = *position;
  Vec4 filtered;
  Vec4 trend;

  if (state == JointState::kNotTracked) {
    h.frame_count = 0;
  }

  if (h.frame_count == 0) {
    filtered = raw;
    trend = Vec4();
    h.frame_count = 1;
  } else if (h.frame_count == 1) {
    filtered = Scale(Add(raw, h.raw_position), 0.5f);
    Vec4 diff = Sub(filtered, h.filtered_position);
    trend = Add(Scale(diff, p.correction), Scale(h.trend, 1.0f - p.correction));
    h.frame_count = 2;
  } else {
    // Jitter filter: small movements are attenuated.
    Vec4 diff = Sub(raw, h.filtered_position);
    float diff_len = Length(diff);
    if (diff_len <= p.jitter_radius) {
      float t = p.jitter_radius > 0.0f ? diff_len / p.jitter_radius : 1.0f;
      filtered = Lerp(h.filtered_position, raw, t);
    } else {
      filtered = raw;
    }
    // Double exponential smoothing.
    Vec4 prev_prediction = Add(h.filtered_position, h.trend);
    filtered = Lerp(filtered, prev_prediction, p.smoothing);
    Vec4 delta = Sub(filtered, h.filtered_position);
    trend = Add(Scale(delta, p.correction), Scale(h.trend, 1.0f - p.correction));
  }

  // Predict forward, then clamp the deviation from the raw sample.
  Vec4 predicted = Add(filtered, Scale(trend, p.prediction));
  Vec4 deviation = Sub(predicted, raw);
  float deviation_len = Length(deviation);
  if (deviation_len > p.max_deviation_radius && deviation_len > 0.0f) {
    float t = p.max_deviation_radius / deviation_len;
    predicted = Lerp(raw, predicted, t);
  }

  h.raw_position = raw;
  h.filtered_position = filtered;
  h.trend = trend;

  predicted.w = raw.w;
  *position = predicted;
}

void HoltSmoother::Apply(SkeletonFrame* frame, const SmoothParameters& params) {
  // Drop history of skeletons no longer present.
  for (auto& history : histories_) {
    if (history.tracking_id == kInvalidTrackingId) {
      continue;
    }
    bool alive = false;
    for (const auto& skeleton : frame->skeletons) {
      if (skeleton.state == SkeletonState::kTracked &&
          skeleton.tracking_id == history.tracking_id) {
        alive = true;
        break;
      }
    }
    if (!alive) {
      history = SkeletonHistory();
    }
  }

  SmoothParameters p = params;
  p.smoothing = std::clamp(p.smoothing, 0.0f, 1.0f);
  p.correction = std::clamp(p.correction, 0.0f, 1.0f);
  p.prediction = std::max(p.prediction, 0.0f);
  p.jitter_radius = std::max(p.jitter_radius, 0.0f);
  p.max_deviation_radius = std::max(p.max_deviation_radius, 0.0f);

  for (auto& skeleton : frame->skeletons) {
    if (skeleton.state != SkeletonState::kTracked) {
      continue;
    }
    auto* history = FindOrAllocate(skeleton.tracking_id);
    if (!history) {
      continue;
    }
    for (uint32_t j = 0; j < kJointCount; ++j) {
      SmoothJoint(history->joints[j], skeleton.joint_states[j],
                  &skeleton.joints[j], p);
    }
  }
}

}  // namespace nui
}  // namespace xe
