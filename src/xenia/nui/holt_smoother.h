/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_HOLT_SMOOTHER_H_
#define XENIA_NUI_HOLT_SMOOTHER_H_

#include <array>
#include <cstdint>

#include "xenia/nui/nui_types.h"

namespace xe {
namespace nui {

// Holt double exponential smoothing of skeleton joints, as documented for
// NuiTransformSmooth in the Kinect SDK v1 ("Skeletal Joint Smoothing White
// Paper"). Keeps per-skeleton, per-joint filter state keyed on the tracking
// id, so a skeleton that disappears and returns starts from raw data again.
class HoltSmoother {
 public:
  HoltSmoother();

  void Reset();

  // Smooths every tracked skeleton in |frame| in place.
  void Apply(SkeletonFrame* frame, const SmoothParameters& params);

 private:
  struct JointHistory {
    Vec4 raw_position;
    Vec4 filtered_position;
    Vec4 trend;
    uint32_t frame_count = 0;
  };
  struct SkeletonHistory {
    uint32_t tracking_id = kInvalidTrackingId;
    std::array<JointHistory, kJointCount> joints{};
  };

  SkeletonHistory* FindOrAllocate(uint32_t tracking_id);
  void SmoothJoint(JointHistory& history, JointState state, Vec4* position,
                   const SmoothParameters& params);

  std::array<SkeletonHistory, kMaxSkeletons> histories_{};
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_HOLT_SMOOTHER_H_
