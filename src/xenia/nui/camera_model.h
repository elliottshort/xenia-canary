/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_CAMERA_MODEL_H_
#define XENIA_NUI_CAMERA_MODEL_H_

#include <cstdint>

#include "xenia/nui/nui_types.h"

namespace xe {
namespace nui {

// Pinhole model of the Kinect depth camera at the working resolution
// (320x240, nominal focal length 285.63 px). Maps between skeleton space and
// depth-image pixels exactly like NuiTransformSkeletonToDepthImage /
// NuiTransformDepthImageToSkeleton do, so joints land on the pixels the
// synthesized depth image renders them at.
struct DepthCameraModel {
  uint32_t width = kDepthWidth;
  uint32_t height = kDepthHeight;
  float focal_px = kDepthNominalFocalLengthPx;

  // Skeleton-space point (metres) to depth pixel; returns false when the
  // point is behind the camera. |depth_mm| receives the Z distance.
  bool Project(const Vec4& point, float* out_u, float* out_v,
               uint16_t* out_depth_mm) const;

  // Depth pixel (with a millimetre reading) to skeleton space.
  Vec4 Unproject(float u, float v, uint16_t depth_mm) const;
};

// Rotates a level-sensor-space point into the frame of a sensor tilted by
// |tilt_degrees| (positive = looking up) about its X axis.
Vec4 ApplyTilt(const Vec4& point, float tilt_degrees);

// Floor clip plane (A, B, C, D) and normal-to-gravity for a sensor at
// |height_m| above the floor, tilted by |tilt_degrees|. A height of 0 means
// unknown and yields a zero plane, as the real sensor reports before it has
// found the floor.
void ComputeFloorPlane(float height_m, float tilt_degrees,
                       Vec4* out_floor_clip_plane, Vec4* out_normal_to_gravity);

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_CAMERA_MODEL_H_
