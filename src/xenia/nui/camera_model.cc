/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/camera_model.h"

#include <cmath>

namespace xe {
namespace nui {

namespace {
constexpr float kPi = 3.14159265358979f;
}  // namespace

bool DepthCameraModel::Project(const Vec4& point, float* out_u, float* out_v,
                               uint16_t* out_depth_mm) const {
  if (point.z <= 0.001f) {
    return false;
  }
  // Skeleton +X is the sensor's left, which is the right side of the mirror
  // image the sensor produces: u grows with +X.
  *out_u = width * 0.5f + point.x * focal_px / point.z;
  *out_v = height * 0.5f - point.y * focal_px / point.z;
  float mm = point.z * 1000.0f;
  if (mm < 0.0f) {
    mm = 0.0f;
  }
  if (mm > 65535.0f) {
    mm = 65535.0f;
  }
  *out_depth_mm = static_cast<uint16_t>(mm + 0.5f);
  return true;
}

Vec4 DepthCameraModel::Unproject(float u, float v, uint16_t depth_mm) const {
  float z = depth_mm / 1000.0f;
  Vec4 out;
  out.x = (u - width * 0.5f) * z / focal_px;
  out.y = (height * 0.5f - v) * z / focal_px;
  out.z = z;
  out.w = 1.0f;
  return out;
}

Vec4 ApplyTilt(const Vec4& point, float tilt_degrees) {
  if (tilt_degrees == 0.0f) {
    return point;
  }
  // A sensor tilted up by theta has its forward axis at (0, sin, cos) and its
  // up axis at (0, cos, -sin) in the level frame; a level-frame point lands
  // in the sensor frame by dotting with those axes (R_x(theta)), so a point
  // straight ahead of a level sensor drops below the tilted sensor's axis.
  // ComputeFloorPlane, DepthSynthesizer and the webcam remap use the same
  // rotation.
  float theta = tilt_degrees * kPi / 180.0f;
  float c = std::cos(theta);
  float s = std::sin(theta);
  Vec4 out;
  out.x = point.x;
  out.y = point.y * c - point.z * s;
  out.z = point.y * s + point.z * c;
  out.w = point.w;
  return out;
}

void ComputeFloorPlane(float height_m, float tilt_degrees,
                       Vec4* out_floor_clip_plane,
                       Vec4* out_normal_to_gravity) {
  float theta = tilt_degrees * kPi / 180.0f;
  float c = std::cos(theta);
  float s = std::sin(theta);
  // World "up" expressed in the tilted sensor frame.
  Vec4 up = {0.0f, c, s, 0.0f};
  if (out_normal_to_gravity) {
    *out_normal_to_gravity = up;
  }
  if (out_floor_clip_plane) {
    if (height_m > 0.0f) {
      *out_floor_clip_plane = {0.0f, c, s, height_m};
    } else {
      *out_floor_clip_plane = Vec4();
    }
  }
}

}  // namespace nui
}  // namespace xe
