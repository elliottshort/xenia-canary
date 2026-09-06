/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_SOURCES_WEBCAM_REMAP_LUT_H_
#define XENIA_NUI_SOURCES_WEBCAM_REMAP_LUT_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xe {
namespace nui {

constexpr uint32_t kInvalidLutEntry = 0xFFFFFFFFu;

// Builds a lookup table mapping every pixel of a Kinect-style image
// (|dst_w| x |dst_h|, focal length |dst_focal_px|) to a pixel of a sample
// buffer (|src_w| x |src_h|) that covers the whole webcam frame. The webcam
// frame is |cam_w| x |cam_h| pixels with a horizontal field of view of
// |hfov_degrees|; the sample buffer may have a different aspect ratio (the
// 320x240 segmentation of a 16:9 frame), so the projection is done in
// normalized camera coordinates and only then scaled to the buffer. The
// Kinect pixel is turned into a ray in the (tilted) sensor frame, rotated
// about X by |angle_degrees| (camera pitch minus sensor tilt) into the webcam
// frame and projected with the webcam pinhole. Entries outside the webcam
// image are kInvalidLutEntry; valid entries are row * src_w + column.
struct WebcamRemapLut {
  uint32_t dst_w = 0;
  uint32_t dst_h = 0;
  float dst_focal_px = 0.0f;
  uint32_t src_w = 0;
  uint32_t src_h = 0;
  uint32_t cam_w = 0;
  uint32_t cam_h = 0;
  float hfov_degrees = 0.0f;
  float angle_degrees = 0.0f;
  std::vector<uint32_t> entries;

  void Update(uint32_t new_dst_w, uint32_t new_dst_h, float new_dst_focal_px,
              uint32_t new_src_w, uint32_t new_src_h, uint32_t new_cam_w,
              uint32_t new_cam_h, float new_hfov_degrees,
              float new_angle_degrees) {
    if (dst_w == new_dst_w && dst_h == new_dst_h &&
        dst_focal_px == new_dst_focal_px && src_w == new_src_w &&
        src_h == new_src_h && cam_w == new_cam_w && cam_h == new_cam_h &&
        hfov_degrees == new_hfov_degrees &&
        angle_degrees == new_angle_degrees && !entries.empty()) {
      return;
    }
    dst_w = new_dst_w;
    dst_h = new_dst_h;
    dst_focal_px = new_dst_focal_px;
    src_w = new_src_w;
    src_h = new_src_h;
    cam_w = new_cam_w;
    cam_h = new_cam_h;
    hfov_degrees = new_hfov_degrees;
    angle_degrees = new_angle_degrees;
    entries.assign(static_cast<size_t>(dst_w) * dst_h, kInvalidLutEntry);
    if (!src_w || !src_h || !cam_w || !cam_h || !dst_w || !dst_h ||
        dst_focal_px <= 0.0f) {
      return;
    }
    constexpr float kPi = 3.14159265358979f;
    const float half_fov =
        std::clamp(hfov_degrees, 10.0f, 170.0f) * 0.5f * kPi / 180.0f;
    // Focal length in units of the frame width; the vertical scale follows
    // from the (square-pixel) frame aspect.
    const float focal_n = 0.5f / std::tan(half_fov);
    const float focal_ny =
        focal_n * static_cast<float>(cam_w) / static_cast<float>(cam_h);
    const float theta = angle_degrees * kPi / 180.0f;
    const float cos_t = std::cos(theta);
    const float sin_t = std::sin(theta);
    const float src_wf = static_cast<float>(src_w);
    const float src_hf = static_cast<float>(src_h);
    for (uint32_t v = 0; v < dst_h; ++v) {
      const float dy = (dst_h * 0.5f - (v + 0.5f)) / dst_focal_px;
      // Rotation about X (same convention as ApplyTilt: positive looks up).
      const float ry = dy * cos_t - sin_t;
      const float rz = dy * sin_t + cos_t;
      if (rz <= 1e-4f) {
        continue;
      }
      const float ny = 0.5f - ry / rz * focal_ny;
      if (ny < 0.0f || ny >= 1.0f) {
        continue;
      }
      const uint32_t sy =
          std::min(static_cast<uint32_t>(ny * src_hf), src_h - 1);
      const uint32_t src_row = sy * src_w;
      uint32_t* out = entries.data() + static_cast<size_t>(v) * dst_w;
      for (uint32_t u = 0; u < dst_w; ++u) {
        const float dx = (u + 0.5f - dst_w * 0.5f) / dst_focal_px;
        const float nx = 0.5f + dx / rz * focal_n;
        if (nx < 0.0f || nx >= 1.0f) {
          continue;
        }
        const uint32_t sx =
            std::min(static_cast<uint32_t>(nx * src_wf), src_w - 1);
        out[u] = src_row + sx;
      }
    }
  }
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_SOURCES_WEBCAM_REMAP_LUT_H_
