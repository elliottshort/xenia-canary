/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/depth_synthesizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace xe {
namespace nui {

namespace {

struct Bone {
  Joint a;
  Joint b;
  float radius_m;
};

// Kinect's 19 bones, with capsule radii approximating an adult body.
constexpr Bone kBones[] = {
    {Joint::kHipCenter, Joint::kSpine, 0.13f},
    {Joint::kSpine, Joint::kShoulderCenter, 0.13f},
    {Joint::kShoulderCenter, Joint::kHead, 0.10f},
    {Joint::kShoulderCenter, Joint::kShoulderLeft, 0.07f},
    {Joint::kShoulderLeft, Joint::kElbowLeft, 0.05f},
    {Joint::kElbowLeft, Joint::kWristLeft, 0.045f},
    {Joint::kWristLeft, Joint::kHandLeft, 0.04f},
    {Joint::kShoulderCenter, Joint::kShoulderRight, 0.07f},
    {Joint::kShoulderRight, Joint::kElbowRight, 0.05f},
    {Joint::kElbowRight, Joint::kWristRight, 0.045f},
    {Joint::kWristRight, Joint::kHandRight, 0.04f},
    {Joint::kHipCenter, Joint::kHipLeft, 0.10f},
    {Joint::kHipLeft, Joint::kKneeLeft, 0.08f},
    {Joint::kKneeLeft, Joint::kAnkleLeft, 0.06f},
    {Joint::kAnkleLeft, Joint::kFootLeft, 0.04f},
    {Joint::kHipCenter, Joint::kHipRight, 0.10f},
    {Joint::kHipRight, Joint::kKneeRight, 0.08f},
    {Joint::kKneeRight, Joint::kAnkleRight, 0.06f},
    {Joint::kAnkleRight, Joint::kFootRight, 0.04f},
};

constexpr float kPi = 3.14159265358979f;

}  // namespace

DepthSynthesizer::DepthSynthesizer() {
  background_.resize(kDepthWidth * kDepthHeight, 0);
}

void DepthSynthesizer::RenderBackground(const Options& options,
                                        uint16_t* depth_mm) {
  const size_t count = kDepthWidth * kDepthHeight;
  if (!options.render_room) {
    std::memset(depth_mm, 0, count * sizeof(uint16_t));
    return;
  }
  if (!background_valid_ ||
      background_options_.room_depth_m != options.room_depth_m ||
      background_options_.camera_height_m != options.camera_height_m ||
      background_options_.tilt_degrees != options.tilt_degrees) {
    background_options_ = options;
    background_valid_ = true;
    // Ray-cast every pixel against the floor plane and a back wall, both
    // expressed in the tilted sensor frame.
    Vec4 floor_plane;
    Vec4 up;
    ComputeFloorPlane(std::max(options.camera_height_m, 0.0f),
                      options.tilt_degrees, &floor_plane, &up);
    const bool have_floor = options.camera_height_m > 0.0f;
    // Back wall: plane perpendicular to the level forward axis at
    // room_depth_m. Level forward in the tilted frame:
    const float theta = options.tilt_degrees * kPi / 180.0f;
    const Vec4 forward = {0.0f, -std::sin(theta), std::cos(theta), 0.0f};
    for (uint32_t v = 0; v < kDepthHeight; ++v) {
      for (uint32_t u = 0; u < kDepthWidth; ++u) {
        // Ray direction for this pixel (unit Z).
        float dx = (u + 0.5f - kDepthWidth * 0.5f) / camera_.focal_px;
        float dy = (kDepthHeight * 0.5f - (v + 0.5f)) / camera_.focal_px;
        float dz = 1.0f;
        float best_t = -1.0f;
        if (have_floor) {
          // n . (t * d) + D = 0  ->  t = -D / (n . d)
          float denom = up.x * dx + up.y * dy + up.z * dz;
          if (denom < -1e-6f) {
            float t = -floor_plane.w / denom;
            if (t > 0.0f) {
              best_t = t;
            }
          }
        }
        {
          float denom = forward.x * dx + forward.y * dy + forward.z * dz;
          if (denom > 1e-6f) {
            float t = options.room_depth_m / denom;
            if (t > 0.0f && (best_t < 0.0f || t < best_t)) {
              best_t = t;
            }
          }
        }
        uint16_t value = 0;
        if (best_t > 0.0f) {
          float mm = best_t * dz * 1000.0f;
          if (mm >= kDepthMinMillimetres && mm <= kDepthMaxMillimetres) {
            value = static_cast<uint16_t>(mm);
          }
        }
        background_[v * kDepthWidth + u] = value;
      }
    }
  }
  std::memcpy(depth_mm, background_.data(), count * sizeof(uint16_t));
}

void DepthSynthesizer::RenderCapsule(const Vec4& a, const Vec4& b,
                                     float radius_m, uint8_t player,
                                     uint16_t* depth_mm, uint8_t* player_mask) {
  float au, av, bu, bv;
  uint16_t ad, bd;
  if (!camera_.Project(a, &au, &av, &ad) || !camera_.Project(b, &bu, &bv, &bd)) {
    return;
  }
  const float ra = radius_m * camera_.focal_px / std::max(a.z, 0.01f);
  const float rb = radius_m * camera_.focal_px / std::max(b.z, 0.01f);
  const float rmax = std::max(ra, rb);
  int x0 = static_cast<int>(std::floor(std::min(au, bu) - rmax));
  int x1 = static_cast<int>(std::ceil(std::max(au, bu) + rmax));
  int y0 = static_cast<int>(std::floor(std::min(av, bv) - rmax));
  int y1 = static_cast<int>(std::ceil(std::max(av, bv) + rmax));
  x0 = std::max(x0, 0);
  y0 = std::max(y0, 0);
  x1 = std::min(x1, static_cast<int>(kDepthWidth) - 1);
  y1 = std::min(y1, static_cast<int>(kDepthHeight) - 1);
  if (x0 > x1 || y0 > y1) {
    return;
  }
  const float dx = bu - au;
  const float dy = bv - av;
  const float len2 = dx * dx + dy * dy;
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      float px = x + 0.5f - au;
      float py = y + 0.5f - av;
      float t = len2 > 1e-6f ? (px * dx + py * dy) / len2 : 0.0f;
      t = std::clamp(t, 0.0f, 1.0f);
      float cx = px - t * dx;
      float cy = py - t * dy;
      float d2 = cx * cx + cy * cy;
      float r = ra + (rb - ra) * t;
      if (d2 > r * r) {
        continue;
      }
      // Depth along the capsule with a spherical bulge towards the camera.
      float z = a.z + (b.z - a.z) * t;
      float bulge = radius_m * std::sqrt(std::max(0.0f, 1.0f - d2 / (r * r)));
      float mm = (z - bulge) * 1000.0f;
      if (mm < 1.0f) {
        continue;
      }
      uint16_t value = static_cast<uint16_t>(std::min(mm, 65535.0f));
      size_t index = static_cast<size_t>(y) * kDepthWidth + x;
      uint16_t existing = depth_mm[index];
      // Nearer surface wins; background (0 = unknown) always loses.
      if (existing == 0 || player_mask[index] == 0 || value < existing) {
        depth_mm[index] = value;
        player_mask[index] = player;
      }
    }
  }
}

void DepthSynthesizer::RenderEllipse(const Vec4& center, float rx_m,
                                     float ry_m, uint8_t player,
                                     uint16_t* depth_mm, uint8_t* player_mask) {
  float cu, cv;
  uint16_t cd;
  if (!camera_.Project(center, &cu, &cv, &cd)) {
    return;
  }
  const float rx = rx_m * camera_.focal_px / std::max(center.z, 0.01f);
  const float ry = ry_m * camera_.focal_px / std::max(center.z, 0.01f);
  int x0 = std::max(static_cast<int>(std::floor(cu - rx)), 0);
  int x1 = std::min(static_cast<int>(std::ceil(cu + rx)),
                    static_cast<int>(kDepthWidth) - 1);
  int y0 = std::max(static_cast<int>(std::floor(cv - ry)), 0);
  int y1 = std::min(static_cast<int>(std::ceil(cv + ry)),
                    static_cast<int>(kDepthHeight) - 1);
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      float nx = (x + 0.5f - cu) / std::max(rx, 0.5f);
      float ny = (y + 0.5f - cv) / std::max(ry, 0.5f);
      if (nx * nx + ny * ny > 1.0f) {
        continue;
      }
      size_t index = static_cast<size_t>(y) * kDepthWidth + x;
      uint16_t existing = depth_mm[index];
      if (existing == 0 || player_mask[index] == 0 || cd < existing) {
        depth_mm[index] = cd;
        player_mask[index] = player;
      }
    }
  }
}

void DepthSynthesizer::Render(const std::array<Skeleton, kMaxSkeletons>& bodies,
                              uint32_t body_count, const Options& options,
                              uint16_t* depth_mm, uint8_t* player_mask) {
  RenderBackground(options, depth_mm);
  std::memset(player_mask, 0, kDepthWidth * kDepthHeight);
  body_count = std::min(body_count, kMaxSkeletons);
  for (uint32_t i = 0; i < body_count; ++i) {
    const Skeleton& body = bodies[i];
    const uint8_t player = static_cast<uint8_t>(i + 1);
    if (body.state == SkeletonState::kTracked) {
      for (const Bone& bone : kBones) {
        const Vec4& a = body.joints[static_cast<size_t>(bone.a)];
        const Vec4& b = body.joints[static_cast<size_t>(bone.b)];
        JointState sa = body.joint_states[static_cast<size_t>(bone.a)];
        JointState sb = body.joint_states[static_cast<size_t>(bone.b)];
        if (sa == JointState::kNotTracked || sb == JointState::kNotTracked) {
          continue;
        }
        RenderCapsule(a, b, bone.radius_m, player, depth_mm, player_mask);
      }
      // Head: a sphere on top of the neck bone.
      const Vec4& head = body.joints[static_cast<size_t>(Joint::kHead)];
      if (body.joint_states[static_cast<size_t>(Joint::kHead)] !=
          JointState::kNotTracked) {
        RenderCapsule(head, head, 0.11f, player, depth_mm, player_mask);
      }
    } else if (body.state == SkeletonState::kPositionOnly &&
               options.render_position_only) {
      RenderEllipse(body.position, 0.25f, 0.85f, player, depth_mm,
                    player_mask);
    }
  }
}

}  // namespace nui
}  // namespace xe
