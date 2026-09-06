/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cmath>

#include "third_party/catch/include/catch.hpp"
#include "xenia/nui/camera_model.h"
#include "xenia/nui/depth_synthesizer.h"
#include "xenia/nui/holt_smoother.h"
#include "xenia/nui/nui_types.h"

namespace xe {
namespace nui {
namespace test {

TEST_CASE("DepthCameraModel projects and unprojects", "[nui]") {
  DepthCameraModel camera;
  Vec4 point = {0.5f, -0.25f, 2.0f, 1.0f};
  float u, v;
  uint16_t depth;
  REQUIRE(camera.Project(point, &u, &v, &depth));
  // +X (sensor left) lands on the right half of the mirror image.
  REQUIRE(u > kDepthWidth * 0.5f);
  REQUIRE(v > kDepthHeight * 0.5f);
  REQUIRE(depth == 2000);
  Vec4 back = camera.Unproject(u, v, depth);
  REQUIRE(std::fabs(back.x - point.x) < 0.01f);
  REQUIRE(std::fabs(back.y - point.y) < 0.01f);
  REQUIRE(std::fabs(back.z - point.z) < 0.01f);
  REQUIRE_FALSE(camera.Project({0.0f, 0.0f, -1.0f, 1.0f}, &u, &v, &depth));
}

TEST_CASE("Floor plane follows sensor height and tilt", "[nui]") {
  Vec4 plane;
  Vec4 up;
  ComputeFloorPlane(1.0f, 10.0f, &plane, &up);
  REQUIRE(std::fabs(plane.x) < 1e-5f);
  REQUIRE(std::fabs(plane.y - 0.9848f) < 1e-3f);
  REQUIRE(std::fabs(plane.z - 0.1736f) < 1e-3f);
  REQUIRE(std::fabs(plane.w - 1.0f) < 1e-5f);
  REQUIRE(std::fabs(up.y - 0.9848f) < 1e-3f);
  // Unknown height: zero plane, but gravity still known.
  ComputeFloorPlane(0.0f, 0.0f, &plane, &up);
  REQUIRE(plane.w == 0.0f);
  REQUIRE(plane.y == 0.0f);
  REQUIRE(up.y == 1.0f);
}

TEST_CASE("ApplyTilt is identity at zero and rotates about X", "[nui]") {
  Vec4 p = {0.3f, -0.8f, 2.0f, 1.0f};
  Vec4 same = ApplyTilt(p, 0.0f);
  REQUIRE(same.y == p.y);
  REQUIRE(same.z == p.z);
  // Tilting the sensor up makes a point ahead of it appear lower.
  Vec4 tilted = ApplyTilt({0.0f, 0.0f, 2.0f, 1.0f}, 20.0f);
  REQUIRE(tilted.y < 0.0f);
  REQUIRE(tilted.z < 2.0f);
  REQUIRE(std::fabs(tilted.y * tilted.y + tilted.z * tilted.z - 4.0f) < 1e-3f);
}

TEST_CASE("Holt smoother passes constants and attenuates jitter", "[nui]") {
  HoltSmoother smoother;
  SmoothParameters params;
  SkeletonFrame frame;
  auto& skeleton = frame.skeletons[0];
  skeleton.state = SkeletonState::kTracked;
  skeleton.tracking_id = 7;
  for (uint32_t j = 0; j < kJointCount; ++j) {
    skeleton.joint_states[j] = JointState::kTracked;
  }
  // Constant input converges to itself.
  for (int i = 0; i < 10; ++i) {
    skeleton.joints[0] = {1.0f, 0.5f, 2.0f, 1.0f};
    smoother.Apply(&frame, params);
  }
  REQUIRE(std::fabs(skeleton.joints[0].x - 1.0f) < 1e-3f);
  REQUIRE(std::fabs(skeleton.joints[0].z - 2.0f) < 1e-3f);
  // Small jitter (below the jitter radius) is strongly attenuated.
  float max_dev = 0.0f;
  for (int i = 0; i < 20; ++i) {
    float noise = (i & 1) ? 0.02f : -0.02f;
    skeleton.joints[0] = {1.0f + noise, 0.5f, 2.0f, 1.0f};
    smoother.Apply(&frame, params);
    max_dev = std::max(max_dev, std::fabs(skeleton.joints[0].x - 1.0f));
  }
  REQUIRE(max_dev < 0.02f);
  // Never deviates more than max_deviation_radius from the raw sample.
  skeleton.joints[0] = {2.0f, 0.5f, 2.0f, 1.0f};
  smoother.Apply(&frame, params);
  REQUIRE(std::fabs(skeleton.joints[0].x - 2.0f) <=
          params.max_deviation_radius + 1e-4f);
}

TEST_CASE("DepthSynthesizer renders a player silhouette", "[nui]") {
  DepthSynthesizer synthesizer;
  DepthSynthesizer::Options options;
  options.render_room = false;
  std::array<Skeleton, kMaxSkeletons> bodies{};
  Skeleton& body = bodies[0];
  body.state = SkeletonState::kTracked;
  body.tracking_id = 1;
  auto set = [&](Joint j, float x, float y, float z) {
    body.joints[static_cast<size_t>(j)] = {x, y, z, 1.0f};
    body.joint_states[static_cast<size_t>(j)] = JointState::kTracked;
  };
  // A simple standing figure 2 m away, sensor at chest height.
  set(Joint::kHipCenter, 0.0f, -0.2f, 2.0f);
  set(Joint::kSpine, 0.0f, 0.0f, 2.0f);
  set(Joint::kShoulderCenter, 0.0f, 0.3f, 2.0f);
  set(Joint::kHead, 0.0f, 0.5f, 2.0f);
  set(Joint::kShoulderLeft, -0.2f, 0.3f, 2.0f);
  set(Joint::kElbowLeft, -0.25f, 0.0f, 2.0f);
  set(Joint::kWristLeft, -0.25f, -0.25f, 2.0f);
  set(Joint::kHandLeft, -0.25f, -0.33f, 2.0f);
  set(Joint::kShoulderRight, 0.2f, 0.3f, 2.0f);
  set(Joint::kElbowRight, 0.25f, 0.0f, 2.0f);
  set(Joint::kWristRight, 0.25f, -0.25f, 2.0f);
  set(Joint::kHandRight, 0.25f, -0.33f, 2.0f);
  set(Joint::kHipLeft, -0.12f, -0.2f, 2.0f);
  set(Joint::kKneeLeft, -0.12f, -0.65f, 2.0f);
  set(Joint::kAnkleLeft, -0.12f, -1.05f, 2.0f);
  set(Joint::kFootLeft, -0.12f, -1.1f, 1.9f);
  set(Joint::kHipRight, 0.12f, -0.2f, 2.0f);
  set(Joint::kKneeRight, 0.12f, -0.65f, 2.0f);
  set(Joint::kAnkleRight, 0.12f, -1.05f, 2.0f);
  set(Joint::kFootRight, 0.12f, -1.1f, 1.9f);
  std::vector<uint16_t> depth(kDepthWidth * kDepthHeight);
  std::vector<uint8_t> mask(kDepthWidth * kDepthHeight);
  synthesizer.Render(bodies, 1, options, depth.data(), mask.data());
  // The spine pixel is covered by player 1 at roughly 2 m.
  DepthCameraModel camera;
  float u, v;
  uint16_t d;
  REQUIRE(camera.Project(body.joints[static_cast<size_t>(Joint::kSpine)], &u,
                         &v, &d));
  size_t index = static_cast<size_t>(v) * kDepthWidth + static_cast<size_t>(u);
  REQUIRE(mask[index] == 1);
  REQUIRE(depth[index] > 1800);
  REQUIRE(depth[index] <= 2000);
  // A corner pixel is background.
  REQUIRE(mask[0] == 0);
  REQUIRE(depth[0] == 0);
  size_t covered = 0;
  for (uint8_t m : mask) {
    covered += m ? 1 : 0;
  }
  REQUIRE(covered > 1000);
  REQUIRE(covered < kDepthWidth * kDepthHeight / 4);
}

}  // namespace test
}  // namespace nui
}  // namespace xe
