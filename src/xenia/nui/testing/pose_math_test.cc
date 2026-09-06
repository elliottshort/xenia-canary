/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cmath>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/nui/camera_model.h"
#include "xenia/nui/nui_types.h"
#include "xenia/nui/person_tracker.h"
#include "xenia/nui/pose_estimator.h"
#include "xenia/nui/skeleton_synthesizer.h"
#include "xenia/nui/sources/webcam_remap_lut.h"

namespace xe {
namespace nui {
namespace test {

namespace {

using LM = PoseLandmarkIndex;

constexpr float kPi = 3.14159265358979f;

size_t J(Joint joint) { return static_cast<size_t>(joint); }

SkeletonSynthesizer::Options DefaultOptions() {
  SkeletonSynthesizer::Options options;
  options.image_width = 640;
  options.image_height = 480;
  options.hfov_degrees = 70.0f;
  options.mirrored = true;
  return options;
}

float FocalPx(const SkeletonSynthesizer::Options& options) {
  return (options.image_width * 0.5f) /
         std::tan(options.hfov_degrees * 0.5f * kPi / 180.0f);
}

// A person standing in a T-pose facing the camera, built in camera space
// (x right, y down, z away from the camera, metres) with the hip centre at
// (0, hip_y, distance), then projected with the pinhole model of |options|
// into MediaPipe-style landmarks. BlazePose labels follow the visual side:
// its "left" landmarks are the ones on the right-hand side of the image
// (positive camera x) whether or not the image is a mirror view. On a mirror
// view that is the user's right arm.
PoseResult MakeTPose(float distance, float hip_y,
                     const SkeletonSynthesizer::Options& options) {
  PoseResult pose;
  pose.valid = true;
  pose.pose_score = 1.0f;
  const float f = FocalPx(options);
  const float w = static_cast<float>(options.image_width);
  const float h = static_cast<float>(options.image_height);
  auto place = [&](LM index, float dx, float dy, float dz) {
    const float z = distance + dz;
    PoseLandmark& lm = pose.landmarks[static_cast<size_t>(index)];
    lm.x = 0.5f + dx * f / (z * w);
    lm.y = 0.5f + (hip_y + dy) * f / (z * h);
    lm.z = dz * f / (distance * w);
    lm.visibility = 1.0f;
    lm.presence = 1.0f;
    PoseWorldLandmark& wl = pose.world_landmarks[static_cast<size_t>(index)];
    wl.x = dx;
    wl.y = dy;
    wl.z = dz;
  };
  place(LM::kNose, 0.0f, -0.62f, -0.05f);
  place(LM::kLeftEyeInner, 0.02f, -0.65f, -0.05f);
  place(LM::kLeftEye, 0.03f, -0.65f, -0.05f);
  place(LM::kLeftEyeOuter, 0.04f, -0.65f, -0.05f);
  place(LM::kRightEyeInner, -0.02f, -0.65f, -0.05f);
  place(LM::kRightEye, -0.03f, -0.65f, -0.05f);
  place(LM::kRightEyeOuter, -0.04f, -0.65f, -0.05f);
  place(LM::kLeftEar, 0.08f, -0.60f, 0.0f);
  place(LM::kRightEar, -0.08f, -0.60f, 0.0f);
  place(LM::kMouthLeft, 0.02f, -0.57f, -0.04f);
  place(LM::kMouthRight, -0.02f, -0.57f, -0.04f);
  place(LM::kLeftShoulder, 0.20f, -0.45f, 0.0f);
  place(LM::kRightShoulder, -0.20f, -0.45f, 0.0f);
  place(LM::kLeftElbow, 0.48f, -0.45f, 0.0f);
  place(LM::kRightElbow, -0.48f, -0.45f, 0.0f);
  place(LM::kLeftWrist, 0.74f, -0.45f, 0.0f);
  place(LM::kRightWrist, -0.74f, -0.45f, 0.0f);
  place(LM::kLeftPinky, 0.82f, -0.44f, 0.0f);
  place(LM::kRightPinky, -0.82f, -0.44f, 0.0f);
  place(LM::kLeftIndex, 0.83f, -0.46f, 0.0f);
  place(LM::kRightIndex, -0.83f, -0.46f, 0.0f);
  place(LM::kLeftThumb, 0.79f, -0.47f, 0.0f);
  place(LM::kRightThumb, -0.79f, -0.47f, 0.0f);
  place(LM::kLeftHip, 0.10f, 0.0f, 0.0f);
  place(LM::kRightHip, -0.10f, 0.0f, 0.0f);
  place(LM::kLeftKnee, 0.10f, 0.45f, 0.0f);
  place(LM::kRightKnee, -0.10f, 0.45f, 0.0f);
  place(LM::kLeftAnkle, 0.10f, 0.85f, 0.0f);
  place(LM::kRightAnkle, -0.10f, 0.85f, 0.0f);
  place(LM::kLeftHeel, 0.10f, 0.90f, 0.03f);
  place(LM::kRightHeel, -0.10f, 0.90f, 0.03f);
  place(LM::kLeftFootIndex, 0.10f, 0.90f, -0.15f);
  place(LM::kRightFootIndex, -0.10f, 0.90f, -0.15f);
  // Region: a square crop of 1.25 x the body height centred on the hips.
  const float body_px = f * 1.75f / distance;
  pose.region.center_x =
      pose.landmarks[static_cast<size_t>(LM::kLeftHip)].x +
      0.5f * (pose.landmarks[static_cast<size_t>(LM::kRightHip)].x -
              pose.landmarks[static_cast<size_t>(LM::kLeftHip)].x);
  pose.region.center_y = pose.landmarks[static_cast<size_t>(LM::kLeftHip)].y;
  pose.region.width = body_px * 1.25f / w;
  pose.region.height = body_px * 1.25f / h;
  pose.region.rotation = 0.0f;
  pose.region.score = 1.0f;
  return pose;
}

void SetVisibility(PoseResult* pose, LM index, float visibility) {
  pose->landmarks[static_cast<size_t>(index)].visibility = visibility;
}

PoseResult MakeRegionResult(float cx, float cy, float w, float h) {
  PoseResult result;
  result.valid = true;
  result.pose_score = 1.0f;
  result.region.center_x = cx;
  result.region.center_y = cy;
  result.region.width = w;
  result.region.height = h;
  return result;
}

}  // namespace

TEST_CASE("ImagePointToSensorSpace follows the mirror convention", "[nui]") {
  SkeletonSynthesizer::Options options = DefaultOptions();
  Vec4 centre = ImagePointToSensorSpace(0.5f, 0.5f, 2.0f, options);
  REQUIRE(std::fabs(centre.x) < 1e-5f);
  REQUIRE(std::fabs(centre.y) < 1e-5f);
  REQUIRE(centre.z == 2.0f);
  // Right of the (mirrored) image is the user's right = Kinect +X.
  Vec4 right = ImagePointToSensorSpace(0.75f, 0.5f, 2.0f, options);
  REQUIRE(right.x > 0.0f);
  // Up in the image is +Y.
  Vec4 up = ImagePointToSensorSpace(0.5f, 0.25f, 2.0f, options);
  REQUIRE(up.y > 0.0f);
  // Quarter of the width at 70 deg hfov: x = tan(hfov/2)/2 * z.
  const float expected = std::tan(35.0f * kPi / 180.0f) * 0.5f * 2.0f;
  REQUIRE(std::fabs(right.x - expected) < 1e-3f);
  // A non-mirrored image flips X.
  options.mirrored = false;
  Vec4 unmirrored = ImagePointToSensorSpace(0.75f, 0.5f, 2.0f, options);
  REQUIRE(unmirrored.x < 0.0f);
  REQUIRE(std::fabs(unmirrored.x + right.x) < 1e-5f);
}

TEST_CASE("SkeletonSynthesizer recovers the root depth of a T-pose", "[nui]") {
  SkeletonSynthesizer synthesizer;
  SkeletonSynthesizer::Options options = DefaultOptions();
  Skeleton skeleton;

  PoseResult pose = MakeTPose(2.0f, 0.1f, options);
  REQUIRE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  REQUIRE(skeleton.state == SkeletonState::kTracked);
  REQUIRE(skeleton.tracking_id == 1);
  const Vec4& hip = skeleton.joints[J(Joint::kHipCenter)];
  REQUIRE(std::fabs(hip.z - 2.0f) < 0.1f);
  REQUIRE(std::fabs(hip.x) < 0.02f);
  // Hips 0.1 m below the optical axis.
  REQUIRE(std::fabs(hip.y + 0.1f) < 0.02f);
  REQUIRE(std::fabs(skeleton.position.z - hip.z) < 1e-5f);
  REQUIRE(std::fabs(synthesizer.last_root_depth(1) - 2.0f) < 0.1f);

  // A second person further away, tracked independently.
  PoseResult far = MakeTPose(3.2f, 0.0f, options);
  REQUIRE(synthesizer.Synthesize(far, 2, options, &skeleton));
  REQUIRE(std::fabs(skeleton.joints[J(Joint::kHipCenter)].z - 3.2f) < 0.16f);
  REQUIRE(std::fabs(synthesizer.last_root_depth(1) - 2.0f) < 0.1f);
  REQUIRE(std::fabs(synthesizer.last_root_depth(2) - 3.2f) < 0.16f);
  REQUIRE(synthesizer.last_root_depth(3) == 0.0f);

  // Smoothing converges when the person stays put.
  for (int i = 0; i < 30; ++i) {
    REQUIRE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  }
  REQUIRE(std::fabs(skeleton.joints[J(Joint::kHipCenter)].z - 2.0f) < 0.05f);

  // user_scale multiplies the distance.
  SkeletonSynthesizer scaled;
  options.user_scale = 1.5f;
  REQUIRE(scaled.Synthesize(pose, 1, options, &skeleton));
  REQUIRE(std::fabs(skeleton.joints[J(Joint::kHipCenter)].z - 3.0f) < 0.15f);

  synthesizer.Forget(1);
  REQUIRE(synthesizer.last_root_depth(1) == 0.0f);
  synthesizer.Reset();
  REQUIRE(synthesizer.last_root_depth(2) == 0.0f);
}

TEST_CASE("SkeletonSynthesizer maps BlazePose landmarks to Kinect joints",
          "[nui]") {
  SkeletonSynthesizer synthesizer;
  SkeletonSynthesizer::Options options = DefaultOptions();
  Skeleton skeleton;
  PoseResult pose = MakeTPose(2.0f, 0.1f, options);
  REQUIRE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  const auto& joints = skeleton.joints;

  // Left/right: on the mirror view MediaPipe "left" (image right, +X) is
  // the user's right; the Kinect left joints end up at negative X.
  REQUIRE(std::fabs(joints[J(Joint::kWristLeft)].x + 0.74f) < 0.03f);
  REQUIRE(std::fabs(joints[J(Joint::kWristRight)].x - 0.74f) < 0.03f);
  REQUIRE(joints[J(Joint::kShoulderLeft)].x < 0.0f);
  REQUIRE(joints[J(Joint::kShoulderRight)].x > 0.0f);
  REQUIRE(joints[J(Joint::kHipLeft)].x < 0.0f);
  REQUIRE(joints[J(Joint::kKneeLeft)].x < 0.0f);
  REQUIRE(joints[J(Joint::kAnkleLeft)].x < 0.0f);
  REQUIRE(joints[J(Joint::kElbowLeft)].x < joints[J(Joint::kShoulderLeft)].x);
  REQUIRE(joints[J(Joint::kWristLeft)].x < joints[J(Joint::kElbowLeft)].x);

  // Vertical order, +Y up.
  REQUIRE(joints[J(Joint::kHead)].y > joints[J(Joint::kShoulderCenter)].y);
  REQUIRE(joints[J(Joint::kShoulderCenter)].y > joints[J(Joint::kSpine)].y);
  REQUIRE(joints[J(Joint::kSpine)].y > joints[J(Joint::kHipCenter)].y);
  REQUIRE(joints[J(Joint::kHipCenter)].y > joints[J(Joint::kKneeLeft)].y);
  REQUIRE(joints[J(Joint::kKneeLeft)].y > joints[J(Joint::kAnkleLeft)].y);
  REQUIRE(joints[J(Joint::kAnkleLeft)].y > joints[J(Joint::kFootLeft)].y);

  // Derived joints.
  REQUIRE(std::fabs(joints[J(Joint::kShoulderCenter)].y - (0.35f + 0.04f)) <
          0.03f);
  REQUIRE(std::fabs(joints[J(Joint::kHead)].y - 0.5f) < 0.03f);
  REQUIRE(std::fabs(joints[J(Joint::kHead)].x) < 0.02f);
  // Foot: 85% of the way from the ankle to the toe (0.15 m nearer).
  REQUIRE(std::fabs(joints[J(Joint::kFootLeft)].z -
                    (joints[J(Joint::kAnkleLeft)].z - 0.85f * 0.15f)) < 0.03f);
  // Hand: between index and pinky knuckles.
  REQUIRE(std::fabs(joints[J(Joint::kHandLeft)].x + 0.825f) < 0.03f);
  REQUIRE(std::fabs(joints[J(Joint::kHandRight)].x - 0.825f) < 0.03f);
  REQUIRE(std::fabs(joints[J(Joint::kSpine)].y -
                    0.5f * (joints[J(Joint::kHipCenter)].y +
                            joints[J(Joint::kShoulderCenter)].y)) < 1e-4f);
  for (uint32_t j = 0; j < kJointCount; ++j) {
    REQUIRE(joints[j].w == 1.0f);
    REQUIRE(skeleton.joint_states[j] == JointState::kTracked);
  }
  // At 2 m the feet of a standing adult fall below the Kinect's 45.6 deg
  // vertical field of view even though the 70 deg webcam still sees them.
  REQUIRE(skeleton.quality_flags == kQualityClippedBottom);

  // Hidden fingers (of the user's left hand, MediaPipe "right" on the
  // mirror view): the hand is extrapolated from the forearm and only
  // inferred; a hidden head falls back to the nose.
  SkeletonSynthesizer fresh;
  PoseResult partial = pose;
  SetVisibility(&partial, LM::kRightIndex, 0.0f);
  SetVisibility(&partial, LM::kRightPinky, 0.0f);
  SetVisibility(&partial, LM::kLeftEar, 0.0f);
  SetVisibility(&partial, LM::kRightEar, 0.0f);
  REQUIRE(fresh.Synthesize(partial, 1, options, &skeleton));
  REQUIRE(std::fabs(skeleton.joints[J(Joint::kHandLeft)].x + 0.805f) < 0.03f);
  REQUIRE(skeleton.joint_states[J(Joint::kHandLeft)] == JointState::kInferred);
  REQUIRE(std::fabs(skeleton.joints[J(Joint::kHead)].z -
                    (2.0f - 0.05f + 0.09f)) < 0.03f);

  // The same landmarks read as a plain camera view: MediaPipe "left" is
  // then anatomical left and the image is flipped into Kinect space, so the
  // user's left wrist still lands at negative X (only the label source
  // changes, the anatomy does not).
  SkeletonSynthesizer unmirrored;
  options.mirrored = false;
  REQUIRE(unmirrored.Synthesize(pose, 1, options, &skeleton));
  REQUIRE(std::fabs(skeleton.joints[J(Joint::kWristLeft)].x + 0.74f) < 0.03f);
  REQUIRE(std::fabs(skeleton.joints[J(Joint::kWristRight)].x - 0.74f) < 0.03f);
}

TEST_CASE("SkeletonSynthesizer maps a raised right hand to HAND_RIGHT at +X",
          "[nui]") {
  // The user raises their right hand in front of a webcam. The pipeline
  // mirrors the frame, so the hand is on the image's right (nx > 0.5) and
  // BlazePose labels it "left". The title must see HAND_RIGHT raised at +X
  // (the sensor's left).
  SkeletonSynthesizer synthesizer;
  SkeletonSynthesizer::Options options = DefaultOptions();
  PoseResult pose = MakeTPose(2.0f, 0.1f, options);
  const float f = FocalPx(options);
  auto raise = [&](LM index, float dx, float dy) {
    PoseLandmark& lm = pose.landmarks[static_cast<size_t>(index)];
    lm.x = 0.5f + dx * f / (2.0f * options.image_width);
    lm.y = 0.5f + (0.1f + dy) * f / (2.0f * options.image_height);
    PoseWorldLandmark& wl = pose.world_landmarks[static_cast<size_t>(index)];
    wl.x = dx;
    wl.y = dy;
  };
  raise(LM::kLeftElbow, 0.35f, -0.70f);
  raise(LM::kLeftWrist, 0.35f, -0.95f);
  raise(LM::kLeftIndex, 0.35f, -1.03f);
  raise(LM::kLeftPinky, 0.37f, -1.02f);
  REQUIRE(pose.landmarks[static_cast<size_t>(LM::kLeftWrist)].x > 0.5f);
  Skeleton skeleton;
  REQUIRE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  const Vec4& hand_right = skeleton.joints[J(Joint::kHandRight)];
  const Vec4& hand_left = skeleton.joints[J(Joint::kHandLeft)];
  REQUIRE(hand_right.x > 0.2f);
  REQUIRE(hand_right.y > skeleton.joints[J(Joint::kHead)].y);
  REQUIRE(hand_left.x < -0.5f);
  REQUIRE(hand_left.y < skeleton.joints[J(Joint::kShoulderLeft)].y + 0.1f);
  REQUIRE(skeleton.joints[J(Joint::kWristRight)].x > 0.2f);
  REQUIRE(skeleton.joints[J(Joint::kWristRight)].y >
          skeleton.joints[J(Joint::kShoulderRight)].y);
}

TEST_CASE("SkeletonSynthesizer joint states use hysteresis", "[nui]") {
  SkeletonSynthesizer synthesizer;
  SkeletonSynthesizer::Options options = DefaultOptions();
  Skeleton skeleton;
  PoseResult pose = MakeTPose(2.0f, 0.1f, options);
  // The user's left wrist is MediaPipe's "right" wrist on the mirror view.
  const size_t wrist = J(Joint::kWristLeft);

  REQUIRE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  REQUIRE(skeleton.joint_states[wrist] == JointState::kTracked);

  // A single dropped frame does not change the state.
  PoseResult hidden = pose;
  SetVisibility(&hidden, LM::kRightWrist, 0.0f);
  REQUIRE(synthesizer.Synthesize(hidden, 1, options, &skeleton));
  REQUIRE(skeleton.joint_states[wrist] == JointState::kTracked);
  REQUIRE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  REQUIRE(skeleton.joint_states[wrist] == JointState::kTracked);

  // Two consecutive frames do.
  REQUIRE(synthesizer.Synthesize(hidden, 1, options, &skeleton));
  REQUIRE(skeleton.joint_states[wrist] == JointState::kTracked);
  REQUIRE(synthesizer.Synthesize(hidden, 1, options, &skeleton));
  REQUIRE(skeleton.joint_states[wrist] != JointState::kTracked);
  for (int i = 0; i < 6; ++i) {
    REQUIRE(synthesizer.Synthesize(hidden, 1, options, &skeleton));
  }
  REQUIRE(skeleton.joint_states[wrist] == JointState::kNotTracked);
  // The position is still filled in.
  REQUIRE(std::fabs(skeleton.joints[wrist].x + 0.74f) < 0.03f);

  // Coming back takes two frames as well.
  REQUIRE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  REQUIRE(skeleton.joint_states[wrist] == JointState::kNotTracked);
  for (int i = 0; i < 6; ++i) {
    REQUIRE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  }
  REQUIRE(skeleton.joint_states[wrist] == JointState::kTracked);

  // Partial visibility lands on Inferred.
  SkeletonSynthesizer fresh;
  PoseResult partial = pose;
  SetVisibility(&partial, LM::kRightWrist, 0.5f);
  REQUIRE(fresh.Synthesize(partial, 1, options, &skeleton));
  REQUIRE(skeleton.joint_states[wrist] == JointState::kInferred);
}

TEST_CASE("SkeletonSynthesizer sets clipped flags", "[nui]") {
  SkeletonSynthesizer synthesizer;
  SkeletonSynthesizer::Options options = DefaultOptions();
  Skeleton skeleton;

  // Fully in view.
  PoseResult pose = MakeTPose(2.5f, 0.0f, options);
  REQUIRE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  REQUIRE(skeleton.quality_flags == 0);

  // Close and low: the feet fall off the bottom of the Kinect image.
  SkeletonSynthesizer low_synth;
  PoseResult low = MakeTPose(1.2f, 0.3f, options);
  REQUIRE(low_synth.Synthesize(low, 1, options, &skeleton));
  REQUIRE((skeleton.quality_flags & kQualityClippedBottom) != 0);
  REQUIRE((skeleton.quality_flags & kQualityClippedTop) == 0);

  // Far to the user's right: clipped on the right of the mirror image.
  SkeletonSynthesizer side_synth;
  PoseResult side = MakeTPose(2.0f, 0.0f, options);
  for (auto& lm : side.landmarks) {
    lm.x += 0.35f;
  }
  REQUIRE(side_synth.Synthesize(side, 1, options, &skeleton));
  REQUIRE((skeleton.quality_flags & kQualityClippedRight) != 0);
  REQUIRE((skeleton.quality_flags & kQualityClippedLeft) == 0);

  // Feet missing while the knees are seen: clipped at the bottom.
  SkeletonSynthesizer feet_synth;
  PoseResult no_feet = pose;
  for (LM index : {LM::kLeftAnkle, LM::kRightAnkle, LM::kLeftHeel,
                   LM::kRightHeel, LM::kLeftFootIndex, LM::kRightFootIndex}) {
    SetVisibility(&no_feet, index, 0.0f);
  }
  REQUIRE(feet_synth.Synthesize(no_feet, 1, options, &skeleton));
  REQUIRE(skeleton.joint_states[J(Joint::kFootLeft)] ==
          JointState::kNotTracked);
  REQUIRE(skeleton.joint_states[J(Joint::kKneeLeft)] == JointState::kTracked);
  REQUIRE((skeleton.quality_flags & kQualityClippedBottom) != 0);

  // Head missing while the shoulders are seen: clipped at the top.
  SkeletonSynthesizer head_synth;
  PoseResult no_head = pose;
  for (LM index : {LM::kNose, LM::kLeftEar, LM::kRightEar}) {
    SetVisibility(&no_head, index, 0.0f);
  }
  REQUIRE(head_synth.Synthesize(no_head, 1, options, &skeleton));
  REQUIRE((skeleton.quality_flags & kQualityClippedTop) != 0);
}

TEST_CASE("SkeletonSynthesizer reports position-only without a torso",
          "[nui]") {
  SkeletonSynthesizer synthesizer;
  SkeletonSynthesizer::Options options = DefaultOptions();
  Skeleton skeleton;
  PoseResult pose = MakeTPose(2.0f, 0.1f, options);
  SetVisibility(&pose, LM::kLeftHip, 0.0f);
  SetVisibility(&pose, LM::kRightHip, 0.0f);
  REQUIRE_FALSE(synthesizer.Synthesize(pose, 1, options, &skeleton));
  REQUIRE(skeleton.state == SkeletonState::kPositionOnly);
  REQUIRE(skeleton.tracking_id == 1);
  // The region height stands in for a 1.7 m body: about 2 m away, centred.
  REQUIRE(skeleton.position.z > 1.7f);
  REQUIRE(skeleton.position.z < 2.3f);
  REQUIRE(std::fabs(skeleton.position.x) < 0.05f);

  // An invalid pose is position-only too, and never touches the depth state.
  PoseResult invalid;
  invalid.valid = false;
  invalid.region.center_x = 0.25f;
  invalid.region.center_y = 0.5f;
  invalid.region.width = 0.5f;
  invalid.region.height = 0.6f;
  REQUIRE_FALSE(synthesizer.Synthesize(invalid, 7, options, &skeleton));
  REQUIRE(skeleton.state == SkeletonState::kPositionOnly);
  REQUIRE(skeleton.position.x < 0.0f);
  REQUIRE(skeleton.position.z >= 0.8f);
  REQUIRE(skeleton.position.z <= 4.0f);
  REQUIRE(synthesizer.last_root_depth(7) == 0.0f);
}

TEST_CASE("SkeletonSynthesizer applies tilt and pitch about X", "[nui]") {
  SkeletonSynthesizer::Options options = DefaultOptions();
  Skeleton level;
  Skeleton tilted;
  PoseResult pose = MakeTPose(2.0f, 0.0f, options);
  SkeletonSynthesizer a;
  REQUIRE(a.Synthesize(pose, 1, options, &level));
  options.tilt_degrees = 15.0f;
  SkeletonSynthesizer b;
  REQUIRE(b.Synthesize(pose, 1, options, &tilted));
  const Vec4& p0 = level.joints[J(Joint::kHead)];
  const Vec4& p1 = tilted.joints[J(Joint::kHead)];
  REQUIRE(std::fabs(p0.x - p1.x) < 1e-4f);
  REQUIRE(p0.y != p1.y);
  // Rotation about X preserves the Y/Z radius and matches ApplyTilt.
  REQUIRE(std::fabs((p0.y * p0.y + p0.z * p0.z) - (p1.y * p1.y + p1.z * p1.z)) <
          1e-3f);
  Vec4 expected = ApplyTilt(p0, 15.0f);
  REQUIRE(std::fabs(expected.y - p1.y) < 1e-4f);
  REQUIRE(std::fabs(expected.z - p1.z) < 1e-4f);
  // Equal tilt and physical pitch cancel out.
  options.camera_pitch_degrees = 15.0f;
  SkeletonSynthesizer c;
  REQUIRE(c.Synthesize(pose, 1, options, &tilted));
  REQUIRE(std::fabs(tilted.joints[J(Joint::kHead)].y - p0.y) < 1e-4f);
}

TEST_CASE("WebcamRemapLut samples the mask where the skeleton projects",
          "[nui]") {
  // A 16:9 camera whose segmentation is delivered as the whole frame
  // resampled to 320x240: a joint's Kinect depth pixel and the mask sample
  // the LUT picks for it must land on the same camera point.
  SkeletonSynthesizer::Options options = DefaultOptions();
  options.image_width = 1280;
  options.image_height = 720;
  options.hfov_degrees = 70.0f;
  WebcamRemapLut lut;
  lut.Update(kDepthWidth, kDepthHeight, kDepthNominalFocalLengthPx, kDepthWidth,
             kDepthHeight, options.image_width, options.image_height,
             options.hfov_degrees, 0.0f);
  REQUIRE(lut.entries.size() ==
          static_cast<size_t>(kDepthWidth) * kDepthHeight);
  DepthCameraModel camera;
  // Camera points well inside the Kinect's field of view, with vertical
  // offsets where the 4:3-versus-16:9 error would be largest.
  const float samples[][2] = {{0.5f, 0.5f},
                              {0.62f, 0.30f},
                              {0.40f, 0.68f},
                              {0.55f, 0.22f},
                              {0.47f, 0.75f}};
  for (const auto& sample : samples) {
    const float nx = sample[0];
    const float ny = sample[1];
    Vec4 point = ImagePointToSensorSpace(nx, ny, 2.0f, options);
    float u = 0.0f, v = 0.0f;
    uint16_t depth = 0;
    REQUIRE(camera.Project(point, &u, &v, &depth));
    REQUIRE(u >= 0.0f);
    REQUIRE(u < kDepthWidth);
    REQUIRE(v >= 0.0f);
    REQUIRE(v < kDepthHeight);
    const uint32_t entry = lut.entries[static_cast<size_t>(v) * kDepthWidth +
                                       static_cast<size_t>(u)];
    REQUIRE(entry != kInvalidLutEntry);
    const float sampled_nx = (entry % kDepthWidth + 0.5f) / kDepthWidth;
    const float sampled_ny = (entry / kDepthWidth + 0.5f) / kDepthHeight;
    // Within one mask pixel (1/320 and 1/240) plus the rounding of u, v.
    REQUIRE(std::fabs(sampled_nx - nx) < 2.0f / kDepthWidth);
    REQUIRE(std::fabs(sampled_ny - ny) < 2.0f / kDepthHeight);
  }
  // With the camera frame itself as the sample target (sample and camera
  // sizes equal) the table is the plain pinhole mapping.
  WebcamRemapLut color;
  color.Update(kColorWidth, kColorHeight, kColorNominalFocalLengthPx,
               options.image_width, options.image_height, options.image_width,
               options.image_height, options.hfov_degrees, 0.0f);
  const uint32_t centre =
      color.entries[static_cast<size_t>(kColorHeight / 2) * kColorWidth +
                    kColorWidth / 2];
  REQUIRE(centre != kInvalidLutEntry);
  REQUIRE(centre % options.image_width == options.image_width / 2);
  REQUIRE(centre / options.image_width == options.image_height / 2);
}

TEST_CASE("PersonTracker keeps keys stable and never reuses them", "[nui]") {
  PersonTracker tracker;
  PersonTracker::Options options;
  const PoseResult a0 = MakeRegionResult(0.30f, 0.50f, 0.40f, 0.50f);
  const PoseResult a1 = MakeRegionResult(0.32f, 0.51f, 0.40f, 0.50f);
  const PoseResult b0 = MakeRegionResult(0.75f, 0.50f, 0.35f, 0.45f);

  auto assignments = tracker.Update({a0}, 0, options);
  REQUIRE(assignments.size() == 1);
  REQUIRE(assignments[0].person_key == 1);
  REQUIRE(assignments[0].is_new);
  REQUIRE(tracker.expired_keys().empty());

  assignments = tracker.Update({a1}, 33333, options);
  REQUIRE(assignments[0].person_key == 1);
  REQUIRE_FALSE(assignments[0].is_new);

  // A second person is appended with the next key; order of results does
  // not matter.
  assignments = tracker.Update({b0, a0}, 66666, options);
  REQUIRE(assignments.size() == 2);
  REQUIRE(assignments[0].person_key == 2);
  REQUIRE(assignments[0].is_new);
  REQUIRE(assignments[1].person_key == 1);
  REQUIRE_FALSE(assignments[1].is_new);

  // A briefly leaves and returns within the timeout: same key.
  assignments = tracker.Update({b0}, 100000, options);
  REQUIRE(assignments[0].person_key == 2);
  assignments = tracker.Update({a1, b0}, 300000, options);
  REQUIRE(assignments[0].person_key == 1);
  REQUIRE_FALSE(assignments[0].is_new);
  REQUIRE(assignments[1].person_key == 2);
  REQUIRE(tracker.expired_keys().empty());

  // A leaves for longer than the timeout (B keeps being seen): A expires
  // and gets a new key on return.
  assignments = tracker.Update({b0}, 333333, options);
  REQUIRE(assignments[0].person_key == 2);
  assignments = tracker.Update({b0}, 600000, options);
  REQUIRE(assignments[0].person_key == 2);
  REQUIRE(tracker.expired_keys().empty());
  assignments = tracker.Update({b0}, 850000, options);
  REQUIRE(assignments[0].person_key == 2);
  REQUIRE(tracker.expired_keys().size() == 1);
  REQUIRE(tracker.expired_keys()[0] == 1);
  assignments = tracker.Update({a0, b0}, 883333, options);
  REQUIRE(assignments[0].person_key == 3);
  REQUIRE(assignments[0].is_new);
  REQUIRE(assignments[1].person_key == 2);
  REQUIRE(tracker.expired_keys().empty());

  // Reset forgets the tracks but keeps handing out fresh keys.
  tracker.Reset();
  assignments = tracker.Update({b0}, 916666, options);
  REQUIRE(assignments[0].person_key == 4);
  REQUIRE(assignments[0].is_new);
}

TEST_CASE("PersonTracker matches by centre distance when boxes do not overlap",
          "[nui]") {
  PersonTracker tracker;
  PersonTracker::Options options;
  // Small regions that move more than their own size between frames.
  auto assignments = tracker.Update(
      {MakeRegionResult(0.30f, 0.50f, 0.05f, 0.05f)}, 0, options);
  REQUIRE(assignments[0].person_key == 1);
  assignments = tracker.Update({MakeRegionResult(0.36f, 0.52f, 0.05f, 0.05f)},
                               33333, options);
  REQUIRE(assignments[0].person_key == 1);
  REQUIRE_FALSE(assignments[0].is_new);
  // Too far: a different person.
  assignments = tracker.Update({MakeRegionResult(0.70f, 0.52f, 0.05f, 0.05f)},
                               66666, options);
  REQUIRE(assignments[0].person_key == 2);
  REQUIRE(assignments[0].is_new);
  // Two people swapping their reported order keep their keys through IoU.
  const PoseResult p1 = MakeRegionResult(0.30f, 0.50f, 0.30f, 0.50f);
  const PoseResult p2 = MakeRegionResult(0.70f, 0.50f, 0.30f, 0.50f);
  assignments = tracker.Update({p1, p2}, 100000, options);
  const uint32_t key1 = assignments[0].person_key;
  const uint32_t key2 = assignments[1].person_key;
  REQUIRE(key1 != key2);
  assignments = tracker.Update({p2, p1}, 133333, options);
  REQUIRE(assignments[0].person_key == key2);
  REQUIRE(assignments[1].person_key == key1);
}

}  // namespace test
}  // namespace nui
}  // namespace xe
