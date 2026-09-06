/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/skeleton_synthesizer.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "xenia/nui/camera_model.h"

namespace xe {
namespace nui {

namespace {

constexpr float kPi = 3.14159265358979f;

// Root depth limits (Kinect default range) and the standing-height prior
// used for position-only skeletons.
constexpr float kMinRootDepth = 0.8f;
constexpr float kMaxRootDepth = 4.0f;
constexpr float kHeightPrior = 1.70f;
// The landmark crop is 1.25 x (2 x hip-to-size-point), which is roughly the
// standing height of the person.
constexpr float kRegionToBodyHeight = 1.0f / 1.25f;

// Synthesize() has no clock; the source passes the camera frame interval in
// Options::frame_dt_seconds, clamped to this range.
constexpr float kMinFrameDt = 1.0f / 120.0f;
constexpr float kMaxFrameDt = 1.0f / 10.0f;
// One Euro filter for the root depth: 1 Hz cutoff when still, opening up
// with the speed of approach (beta), derivative smoothed at 1 Hz.
constexpr float kMinCutoffHz = 1.0f;
constexpr float kBeta = 0.5f;
constexpr float kDerivativeCutoffHz = 1.0f;

// Visibility EMA. A joint changes state only when the classified EMA has
// been away from the current state for two consecutive frames, so a single
// dropped frame never flips a state.
constexpr float kVisibilityAlpha = 0.4f;
// Joints this close to a depth-image edge set the clipped flags.
constexpr float kEdgeFraction = 0.03f;

// Vertical offset of SHOULDER_CENTER above the shoulder midpoint and the
// depth offset of HEAD behind the nose.
constexpr float kShoulderCenterLift = 0.04f;
constexpr float kNoseToHeadCentre = 0.09f;
constexpr float kFootLerp = 0.85f;

using LM = PoseLandmarkIndex;

inline const PoseLandmark& L(const PoseResult& pose, LM index) {
  return pose.landmarks[static_cast<size_t>(index)];
}
inline const PoseWorldLandmark& W(const PoseResult& pose, LM index) {
  return pose.world_landmarks[static_cast<size_t>(index)];
}
inline size_t J(Joint joint) { return static_cast<size_t>(joint); }

inline Vec4 Add(const Vec4& a, const Vec4& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, 1.0f};
}
inline Vec4 Sub(const Vec4& a, const Vec4& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z, 1.0f};
}
inline Vec4 Scale(const Vec4& a, float s) {
  return {a.x * s, a.y * s, a.z * s, 1.0f};
}
inline Vec4 Mid(const Vec4& a, const Vec4& b) { return Scale(Add(a, b), 0.5f); }
inline Vec4 Lerp(const Vec4& a, const Vec4& b, float t) {
  return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t,
          1.0f};
}

float FocalPx(const SkeletonSynthesizer::Options& options) {
  const float hfov = std::max(options.hfov_degrees, 1.0f) * kPi / 180.0f;
  return (static_cast<float>(options.image_width) * 0.5f) /
         std::tan(hfov * 0.5f);
}

float Clamp(float v, float lo, float hi) {
  return std::min(std::max(v, lo), hi);
}

JointState Classify(float visibility,
                    const SkeletonSynthesizer::Options& options) {
  if (visibility >= options.tracked_threshold) {
    return JointState::kTracked;
  }
  if (visibility >= options.inferred_threshold) {
    return JointState::kInferred;
  }
  return JointState::kNotTracked;
}

// A 2D/3D endpoint for the depth estimate: either a landmark or the
// midpoint of two.
struct Endpoint {
  float x;   // normalized image x
  float y;   // normalized image y
  float wx;  // world (metres)
  float wy;
  float wz;
  float vis;  // visibility
};

Endpoint MakeEndpoint(const PoseResult& pose, LM a) {
  const PoseLandmark& l = L(pose, a);
  const PoseWorldLandmark& w = W(pose, a);
  return {l.x, l.y, w.x, w.y, w.z, l.visibility};
}

Endpoint MakeMidpoint(const PoseResult& pose, LM a, LM b) {
  const Endpoint ea = MakeEndpoint(pose, a);
  const Endpoint eb = MakeEndpoint(pose, b);
  return {(ea.x + eb.x) * 0.5f,   (ea.y + eb.y) * 0.5f,
          (ea.wx + eb.wx) * 0.5f, (ea.wy + eb.wy) * 0.5f,
          (ea.wz + eb.wz) * 0.5f, std::min(ea.vis, eb.vis)};
}

struct DepthSample {
  float z;
  float weight;
};

// Z from one body segment: the pinhole model gives Z = f * L / l for a
// segment of lateral (image-plane) extent L metres spanning l pixels. The
// world landmarks supply L; the weight drops with foreshortening (a segment
// pointing at the camera has no usable extent) and with low visibility.
bool SegmentDepth(const Endpoint& a, const Endpoint& b, float focal_px,
                  const SkeletonSynthesizer::Options& options,
                  DepthSample* out) {
  const float px = std::hypot((a.x - b.x) * options.image_width,
                              (a.y - b.y) * options.image_height);
  const float lateral = std::hypot(a.wx - b.wx, a.wy - b.wy);
  const float full = std::hypot(lateral, a.wz - b.wz);
  if (px < 2.0f || lateral < 0.02f || full < 1e-4f) {
    return false;
  }
  const float foreshortening = lateral / full;
  const float visibility = std::min(a.vis, b.vis);
  const float weight = visibility * foreshortening * foreshortening;
  if (weight < 0.05f) {
    return false;
  }
  out->z = focal_px * lateral / px;
  out->weight = weight;
  return std::isfinite(out->z) && out->z > 0.0f;
}

float OneEuroAlpha(float cutoff_hz, float dt) {
  const float tau = 1.0f / (2.0f * kPi * cutoff_hz);
  return 1.0f / (1.0f + tau / dt);
}

}  // namespace

Vec4 ImagePointToSensorSpace(float nx, float ny, float depth_m,
                             const SkeletonSynthesizer::Options& options) {
  const float f = FocalPx(options);
  // A mirrored image puts the user's right (Kinect +X, the sensor's left)
  // on the right side of the image, so +X grows with image x.
  float x = (nx - 0.5f) * static_cast<float>(options.image_width) / f * depth_m;
  if (!options.mirrored) {
    x = -x;
  }
  const float y =
      (0.5f - ny) * static_cast<float>(options.image_height) / f * depth_m;
  return {x, y, depth_m, 1.0f};
}

SkeletonSynthesizer::SkeletonSynthesizer() = default;

void SkeletonSynthesizer::Forget(uint32_t person_key) {
  persons_.erase(person_key);
}

void SkeletonSynthesizer::Reset() { persons_.clear(); }

float SkeletonSynthesizer::last_root_depth(uint32_t person_key) const {
  auto it = persons_.find(person_key);
  if (it == persons_.end() || !it->second.valid) {
    return 0.0f;
  }
  return it->second.root_z;
}

bool SkeletonSynthesizer::Synthesize(const PoseResult& pose,
                                     uint32_t person_key,
                                     const Options& options,
                                     Skeleton* out_skeleton) {
  const float focal_px = FocalPx(options);
  const float tilt = options.tilt_degrees - options.camera_pitch_degrees;
  const float user_scale =
      options.user_scale > 0.0f ? options.user_scale : 1.0f;
  const float dt =
      std::isfinite(options.frame_dt_seconds)
          ? Clamp(options.frame_dt_seconds, kMinFrameDt, kMaxFrameDt)
          : 1.0f / 30.0f;

  *out_skeleton = Skeleton();
  out_skeleton->tracking_id = person_key;

  // Position-only fallback from the region: the crop height stands in for
  // the body height (1.70 m prior).
  auto region_position = [&]() {
    const float body_px =
        std::max(pose.region.height * static_cast<float>(options.image_height) *
                     kRegionToBodyHeight,
                 1.0f);
    const float z = Clamp(focal_px * kHeightPrior / body_px * user_scale,
                          kMinRootDepth, kMaxRootDepth);
    return ApplyTilt(ImagePointToSensorSpace(pose.region.center_x,
                                             pose.region.center_y, z, options),
                     tilt);
  };
  auto position_only = [&]() {
    out_skeleton->state = SkeletonState::kPositionOnly;
    out_skeleton->position = region_position();
    for (uint32_t j = 0; j < kJointCount; ++j) {
      out_skeleton->joints[j] = out_skeleton->position;
      out_skeleton->joint_states[j] = JointState::kNotTracked;
    }
    return false;
  };

  if (!pose.valid) {
    return position_only();
  }

  PersonState& state = persons_[person_key];
  const bool first_frame = !state.valid;

  // --- Root depth -------------------------------------------------------
  DepthSample samples[7];
  size_t sample_count = 0;
  auto add = [&](const Endpoint& a, const Endpoint& b) {
    DepthSample sample;
    if (SegmentDepth(a, b, focal_px, options, &sample)) {
      samples[sample_count++] = sample;
    }
  };
  const Endpoint shoulder_l = MakeEndpoint(pose, LM::kLeftShoulder);
  const Endpoint shoulder_r = MakeEndpoint(pose, LM::kRightShoulder);
  const Endpoint hip_l = MakeEndpoint(pose, LM::kLeftHip);
  const Endpoint hip_r = MakeEndpoint(pose, LM::kRightHip);
  add(shoulder_l, shoulder_r);
  add(hip_l, hip_r);
  add(MakeMidpoint(pose, LM::kLeftShoulder, LM::kRightShoulder),
      MakeMidpoint(pose, LM::kLeftHip, LM::kRightHip));
  add(shoulder_l, MakeEndpoint(pose, LM::kLeftElbow));
  add(shoulder_r, MakeEndpoint(pose, LM::kRightElbow));
  add(hip_l, MakeEndpoint(pose, LM::kLeftKnee));
  add(hip_r, MakeEndpoint(pose, LM::kRightKnee));

  float root_z = 0.0f;
  bool have_measurement = false;
  if (sample_count > 0) {
    // Weighted median.
    std::sort(
        samples, samples + sample_count,
        [](const DepthSample& a, const DepthSample& b) { return a.z < b.z; });
    float total = 0.0f;
    for (size_t i = 0; i < sample_count; ++i) {
      total += samples[i].weight;
    }
    float accumulated = 0.0f;
    root_z = samples[sample_count - 1].z;
    for (size_t i = 0; i < sample_count; ++i) {
      accumulated += samples[i].weight;
      if (accumulated >= total * 0.5f) {
        root_z = samples[i].z;
        break;
      }
    }
    root_z = Clamp(root_z * user_scale, kMinRootDepth, kMaxRootDepth);
    have_measurement = true;
  }

  if (have_measurement) {
    if (first_frame) {
      state.root_z = root_z;
      state.root_z_velocity = 0.0f;
      state.valid = true;
    } else {
      const float dz = (root_z - state.root_z) / dt;
      state.root_z_velocity +=
          (dz - state.root_z_velocity) * OneEuroAlpha(kDerivativeCutoffHz, dt);
      const float cutoff =
          kMinCutoffHz + kBeta * std::fabs(state.root_z_velocity);
      state.root_z += (root_z - state.root_z) * OneEuroAlpha(cutoff, dt);
    }
  } else if (!state.valid) {
    // Nothing measurable yet: start from the region prior.
    state.root_z = region_position().z;
    state.root_z_velocity = 0.0f;
    state.valid = true;
  }
  const float z_root = state.root_z;

  // --- Landmark points in the level camera frame -------------------------
  std::array<Vec4, kPoseLandmarkCount> points;
  for (uint32_t i = 0; i < kPoseLandmarkCount; ++i) {
    const PoseLandmark& l = pose.landmarks[i];
    const float z = std::max(z_root + pose.world_landmarks[i].z, 0.05f);
    points[i] = ImagePointToSensorSpace(l.x, l.y, z, options);
  }
  auto P = [&](LM index) -> const Vec4& {
    return points[static_cast<size_t>(index)];
  };
  auto V = [&](LM index) { return L(pose, index).visibility; };
  const float inferred = options.inferred_threshold;

  std::array<Vec4, kJointCount> joints{};
  std::array<float, kJointCount> visibility{};

  auto set = [&](Joint joint, const Vec4& p, float vis) {
    joints[J(joint)] = p;
    joints[J(joint)].w = 1.0f;
    visibility[J(joint)] = vis;
  };

  // BlazePose's left/right labels follow the visual side of the image it
  // saw (a "left" landmark is the one on the image's left-hand side of the
  // body as the model was trained on plain camera views). On a mirror view
  // - what this pipeline always feeds it - the model's "left" is therefore
  // the user's RIGHT, which is the Kinect's *Right joint (at +X, the
  // sensor's left). Plain camera views keep the anatomical labels.
  struct Side {
    LM left;
    LM right;
  };
  auto side = [&](LM model_left, LM model_right) {
    return options.mirrored ? Side{model_right, model_left}
                            : Side{model_left, model_right};
  };
  const Side shoulder = side(LM::kLeftShoulder, LM::kRightShoulder);
  const Side elbow = side(LM::kLeftElbow, LM::kRightElbow);
  const Side wrist = side(LM::kLeftWrist, LM::kRightWrist);
  const Side index = side(LM::kLeftIndex, LM::kRightIndex);
  const Side pinky = side(LM::kLeftPinky, LM::kRightPinky);
  const Side hip = side(LM::kLeftHip, LM::kRightHip);
  const Side knee = side(LM::kLeftKnee, LM::kRightKnee);
  const Side ankle = side(LM::kLeftAnkle, LM::kRightAnkle);
  const Side foot_index = side(LM::kLeftFootIndex, LM::kRightFootIndex);

  // Direct joints.
  set(Joint::kShoulderLeft, P(shoulder.left), V(shoulder.left));
  set(Joint::kShoulderRight, P(shoulder.right), V(shoulder.right));
  set(Joint::kElbowLeft, P(elbow.left), V(elbow.left));
  set(Joint::kElbowRight, P(elbow.right), V(elbow.right));
  set(Joint::kWristLeft, P(wrist.left), V(wrist.left));
  set(Joint::kWristRight, P(wrist.right), V(wrist.right));
  set(Joint::kHipLeft, P(hip.left), V(hip.left));
  set(Joint::kHipRight, P(hip.right), V(hip.right));
  set(Joint::kKneeLeft, P(knee.left), V(knee.left));
  set(Joint::kKneeRight, P(knee.right), V(knee.right));
  set(Joint::kAnkleLeft, P(ankle.left), V(ankle.left));
  set(Joint::kAnkleRight, P(ankle.right), V(ankle.right));

  // Feet: most of the way from the ankle to the toe.
  set(Joint::kFootLeft, Lerp(P(ankle.left), P(foot_index.left), kFootLerp),
      std::max(V(foot_index.left), 0.5f * V(ankle.left)));
  set(Joint::kFootRight, Lerp(P(ankle.right), P(foot_index.right), kFootLerp),
      std::max(V(foot_index.right), 0.5f * V(ankle.right)));

  // Hands: between index and pinky knuckles when the fingers are seen,
  // otherwise extrapolated from the forearm.
  auto hand = [&](Joint joint, LM wrist_lm, LM elbow_lm, LM index_lm,
                  LM pinky_lm) {
    if (V(index_lm) >= inferred && V(pinky_lm) >= inferred) {
      set(joint, Mid(P(index_lm), P(pinky_lm)),
          std::min(V(index_lm), V(pinky_lm)));
    } else {
      set(joint, Add(P(wrist_lm), Scale(Sub(P(wrist_lm), P(elbow_lm)), 0.25f)),
          0.5f * V(wrist_lm));
    }
  };
  hand(Joint::kHandLeft, wrist.left, elbow.left, index.left, pinky.left);
  hand(Joint::kHandRight, wrist.right, elbow.right, index.right, pinky.right);

  // Torso.
  const Vec4 hip_center = Mid(P(LM::kLeftHip), P(LM::kRightHip));
  Vec4 shoulder_center = Mid(P(LM::kLeftShoulder), P(LM::kRightShoulder));
  shoulder_center.y += kShoulderCenterLift;
  set(Joint::kHipCenter, hip_center,
      std::min(V(LM::kLeftHip), V(LM::kRightHip)));
  set(Joint::kShoulderCenter, shoulder_center,
      std::min(V(LM::kLeftShoulder), V(LM::kRightShoulder)));
  set(Joint::kSpine, Mid(hip_center, shoulder_center),
      std::min(visibility[J(Joint::kHipCenter)],
               visibility[J(Joint::kShoulderCenter)]));

  // Head: between the ears, or behind the nose.
  if (V(LM::kLeftEar) >= inferred && V(LM::kRightEar) >= inferred) {
    set(Joint::kHead, Mid(P(LM::kLeftEar), P(LM::kRightEar)),
        std::min(V(LM::kLeftEar), V(LM::kRightEar)));
  } else {
    Vec4 head = P(LM::kNose);
    head.z += kNoseToHeadCentre;
    set(Joint::kHead, head, V(LM::kNose));
  }

  // --- Joint states: EMA of visibility with 2-frame hysteresis -----------
  // kVisibilityAlpha is defined per 30 Hz frame; keep the time constant at
  // other camera rates.
  const float visibility_alpha =
      1.0f - std::pow(1.0f - kVisibilityAlpha, dt * 30.0f);
  for (uint32_t j = 0; j < kJointCount; ++j) {
    const float vis = Clamp(visibility[j], 0.0f, 1.0f);
    JointState next;
    if (first_frame) {
      state.visibility_ema[j] = vis;
      next = Classify(vis, options);
    } else {
      const JointState previous_candidate =
          Classify(state.visibility_ema[j], options);
      state.visibility_ema[j] +=
          (vis - state.visibility_ema[j]) * visibility_alpha;
      const JointState candidate = Classify(state.visibility_ema[j], options);
      const JointState last = state.last_state[j];
      next =
          (candidate != last && previous_candidate != last) ? candidate : last;
    }
    state.last_state[j] = next;
    out_skeleton->joint_states[j] = next;
  }

  // --- Tilt into the (virtual) sensor frame -----------------------------
  for (uint32_t j = 0; j < kJointCount; ++j) {
    out_skeleton->joints[j] = ApplyTilt(joints[j], tilt);
    out_skeleton->joints[j].w = 1.0f;
  }

  // --- Quality flags -----------------------------------------------------
  DepthCameraModel camera;
  const float edge_x = kEdgeFraction * static_cast<float>(camera.width);
  const float edge_y = kEdgeFraction * static_cast<float>(camera.height);
  uint32_t flags = 0;
  for (uint32_t j = 0; j < kJointCount; ++j) {
    if (out_skeleton->joint_states[j] == JointState::kNotTracked) {
      continue;
    }
    float u = 0.0f;
    float v = 0.0f;
    uint16_t depth_mm = 0;
    if (!camera.Project(out_skeleton->joints[j], &u, &v, &depth_mm)) {
      continue;
    }
    if (u < edge_x) {
      flags |= kQualityClippedLeft;
    } else if (u > static_cast<float>(camera.width) - edge_x) {
      flags |= kQualityClippedRight;
    }
    if (v < edge_y) {
      flags |= kQualityClippedTop;
    } else if (v > static_cast<float>(camera.height) - edge_y) {
      flags |= kQualityClippedBottom;
    }
  }
  auto tracked_at_least_inferred = [&](Joint joint) {
    return out_skeleton->joint_states[J(joint)] != JointState::kNotTracked;
  };
  if (!tracked_at_least_inferred(Joint::kFootLeft) &&
      !tracked_at_least_inferred(Joint::kFootRight) &&
      (tracked_at_least_inferred(Joint::kKneeLeft) ||
       tracked_at_least_inferred(Joint::kKneeRight))) {
    flags |= kQualityClippedBottom;
  }
  if (!tracked_at_least_inferred(Joint::kHead) &&
      (tracked_at_least_inferred(Joint::kShoulderLeft) ||
       tracked_at_least_inferred(Joint::kShoulderRight))) {
    flags |= kQualityClippedTop;
  }
  out_skeleton->quality_flags = flags;

  // --- Skeleton state ----------------------------------------------------
  const bool torso_seen = tracked_at_least_inferred(Joint::kHipLeft) &&
                          tracked_at_least_inferred(Joint::kHipRight) &&
                          tracked_at_least_inferred(Joint::kShoulderLeft) &&
                          tracked_at_least_inferred(Joint::kShoulderRight);
  if (!torso_seen) {
    out_skeleton->state = SkeletonState::kPositionOnly;
    out_skeleton->position = region_position();
    return false;
  }
  out_skeleton->state = SkeletonState::kTracked;
  out_skeleton->position = out_skeleton->joints[J(Joint::kHipCenter)];
  return true;
}

}  // namespace nui
}  // namespace xe
