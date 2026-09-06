/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/sources/virtual_nui_source.h"

#include <algorithm>
#include <cmath>

#include "xenia/base/logging.h"
#include "xenia/hid/input.h"
#include "xenia/hid/input_system.h"
#include "xenia/nui/camera_model.h"

namespace xe {
namespace nui {

namespace {

// Rough adult proportions, metres above the floor.
constexpr float kHipHeight = 0.95f;
constexpr float kSpineHeight = 1.15f;
constexpr float kShoulderHeight = 1.45f;
constexpr float kHeadHeight = 1.65f;
constexpr float kShoulderHalfWidth = 0.20f;
constexpr float kHipHalfWidth = 0.12f;
constexpr float kUpperArm = 0.30f;
constexpr float kForearm = 0.28f;
constexpr float kHandLength = 0.08f;
constexpr float kKneeHeight = 0.50f;
constexpr float kAnkleHeight = 0.08f;

constexpr float kHandRangeX = 0.55f;
constexpr float kHandRangeY = 1.15f;
constexpr float kHandRangeZ = 0.55f;
constexpr float kWalkSpeed = 0.8f;   // m/s
constexpr float kJumpDuration = 0.5f;  // s
constexpr float kJumpHeight = 0.35f;   // m

float Deadzone(int16_t v) {
  constexpr float kDead = 0.15f;
  float f = static_cast<float>(v) / 32767.0f;
  if (std::fabs(f) < kDead) {
    return 0.0f;
  }
  return std::clamp((f - std::copysign(kDead, f)) / (1.0f - kDead), -1.0f,
                    1.0f);
}

}  // namespace

VirtualNuiSource::VirtualNuiSource(hid::InputSystem* input_system)
    : input_system_(input_system) {}

bool VirtualNuiSource::Start(const DeviceState& initial_state) {
  std::lock_guard<std::mutex> lock(mutex_);
  device_state_ = initial_state;
  started_ = true;
  start_time_ = std::chrono::steady_clock::now();
  last_update_ = start_time_;
  return true;
}

void VirtualNuiSource::Stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  started_ = false;
}

void VirtualNuiSource::SetDeviceState(const DeviceState& state) {
  std::lock_guard<std::mutex> lock(mutex_);
  device_state_ = state;
}

void VirtualNuiSource::PollInput(float dt) {
  if (!input_system_) {
    return;
  }
  hid::X_INPUT_STATE state = {};
  bool have_input = false;
  {
    auto lock = input_system_->lock();
    have_input = input_system_->GetState(0, hid::X_INPUT_FLAG_GAMEPAD,
                                         &state) == X_ERROR_SUCCESS;
  }
  if (!have_input) {
    return;
  }
  const auto& pad = state.gamepad;
  const uint16_t buttons = pad.buttons;

  pose_.left_dx = Deadzone(pad.thumb_lx) * kHandRangeX;
  pose_.left_dy = Deadzone(pad.thumb_ly) * kHandRangeY;
  pose_.right_dx = Deadzone(pad.thumb_rx) * kHandRangeX;
  pose_.right_dy = Deadzone(pad.thumb_ry) * kHandRangeY;
  pose_.left_dz = (pad.left_trigger / 255.0f) * kHandRangeZ;
  pose_.right_dz = (pad.right_trigger / 255.0f) * kHandRangeZ;

  pose_.engage = (buttons & hid::X_INPUT_GAMEPAD_A) != 0;
  pose_.engage_time = pose_.engage ? pose_.engage_time + dt : 0.0f;
  if (buttons & hid::X_INPUT_GAMEPAD_B) {
    pose_.left_dx = pose_.left_dy = pose_.left_dz = 0.0f;
    pose_.right_dx = pose_.right_dy = pose_.right_dz = 0.0f;
  }
  pose_.t_pose = (buttons & hid::X_INPUT_GAMEPAD_Y) != 0;

  const bool x_down = (buttons & hid::X_INPUT_GAMEPAD_X) != 0;
  if (x_down && !x_was_down_ && jump_time_ < 0.0f) {
    jump_time_ = 0.0f;
  }
  x_was_down_ = x_down;
  if (jump_time_ >= 0.0f) {
    jump_time_ += dt;
    float t = jump_time_ / kJumpDuration;
    if (t >= 1.0f) {
      jump_time_ = -1.0f;
      pose_.body_y_offset = 0.0f;
    } else {
      pose_.body_y_offset = kJumpHeight * 4.0f * t * (1.0f - t);
    }
  } else if (buttons & hid::X_INPUT_GAMEPAD_BACK) {
    pose_.body_y_offset = -0.35f;
  } else {
    pose_.body_y_offset = 0.0f;
  }

  float lean_target = 0.0f;
  if (buttons & hid::X_INPUT_GAMEPAD_LEFT_SHOULDER) lean_target -= 1.0f;
  if (buttons & hid::X_INPUT_GAMEPAD_RIGHT_SHOULDER) lean_target += 1.0f;
  pose_.lean += (lean_target - pose_.lean) * std::min(1.0f, dt * 8.0f);

  const bool start_down = (buttons & hid::X_INPUT_GAMEPAD_START) != 0;
  if (start_down && !start_was_down_) {
    second_player_ = !second_player_;
    XELOGI("NUI virtual source: second player {}",
           second_player_ ? "on" : "off");
  }
  start_was_down_ = start_down;

  float strafe = 0.0f, forward = 0.0f;
  if (buttons & hid::X_INPUT_GAMEPAD_DPAD_LEFT) strafe -= 1.0f;
  if (buttons & hid::X_INPUT_GAMEPAD_DPAD_RIGHT) strafe += 1.0f;
  if (buttons & hid::X_INPUT_GAMEPAD_DPAD_UP) forward -= 1.0f;
  if (buttons & hid::X_INPUT_GAMEPAD_DPAD_DOWN) forward += 1.0f;
  pose_.body_x = std::clamp(pose_.body_x + strafe * kWalkSpeed * dt, -1.2f,
                            1.2f);
  pose_.body_z =
      std::clamp(pose_.body_z + forward * kWalkSpeed * dt, 1.2f, 3.5f);
}

void VirtualNuiSource::ComputeBody(const Pose& pose, uint32_t person_key,
                                   Skeleton* out) const {
  const float sensor_height = device_state_.camera_height_m > 0.0f
                                  ? device_state_.camera_height_m
                                  : 1.0f;
  const float tilt = device_state_.tilt_degrees;
  std::array<Vec4, kJointCount> joints{};
  auto set = [&](Joint j, float x, float y, float z) {
    joints[static_cast<size_t>(j)] = {x, y - sensor_height, z, 1.0f};
  };
  const float bx = pose.body_x;
  const float bz = pose.body_z;
  const float by = pose.body_y_offset;
  const float lean_dx = pose.lean * 0.25f;

  set(Joint::kHipCenter, bx, kHipHeight + by, bz);
  set(Joint::kSpine, bx + lean_dx * 0.4f, kSpineHeight + by, bz);
  set(Joint::kShoulderCenter, bx + lean_dx, kShoulderHeight + by, bz);
  set(Joint::kHead, bx + lean_dx * 1.3f, kHeadHeight + by, bz);
  set(Joint::kShoulderLeft, bx + lean_dx - kShoulderHalfWidth,
      kShoulderHeight + by, bz);
  set(Joint::kShoulderRight, bx + lean_dx + kShoulderHalfWidth,
      kShoulderHeight + by, bz);
  set(Joint::kHipLeft, bx - kHipHalfWidth, kHipHeight + by, bz);
  set(Joint::kHipRight, bx + kHipHalfWidth, kHipHeight + by, bz);
  // Legs straight down; feet stay on the floor when crouching, leave it
  // when jumping.
  const float leg_lift = by > 0.0f ? by : 0.0f;
  const float knee = by < 0.0f ? kKneeHeight + by * 0.5f : kKneeHeight;
  set(Joint::kKneeLeft, bx - kHipHalfWidth, knee + leg_lift, bz - 0.05f);
  set(Joint::kKneeRight, bx + kHipHalfWidth, knee + leg_lift, bz - 0.05f);
  set(Joint::kAnkleLeft, bx - kHipHalfWidth, kAnkleHeight + leg_lift, bz);
  set(Joint::kAnkleRight, bx + kHipHalfWidth, kAnkleHeight + leg_lift, bz);
  set(Joint::kFootLeft, bx - kHipHalfWidth, leg_lift, bz - 0.1f);
  set(Joint::kFootRight, bx + kHipHalfWidth, leg_lift, bz - 0.1f);

  auto arm = [&](Joint shoulder, Joint elbow, Joint wrist, Joint hand,
                 float side, float dx, float dy, float dz, bool engage) {
    const Vec4 s = joints[static_cast<size_t>(shoulder)];
    const float reach = kUpperArm + kForearm + kHandLength;
    float hx, hy, hz;
    if (pose.t_pose) {
      hx = s.x + side * reach;
      hy = s.y;
      hz = s.z;
    } else {
      hx = s.x + side * 0.05f + dx;
      hy = s.y - reach + dy;
      hz = s.z - dz;
      if (engage && side > 0.0f) {
        const Vec4 head = joints[static_cast<size_t>(Joint::kHead)];
        const float sweep = 0.15f * std::sin(pose.engage_time * 3.0f);
        hx = head.x + 0.30f + sweep;
        hy = head.y - 0.40f;
        hz = head.z - 0.60f;
      }
    }
    float vx = hx - s.x, vy = hy - s.y, vz = hz - s.z;
    float len = std::sqrt(vx * vx + vy * vy + vz * vz);
    if (len > reach) {
      vx *= reach / len;
      vy *= reach / len;
      vz *= reach / len;
      len = reach;
    }
    const float inv = 1.0f / std::max(len, 1e-3f);
    const float ux = vx * inv, uy = vy * inv, uz = vz * inv;
    const float half = len * 0.5f;
    const float bend =
        std::sqrt(std::max(0.0f, kUpperArm * kUpperArm - half * half));
    joints[static_cast<size_t>(elbow)] = {
        s.x + ux * half + side * bend * 0.6f, s.y + uy * half - bend * 0.3f,
        s.z + uz * half + bend * 0.7f, 1.0f};
    joints[static_cast<size_t>(wrist)] = {s.x + ux * (len - kHandLength),
                                          s.y + uy * (len - kHandLength),
                                          s.z + uz * (len - kHandLength),
                                          1.0f};
    joints[static_cast<size_t>(hand)] = {s.x + vx, s.y + vy, s.z + vz, 1.0f};
  };
  arm(Joint::kShoulderLeft, Joint::kElbowLeft, Joint::kWristLeft,
      Joint::kHandLeft, -1.0f, pose.left_dx, pose.left_dy, pose.left_dz,
      false);
  arm(Joint::kShoulderRight, Joint::kElbowRight, Joint::kWristRight,
      Joint::kHandRight, +1.0f, pose.right_dx, pose.right_dy, pose.right_dz,
      pose.engage);

  out->state = SkeletonState::kTracked;
  out->tracking_id = person_key;
  out->enrollment_index = kInvalidUserIndex;
  out->user_index = kInvalidUserIndex;
  out->quality_flags = 0;
  for (uint32_t j = 0; j < kJointCount; ++j) {
    out->joints[j] = ApplyTilt(joints[j], tilt);
    out->joint_states[j] = JointState::kTracked;
  }
  out->position = out->joints[static_cast<size_t>(Joint::kHipCenter)];

  // Clipping flags from the depth camera frustum.
  DepthCameraModel camera;
  for (uint32_t j = 0; j < kJointCount; ++j) {
    float u, v;
    uint16_t d;
    if (!camera.Project(out->joints[j], &u, &v, &d)) {
      continue;
    }
    if (u < 0.0f) out->quality_flags |= kQualityClippedLeft;
    if (u >= kDepthWidth) out->quality_flags |= kQualityClippedRight;
    if (v < 0.0f) out->quality_flags |= kQualityClippedTop;
    if (v >= kDepthHeight) out->quality_flags |= kQualityClippedBottom;
  }
}

std::shared_ptr<SourceFrame> VirtualNuiSource::AllocateFrame() {
  for (auto& frame : pool_) {
    if (frame.use_count() == 1) {
      frame->Reset();
      return frame;
    }
  }
  auto frame = std::make_shared<SourceFrame>();
  frame->player_mask.resize(kDepthWidth * kDepthHeight);
  frame->depth_mm.resize(kDepthWidth * kDepthHeight);
  pool_.push_back(frame);
  return frame;
}

std::shared_ptr<const SourceFrame> VirtualNuiSource::AcquireLatest(
    uint64_t last_sequence) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!started_) {
    return nullptr;
  }
  auto now = std::chrono::steady_clock::now();
  float dt = std::chrono::duration<float>(now - last_update_).count();
  last_update_ = now;
  dt = std::clamp(dt, 0.0f, 0.1f);
  PollInput(dt);

  auto frame = AllocateFrame();
  frame->sequence = ++sequence_;
  frame->capture_time_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(now -
                                                            start_time_)
          .count());
  frame->body_count = 0;
  ComputeBody(pose_, 1, &frame->bodies[frame->body_count++]);
  if (second_player_) {
    Pose second;
    second.body_x = std::clamp(pose_.body_x + 0.9f, -1.2f, 1.6f);
    second.body_z = pose_.body_z;
    ComputeBody(second, 2, &frame->bodies[frame->body_count++]);
  }
  ComputeFloorPlane(device_state_.camera_height_m, device_state_.tilt_degrees,
                    &frame->floor_clip_plane, &frame->normal_to_gravity);

  if (device_state_.want_depth || device_state_.want_player_mask) {
    DepthSynthesizer::Options options;
    options.camera_height_m = device_state_.camera_height_m;
    options.tilt_degrees = device_state_.tilt_degrees;
    depth_synthesizer_.Render(frame->bodies, frame->body_count, options,
                              frame->depth_mm.data(),
                              frame->player_mask.data());
    frame->has_depth = true;
    frame->has_player_mask = true;
  }
  frames_produced_++;
  return frame;
}

void VirtualNuiSource::GetStats(SourceStats* out_stats) const {
  *out_stats = SourceStats();
  out_stats->capture_fps = 30.0;
  out_stats->frames_produced = frames_produced_;
  out_stats->status = "virtual player (gamepad/keyboard)";
}

}  // namespace nui
}  // namespace xe
