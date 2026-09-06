/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/title_hooks/virtual_kinect.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "xenia/base/byte_order.h"
#include "xenia/hid/input.h"
#include "xenia/hid/input_system.h"

namespace xe {
namespace kernel {
namespace hooks {

namespace {

constexpr float kPi = 3.14159265f;

// Rough adult proportions (metres above the floor). The sensor sits
// kSensorHeight above the floor and is the origin of skeleton space, so every
// height below is shifted down by that amount when joints are emitted.
constexpr float kSensorHeight = 1.0f;
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

// How far the sticks/triggers can move a hand from its neutral spot.
constexpr float kHandRangeX = 0.55f;
constexpr float kHandRangeY = 1.15f;
constexpr float kHandRangeZ = 0.55f;
constexpr float kWalkSpeed = 0.8f;  // m/s

float Deadzone(int16_t v) {
  constexpr float kDead = 0.15f;
  float f = static_cast<float>(v) / 32767.0f;
  if (std::fabs(f) < kDead) {
    return 0.0f;
  }
  return std::clamp((f - std::copysign(kDead, f)) / (1.0f - kDead), -1.0f,
                    1.0f);
}

void StoreF32(uint8_t* p, float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  xe::store_and_swap<uint32_t>(p, bits);
}

void StoreVector4(uint8_t* p, const VirtualKinect::Vector4& v) {
  StoreF32(p + 0, v.x);
  StoreF32(p + 4, v.y);
  StoreF32(p + 8, v.z);
  StoreF32(p + 12, v.w);
}

}  // namespace

VirtualKinect::VirtualKinect() {
  start_time_ = std::chrono::steady_clock::now();
  last_update_ = start_time_;
  last_frame_ = start_time_;
  ComputePose();
}

void VirtualKinect::Update(hid::InputSystem* input_system) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto now = std::chrono::steady_clock::now();
  float dt = std::chrono::duration<float>(now - last_update_).count();
  last_update_ = now;
  dt = std::clamp(dt, 0.0f, 0.1f);

  hid::X_INPUT_STATE state = {};
  bool have_input = false;
  if (input_system) {
    auto lock = input_system->lock();
    have_input = input_system->GetState(0, hid::X_INPUT_FLAG_GAMEPAD,
                                        &state) == X_ERROR_SUCCESS;
  }
  if (have_input) {
    const auto& pad = state.gamepad;
    uint16_t buttons = pad.buttons;
    // Hands.
    left_hand_dx_ = Deadzone(pad.thumb_lx) * kHandRangeX;
    left_hand_dy_ = Deadzone(pad.thumb_ly) * kHandRangeY;
    right_hand_dx_ = Deadzone(pad.thumb_rx) * kHandRangeX;
    right_hand_dy_ = Deadzone(pad.thumb_ry) * kHandRangeY;
    left_hand_dz_ = (pad.left_trigger / 255.0f) * kHandRangeZ;
    right_hand_dz_ = (pad.right_trigger / 255.0f) * kHandRangeZ;
    wave_ = (buttons & hid::X_INPUT_GAMEPAD_A) != 0;
    if (wave_) {
      engage_time_ += dt;
    } else {
      engage_time_ = 0.0f;
    }
    if (buttons & hid::X_INPUT_GAMEPAD_B) {
      left_hand_dx_ = left_hand_dy_ = left_hand_dz_ = 0.0f;
      right_hand_dx_ = right_hand_dy_ = right_hand_dz_ = 0.0f;
    }
    // Walking.
    float strafe = 0.0f, forward = 0.0f;
    if (buttons & hid::X_INPUT_GAMEPAD_DPAD_LEFT) strafe -= 1.0f;
    if (buttons & hid::X_INPUT_GAMEPAD_DPAD_RIGHT) strafe += 1.0f;
    if (buttons & hid::X_INPUT_GAMEPAD_DPAD_UP) forward -= 1.0f;
    if (buttons & hid::X_INPUT_GAMEPAD_DPAD_DOWN) forward += 1.0f;
    body_x_ = std::clamp(body_x_ + strafe * kWalkSpeed * dt, -1.2f, 1.2f);
    body_z_ = std::clamp(body_z_ + forward * kWalkSpeed * dt, 1.2f, 3.5f);
  }
  ComputePose();
}

bool VirtualKinect::NewFrameDue() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto now = std::chrono::steady_clock::now();
  constexpr auto kFramePeriod = std::chrono::microseconds(33333);
  if (now - last_frame_ < kFramePeriod) {
    return false;
  }
  last_frame_ = now;
  ++frame_number_;
  return true;
}

void VirtualKinect::ComputePose() {
  // Heights are given above the floor; skeleton space is sensor-relative.
  auto set = [&](Joint j, float x, float y, float z) {
    joints_[j] = {x, y - kSensorHeight, z, 1.0f};
  };
  const float bx = body_x_;
  const float bz = body_z_;

  set(kHipCenter, bx, kHipHeight, bz);
  set(kSpine, bx, kSpineHeight, bz);
  set(kShoulderCenter, bx, kShoulderHeight, bz);
  set(kHead, bx, kHeadHeight, bz);

  // NUI skeleton space: +x is the player's right as seen from the sensor.
  set(kShoulderLeft, bx - kShoulderHalfWidth, kShoulderHeight, bz);
  set(kShoulderRight, bx + kShoulderHalfWidth, kShoulderHeight, bz);
  set(kHipLeft, bx - kHipHalfWidth, kHipHeight, bz);
  set(kHipRight, bx + kHipHalfWidth, kHipHeight, bz);

  // Legs, straight down.
  set(kKneeLeft, bx - kHipHalfWidth, kKneeHeight, bz);
  set(kKneeRight, bx + kHipHalfWidth, kKneeHeight, bz);
  set(kAnkleLeft, bx - kHipHalfWidth, kAnkleHeight, bz);
  set(kAnkleRight, bx + kHipHalfWidth, kAnkleHeight, bz);
  set(kFootLeft, bx - kHipHalfWidth, 0.0f, bz - 0.1f);
  set(kFootRight, bx + kHipHalfWidth, 0.0f, bz - 0.1f);

  // Arms: neutral pose hangs at the sides; the controller offsets the hand
  // and the elbow is placed on the way there so the arm keeps a plausible
  // length.
  auto arm = [&](Joint shoulder, Joint elbow, Joint wrist, Joint hand,
                 float side, float dx, float dy, float dz, bool raise) {
    const Vector4 s = joints_[shoulder];
    float hx = s.x + side * 0.05f + dx;
    float hy = s.y - (kUpperArm + kForearm + kHandLength) + dy;
    float hz = s.z - dz;
    if (raise && side > 0.0f) {
      // "Engage": present the right hand to the sensor the way the title's
      // hand cursor expects it - about 0.6 m in front of the head, slightly
      // below it - with a gentle sideways sweep so the cursor registers as
      // moving. (Right hand: side == +1.)
      const Vector4 head = joints_[kHead];
      const float sweep = 0.15f * std::sin(engage_time_ * 3.0f);
      hx = head.x + 0.30f + sweep;
      hy = head.y - 0.40f;
      hz = head.z - 0.60f;
    }
    // Clamp to the reachable sphere.
    const float reach = kUpperArm + kForearm + kHandLength;
    float vx = hx - s.x, vy = hy - s.y, vz = hz - s.z;
    float len = std::sqrt(vx * vx + vy * vy + vz * vz);
    if (len > reach) {
      vx *= reach / len;
      vy *= reach / len;
      vz *= reach / len;
      len = reach;
    }
    float ux = vx / std::max(len, 1e-3f), uy = vy / std::max(len, 1e-3f),
          uz = vz / std::max(len, 1e-3f);
    // Elbow: partway along, bent slightly outwards/back.
    float bend = std::sqrt(std::max(0.0f, kUpperArm * kUpperArm -
                                              (len * 0.5f) * (len * 0.5f)));
    Vector4 e = {s.x + ux * (len * 0.5f) + side * bend * 0.6f,
                 s.y + uy * (len * 0.5f) - bend * 0.3f,
                 s.z + uz * (len * 0.5f) + bend * 0.7f, 1.0f};
    joints_[elbow] = e;
    joints_[wrist] = {s.x + ux * (len - kHandLength),
                      s.y + uy * (len - kHandLength),
                      s.z + uz * (len - kHandLength), 1.0f};
    joints_[hand] = {s.x + vx, s.y + vy, s.z + vz, 1.0f};
  };
  arm(kShoulderLeft, kElbowLeft, kWristLeft, kHandLeft, -1.0f, left_hand_dx_,
      left_hand_dy_, left_hand_dz_, false);
  arm(kShoulderRight, kElbowRight, kWristRight, kHandRight, +1.0f,
      right_hand_dx_, right_hand_dy_, right_hand_dz_, wave_);
}

void VirtualKinect::WriteSkeletonFrame(uint8_t* frame) {
  std::lock_guard<std::mutex> lock(mutex_);
  const Layout& l = layout_;
  std::memset(frame, 0, l.frame_size);

  auto now = std::chrono::steady_clock::now();
  int64_t timestamp_100ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_)
          .count() /
      100;
  xe::store_and_swap<int64_t>(frame + l.frame_timestamp, timestamp_100ns);
  xe::store_and_swap<uint32_t>(frame + l.frame_number, frame_number_);
  xe::store_and_swap<uint32_t>(frame + l.frame_flags, 0);
  // Floor plane Ax + By + Cz + D = 0 with the normal pointing up; the floor is
  // kSensorHeight below the sensor. This runtime expresses D in millimetres
  // (the title scales it by 0.001).
  StoreVector4(frame + l.frame_floor_clip_plane,
               {0.0f, 1.0f, 0.0f, kSensorHeight * 1000.0f});
  StoreVector4(frame + l.frame_normal_to_gravity, {0.0f, 1.0f, 0.0f, 0.0f});

  // Skeleton 0 is our player, the rest stay "not tracked".
  uint8_t* sk = frame + l.frame_skeletons;
  xe::store_and_swap<uint32_t>(sk + l.skeleton_tracking_state, 2);
  xe::store_and_swap<uint32_t>(sk + l.skeleton_tracking_id, tracking_id_);
  xe::store_and_swap<uint32_t>(sk + l.skeleton_enrollment_index, 0xFFFFFFFF);
  xe::store_and_swap<uint32_t>(sk + l.skeleton_user_index, 0xFFFFFFFF);
  StoreVector4(sk + l.skeleton_position, joints_[kHipCenter]);
  for (uint32_t j = 0; j < kJointCount; ++j) {
    StoreVector4(sk + l.skeleton_joints + j * 16, joints_[j]);
    xe::store_and_swap<uint32_t>(sk + l.skeleton_joint_states + j * 4, 2);
  }
  xe::store_and_swap<uint32_t>(sk + l.skeleton_quality_flags, 0);
}

}  // namespace hooks
}  // namespace kernel
}  // namespace xe
