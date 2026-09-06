/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_TITLE_HOOKS_VIRTUAL_KINECT_H_
#define XENIA_KERNEL_TITLE_HOOKS_VIRTUAL_KINECT_H_

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace xe {
namespace hid {
class InputSystem;
}  // namespace hid
namespace kernel {
namespace hooks {

// A stand-in for a Kinect sensor's skeletal tracker: one tracked player,
// standing in front of the sensor, whose hands and body are moved with the
// emulated gamepad (which keyboard/controller drivers already feed).
//
// Controls (user slot 0):
//   left stick    left hand   (x: sideways, y: up/down)
//   right stick   right hand
//   left trigger  push left hand towards the sensor
//   right trigger push right hand towards the sensor
//   d-pad         walk: left/right strafe, up/down towards/away from sensor
//   A             "engage": hold the right hand out towards the sensor at
//                 chest height with a gentle sweep (what the title's hand
//                 cursor wants during calibration)
//   B             hold both hands at the sides (neutral)
//
// Coordinates follow the NUI skeleton space: metres, origin at the sensor,
// +x to the player's right as seen by the sensor, +y up, +z away from the
// sensor towards the player.
class VirtualKinect {
 public:
  // NUI_SKELETON_POSITION_INDEX (Kinect SDK v1 order).
  enum Joint : uint32_t {
    kHipCenter = 0,
    kSpine,
    kShoulderCenter,
    kHead,
    kShoulderLeft,
    kElbowLeft,
    kWristLeft,
    kHandLeft,
    kShoulderRight,
    kElbowRight,
    kWristRight,
    kHandRight,
    kHipLeft,
    kKneeLeft,
    kAnkleLeft,
    kFootLeft,
    kHipRight,
    kKneeRight,
    kAnkleRight,
    kFootRight,
    kJointCount
  };

  struct Vector4 {
    float x, y, z, w;
  };

  // Layout of NUI_SKELETON_FRAME / NUI_SKELETON_DATA as consumed by the
  // target runtime (XDK 11427 beta: 0x1C0-byte skeleton records).
  struct Layout {
    uint32_t frame_size = 0xAB0;
    uint32_t frame_timestamp = 0x00;         // s64 100 ns units
    uint32_t frame_number = 0x08;            // u32
    uint32_t frame_flags = 0x0C;             // u32
    uint32_t frame_floor_clip_plane = 0x10;  // Vector4
    uint32_t frame_normal_to_gravity = 0x20; // Vector4
    uint32_t frame_skeletons = 0x30;         // NUI_SKELETON_DATA[6]
    uint32_t skeleton_count = 6;
    uint32_t skeleton_size = 0x1C0;
    uint32_t skeleton_tracking_state = 0x00;  // u32: 0 none, 1 position, 2 tracked
    uint32_t skeleton_tracking_id = 0x04;     // u32
    uint32_t skeleton_enrollment_index = 0x08;
    uint32_t skeleton_user_index = 0x0C;
    uint32_t skeleton_position = 0x10;        // Vector4
    uint32_t skeleton_joints = 0x20;          // Vector4[20]
    uint32_t skeleton_joint_states = 0x160;   // u32[20]: 0 none, 1 inferred, 2 tracked
    uint32_t skeleton_quality_flags = 0x1B0;  // u32
  };

  VirtualKinect();

  Layout& layout() { return layout_; }

  // Polls the emulated gamepad and advances the pose. Safe to call from any
  // thread; cheap enough to call per frame request.
  void Update(hid::InputSystem* input_system);

  // Sensor frame rate (30 Hz). Returns true if a new frame is due since the
  // last call that returned true.
  bool NewFrameDue();

  // Writes a complete big-endian NUI_SKELETON_FRAME (layout().frame_size
  // bytes) describing the current pose into |frame|.
  void WriteSkeletonFrame(uint8_t* frame);

  const std::array<Vector4, kJointCount>& joints() const { return joints_; }
  uint32_t frame_number() const { return frame_number_; }

 private:
  void ComputePose();

  std::mutex mutex_;
  Layout layout_;

  // Body root position (hip centre) in sensor space.
  float body_x_ = 0.0f;
  float body_z_ = 2.2f;
  // Hand offsets relative to the neutral "arms at the sides" pose.
  float left_hand_dx_ = 0.0f, left_hand_dy_ = 0.0f, left_hand_dz_ = 0.0f;
  float right_hand_dx_ = 0.0f, right_hand_dy_ = 0.0f, right_hand_dz_ = 0.0f;
  bool wave_ = false;
  float engage_time_ = 0.0f;

  std::array<Vector4, kJointCount> joints_{};
  uint32_t frame_number_ = 0;
  uint32_t tracking_id_ = 1;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_update_;
  std::chrono::steady_clock::time_point last_frame_;
};

}  // namespace hooks
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_TITLE_HOOKS_VIRTUAL_KINECT_H_
