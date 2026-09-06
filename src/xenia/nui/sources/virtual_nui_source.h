/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_SOURCES_VIRTUAL_NUI_SOURCE_H_
#define XENIA_NUI_SOURCES_VIRTUAL_NUI_SOURCE_H_

#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include "xenia/nui/depth_synthesizer.h"
#include "xenia/nui/nui_source.h"

namespace xe {
namespace hid {
class InputSystem;
}  // namespace hid

namespace nui {

// A stand-in for a person in front of the sensor, driven by the emulated
// gamepad of user slot 0 (which the keyboard driver also feeds, so keyboard
// bindings work too). Produces skeletons plus a synthesized depth/player
// image so titles that use silhouettes or hand cursors work without a
// camera. The pose logic follows the Milo virtual Kinect written for the
// title-specific hooks.
//
// Controls:
//   left stick     left hand (sideways / up-down)
//   right stick    right hand
//   left trigger   push the left hand towards the sensor
//   right trigger  push the right hand towards the sensor
//   d-pad          walk: left/right strafe, up = towards the sensor, down =
//                  away
//   A              "engage": right hand held out in front at chest height
//                  with a gentle sweep (Kinect Guide / hand cursor wave)
//   B              both hands at the sides
//   X              jump
//   Y              T-pose
//   left shoulder  lean left, right shoulder lean right
//   back           crouch while held
//   start          toggle a second (idle) player standing to the right
class VirtualNuiSource : public NuiSource {
 public:
  explicit VirtualNuiSource(hid::InputSystem* input_system);

  std::string_view name() const override { return "virtual"; }
  bool Start(const DeviceState& initial_state) override;
  void Stop() override;
  void SetDeviceState(const DeviceState& state) override;
  std::shared_ptr<const SourceFrame> AcquireLatest(
      uint64_t last_sequence) override;
  void GetStats(SourceStats* out_stats) const override;

 private:
  struct Pose {
    float body_x = 0.0f;
    float body_z = 2.2f;
    float body_y_offset = 0.0f;  // jump / crouch
    float lean = 0.0f;           // -1 left .. +1 right
    float left_dx = 0.0f, left_dy = 0.0f, left_dz = 0.0f;
    float right_dx = 0.0f, right_dy = 0.0f, right_dz = 0.0f;
    bool engage = false;
    bool t_pose = false;
    float engage_time = 0.0f;
  };

  void PollInput(float dt);
  void ComputeBody(const Pose& pose, uint32_t person_key, Skeleton* out) const;
  std::shared_ptr<SourceFrame> AllocateFrame();

  hid::InputSystem* input_system_ = nullptr;
  std::mutex mutex_;
  DeviceState device_state_;
  bool started_ = false;
  Pose pose_;
  bool second_player_ = false;
  bool start_was_down_ = false;
  bool x_was_down_ = false;
  float jump_time_ = -1.0f;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_update_;
  uint64_t sequence_ = 0;
  uint64_t frames_produced_ = 0;
  std::vector<std::shared_ptr<SourceFrame>> pool_;
  DepthSynthesizer depth_synthesizer_;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_SOURCES_VIRTUAL_NUI_SOURCE_H_
