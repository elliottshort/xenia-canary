/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_SKELETON_SYNTHESIZER_H_
#define XENIA_NUI_SKELETON_SYNTHESIZER_H_

#include <array>
#include <cstdint>
#include <map>

#include "xenia/nui/nui_types.h"
#include "xenia/nui/pose_estimator.h"

namespace xe {
namespace nui {

// Turns a BlazePose result into a Kinect skeleton in sensor space.
//
// Absolute distance is not observable from a single RGB camera, so the root
// depth is estimated from the pinhole model: the metric size of body
// segments (from the world landmarks) compared with their size in pixels
// gives Z for each segment; a visibility-weighted median is smoothed over
// time. Joints then get Z_root + world_z and X/Y from the pixel position.
class SkeletonSynthesizer {
 public:
  struct Options {
    uint32_t image_width = 640;
    uint32_t image_height = 480;
    // Horizontal field of view of the camera in degrees.
    float hfov_degrees = 70.0f;
    // The image the model saw was a mirror view (user's right hand on the
    // right side), which is what the Kinect convention expects. BlazePose
    // labels follow the visual side of the image, so in a mirror view its
    // "left" landmarks are the user's right and are mapped to the Kinect
    // right-hand joints; on a plain camera view they are anatomical.
    bool mirrored = true;
    // Time since the previous frame of the same person, for the depth and
    // visibility filters; clamped to [1/120, 1/10] s.
    float frame_dt_seconds = 1.0f / 30.0f;
    // Physical pitch of the camera (positive looking up) and the sensor
    // tilt the title requested; both rotate the result about X.
    float camera_pitch_degrees = 0.0f;
    float tilt_degrees = 0.0f;
    // Multiplies the estimated distance (per-user calibration).
    float user_scale = 1.0f;
    // Visibility thresholds for joint tracking states.
    float tracked_threshold = 0.65f;
    float inferred_threshold = 0.3f;
  };

  SkeletonSynthesizer();

  // |person_key| selects the temporal smoothing state; returns false when
  // the pose has too few usable landmarks (the skeleton is then reported as
  // position-only with the region centre as position).
  bool Synthesize(const PoseResult& pose, uint32_t person_key,
                  const Options& options, Skeleton* out_skeleton);
  void Forget(uint32_t person_key);
  void Reset();

  // Estimated distance of the person's hip centre from the camera (metres)
  // after smoothing; 0 if unknown.
  float last_root_depth(uint32_t person_key) const;

 private:
  struct PersonState {
    float root_z = 0.0f;
    float root_z_velocity = 0.0f;
    bool valid = false;
    std::array<float, kJointCount> visibility_ema{};
    std::array<JointState, kJointCount> last_state{};
  };
  std::map<uint32_t, PersonState> persons_;
};

// Converts a normalized image point plus depth to sensor space using the
// pinhole model of |options|. Shared with the depth/colour remapping.
Vec4 ImagePointToSensorSpace(float nx, float ny, float depth_m,
                             const SkeletonSynthesizer::Options& options);

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_SKELETON_SYNTHESIZER_H_
