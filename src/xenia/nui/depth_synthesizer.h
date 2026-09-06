/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_DEPTH_SYNTHESIZER_H_
#define XENIA_NUI_DEPTH_SYNTHESIZER_H_

#include <array>
#include <cstdint>
#include <vector>

#include "xenia/nui/camera_model.h"
#include "xenia/nui/nui_types.h"

namespace xe {
namespace nui {

// Renders a Kinect-style depth image (millimetres) and player-index map from
// skeletons alone: every bone becomes a capsule whose radius follows rough
// adult proportions. Enough for titles that need a silhouette or the depth
// around the hands, and for sources that have no real depth information.
class DepthSynthesizer {
 public:
  struct Options {
    // Background: 0 = unknown (as the sensor reports where nothing is seen),
    // or a synthetic room made of a floor plane and a back wall.
    bool render_room = true;
    float room_depth_m = 3.5f;
    float camera_height_m = 1.0f;
    float tilt_degrees = 0.0f;
    // Bodies that are only position-tracked get an ellipse at their position
    // so player counts stay plausible.
    bool render_position_only = true;
  };

  DepthSynthesizer();

  // |depth_mm| and |player_mask| must hold kDepthWidth * kDepthHeight
  // entries. |bodies| are in (tilted) sensor space; player index i+1 is
  // written for body i.
  void Render(const std::array<Skeleton, kMaxSkeletons>& bodies,
              uint32_t body_count, const Options& options, uint16_t* depth_mm,
              uint8_t* player_mask);

 private:
  void RenderBackground(const Options& options, uint16_t* depth_mm);
  void RenderCapsule(const Vec4& a, const Vec4& b, float radius_m,
                     uint8_t player, uint16_t* depth_mm, uint8_t* player_mask);
  void RenderEllipse(const Vec4& center, float rx_m, float ry_m,
                     uint8_t player, uint16_t* depth_mm, uint8_t* player_mask);

  DepthCameraModel camera_;
  // Cached background for the last (room, height, tilt) combination.
  Options background_options_;
  bool background_valid_ = false;
  std::vector<uint16_t> background_;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_DEPTH_SYNTHESIZER_H_
