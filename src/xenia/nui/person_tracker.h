/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_PERSON_TRACKER_H_
#define XENIA_NUI_PERSON_TRACKER_H_

#include <cstdint>
#include <vector>

#include "xenia/nui/pose_estimator.h"

namespace xe {
namespace nui {

// Assigns stable, never-reused person keys to pose results across frames
// by matching regions (IoU, then centre distance). A person who disappears
// keeps their key for |timeout_us|; after that a reappearance gets a new key,
// mirroring how the Kinect issues a new tracking id.
class PersonTracker {
 public:
  struct Options {
    float min_iou = 0.3f;
    float max_center_distance = 0.15f;  // normalized image units
    uint64_t timeout_us = 500000;
  };

  struct Assignment {
    uint32_t person_key = 0;  // never 0 for a matched/new person
    bool is_new = false;
  };

  PersonTracker();

  // |results| are this frame's poses; returns one assignment per result.
  std::vector<Assignment> Update(const std::vector<PoseResult>& results,
                                 uint64_t timestamp_us,
                                 const Options& options);
  // Keys that timed out during the last Update (for cleaning up per-person
  // state elsewhere).
  const std::vector<uint32_t>& expired_keys() const { return expired_; }
  void Reset();

 private:
  struct Track {
    uint32_t key = 0;
    PoseRegion region;
    uint64_t last_seen_us = 0;
  };
  std::vector<Track> tracks_;
  std::vector<uint32_t> expired_;
  uint32_t next_key_ = 1;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_PERSON_TRACKER_H_
