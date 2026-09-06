/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/person_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace xe {
namespace nui {

namespace {

// Axis-aligned IoU of two regions (rotation ignored: the crops are squares
// centred on the hips, whose overlap is what identifies a person).
float RegionIou(const PoseRegion& a, const PoseRegion& b) {
  const float ax0 = a.center_x - a.width * 0.5f;
  const float ay0 = a.center_y - a.height * 0.5f;
  const float ax1 = a.center_x + a.width * 0.5f;
  const float ay1 = a.center_y + a.height * 0.5f;
  const float bx0 = b.center_x - b.width * 0.5f;
  const float by0 = b.center_y - b.height * 0.5f;
  const float bx1 = b.center_x + b.width * 0.5f;
  const float by1 = b.center_y + b.height * 0.5f;
  const float ix = std::max(0.0f, std::min(ax1, bx1) - std::max(ax0, bx0));
  const float iy = std::max(0.0f, std::min(ay1, by1) - std::max(ay0, by0));
  const float inter = ix * iy;
  const float area_a = (ax1 - ax0) * (ay1 - ay0);
  const float area_b = (bx1 - bx0) * (by1 - by0);
  const float uni = area_a + area_b - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

float CenterDistance(const PoseRegion& a, const PoseRegion& b) {
  return std::hypot(a.center_x - b.center_x, a.center_y - b.center_y);
}

}  // namespace

PersonTracker::PersonTracker() { tracks_.reserve(kMaxSkeletons); }

void PersonTracker::Reset() {
  // Keys stay monotonic across resets so downstream maps never see a key
  // come back for a different person.
  tracks_.clear();
  expired_.clear();
}

std::vector<PersonTracker::Assignment> PersonTracker::Update(
    const std::vector<PoseResult>& results, uint64_t timestamp_us,
    const Options& options) {
  // Drop tracks that have been unseen for longer than the timeout before
  // matching: a person returning after that gets a new key, like a Kinect
  // tracking id.
  expired_.clear();
  for (size_t i = 0; i < tracks_.size();) {
    const Track& track = tracks_[i];
    if (timestamp_us > track.last_seen_us &&
        timestamp_us - track.last_seen_us > options.timeout_us) {
      expired_.push_back(track.key);
      tracks_.erase(tracks_.begin() + static_cast<std::ptrdiff_t>(i));
    } else {
      ++i;
    }
  }

  std::vector<Assignment> assignments(results.size());
  std::vector<bool> result_matched(results.size(), false);
  std::vector<bool> track_matched(tracks_.size(), false);
  std::vector<size_t> result_track(results.size(), 0);

  // Pass 1: greedy best-IoU matching.
  for (;;) {
    float best_iou = options.min_iou;
    size_t best_result = 0;
    size_t best_track = 0;
    bool found = false;
    for (size_t r = 0; r < results.size(); ++r) {
      if (result_matched[r]) {
        continue;
      }
      for (size_t t = 0; t < tracks_.size(); ++t) {
        if (track_matched[t]) {
          continue;
        }
        const float iou = RegionIou(results[r].region, tracks_[t].region);
        if (iou >= best_iou && (!found || iou > best_iou)) {
          best_iou = iou;
          best_result = r;
          best_track = t;
          found = true;
        }
      }
    }
    if (!found) {
      break;
    }
    result_matched[best_result] = true;
    track_matched[best_track] = true;
    result_track[best_result] = best_track;
  }

  // Pass 2: nearest centres for what is left.
  for (;;) {
    float best_distance = options.max_center_distance;
    size_t best_result = 0;
    size_t best_track = 0;
    bool found = false;
    for (size_t r = 0; r < results.size(); ++r) {
      if (result_matched[r]) {
        continue;
      }
      for (size_t t = 0; t < tracks_.size(); ++t) {
        if (track_matched[t]) {
          continue;
        }
        const float distance =
            CenterDistance(results[r].region, tracks_[t].region);
        if (distance <= best_distance && (!found || distance < best_distance)) {
          best_distance = distance;
          best_result = r;
          best_track = t;
          found = true;
        }
      }
    }
    if (!found) {
      break;
    }
    result_matched[best_result] = true;
    track_matched[best_track] = true;
    result_track[best_result] = best_track;
  }

  // Apply matches and create tracks for the rest.
  for (size_t r = 0; r < results.size(); ++r) {
    if (result_matched[r]) {
      Track& track = tracks_[result_track[r]];
      track.region = results[r].region;
      track.last_seen_us = timestamp_us;
      assignments[r].person_key = track.key;
      assignments[r].is_new = false;
    } else {
      Track track;
      track.key = next_key_++;
      if (next_key_ == 0) {
        next_key_ = 1;  // never hand out the invalid key
      }
      track.region = results[r].region;
      track.last_seen_us = timestamp_us;
      tracks_.push_back(track);
      assignments[r].person_key = track.key;
      assignments[r].is_new = true;
    }
  }
  return assignments;
}

}  // namespace nui
}  // namespace xe
