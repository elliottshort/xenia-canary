/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_NUI_SOURCE_H_
#define XENIA_NUI_NUI_SOURCE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "xenia/nui/nui_types.h"

namespace xe {
namespace hid {
class InputSystem;
}  // namespace hid

namespace nui {

// What the device model asks a source to honour. Copied to the source
// whenever it changes; sources read it at the start of each frame.
struct DeviceState {
  // NUI_INITIALIZE_FLAG_* the title initialized with.
  uint32_t init_flags = 0;
  bool skeleton_tracking = false;
  bool seated_mode = false;
  bool near_mode = false;
  // Which image streams are open, so sources can skip work.
  bool want_player_mask = false;
  bool want_depth = false;
  bool want_color = false;
  // Effective sensor tilt in degrees (title request, slewed) and the
  // physical pitch of the capture device; both positive looking up.
  float tilt_degrees = 0.0f;
  float camera_pitch_degrees = 0.0f;
  float camera_height_m = 1.0f;
  // Person keys the title asked to be fully tracked
  // (NUI_SKELETON_TRACKING_FLAG_TITLE_SETS_TRACKED_SKELETONS); 0 = none.
  uint32_t preferred_person_keys[kMaxTrackedSkeletons] = {0, 0};
  uint32_t max_tracked_players = kMaxTrackedSkeletons;
};

struct SourceStats {
  double capture_fps = 0.0;
  double inference_ms = 0.0;
  uint64_t frames_produced = 0;
  uint64_t frames_dropped = 0;
  std::string status;
};

// A producer of synthesized sensor observations. Implementations own their
// own threads (or none: the virtual source is advanced when polled).
//
// AcquireLatest is the only hot-path call: the device model's pacer thread
// calls it once per 30 Hz tick and must never block for long.
class NuiSource {
 public:
  virtual ~NuiSource() = default;

  virtual std::string_view name() const = 0;

  // Opens the camera / loads models / starts threads. Returns false if the
  // source cannot work at all (the device model then behaves as "no sensor").
  virtual bool Start(const DeviceState& initial_state) = 0;
  virtual void Stop() = 0;

  virtual void SetDeviceState(const DeviceState& state) = 0;

  // Returns the newest frame if its sequence is greater than |last_sequence|,
  // otherwise nullptr. The returned frame stays valid until the source is
  // stopped; sources recycle frames only after they have handed out a newer
  // one, so the caller must not keep more than the latest pointer.
  virtual std::shared_ptr<const SourceFrame> AcquireLatest(
      uint64_t last_sequence) = 0;

  virtual void GetStats(SourceStats* out_stats) const = 0;
};

// Creates the source selected by the nui_source cvar (or |name| if given).
// Returns nullptr for an unknown name.
std::unique_ptr<NuiSource> CreateNuiSource(std::string_view name);

// Gives sources that synthesize input from the emulated gamepad access to
// the input system. Call once before creating sources.
void SetNuiSourceInputSystem(hid::InputSystem* input_system);

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_NUI_SOURCE_H_
