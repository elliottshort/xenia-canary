/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_NUI_NUI_HLE_H_
#define XENIA_KERNEL_NUI_NUI_HLE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "xenia/nui/nui_hook_sets.h"

namespace xe {
namespace kernel {

class KernelState;
class UserModule;

namespace nui {

// Per NUI-library-version behaviour differences the handlers must reproduce.
struct NuiHleQuirks {
  // NUI_SKELETON_FRAME.liTimeStamp unit: 100 ns ticks (XDK 11427) or
  // milliseconds (Kinect SDK v1).
  bool timestamp_100ns = false;
  // vFloorClipPlane.w in millimetres (XDK 11427) instead of metres.
  bool floor_plane_millimetres = false;
  // NuiInitialize flag bits the library accepts, and which of them mean
  // "skeletal tracking" (the XDK 11427 beta used different values).
  uint32_t init_flags_mask = 0x1000006B;  // Kinect SDK v1 set
  uint32_t init_skeleton_flags = 0x00000008;
  // HRESULTs whose values moved between library generations.
  uint32_t hr_wait_timeout = 0x83010001;            // E_NUI_FRAME_NO_DATA
  uint32_t hr_stream_not_enabled = 0x83010003;
  uint32_t hr_image_stream_in_use = 0x83010004;
  uint32_t hr_frame_limit_exceeded = 0x83010005;
  uint32_t hr_feature_not_initialized = 0x83010006;
};

// The ownership groups of xe::nui, spelled without the namespace dance in
// the signature tables (xe::kernel::nui would otherwise shadow xe::nui).
using HookSet = xe::nui::NuiHookSet;

// A hookable function of the NUI runtime inside a title.
struct NuiFunctionSignature {
  const char* name;
  // "hle": replaced by our implementation; "tripwire": replaced by a stub
  // that logs and fails (unsupported feature); "native": left alone.
  const char* mode;
  // The group of functions this one shares state with. Groups are hooked
  // all-or-nothing: if a title-specific hook layer already owns one function
  // of a group, we cede the whole group to it (see nui_hle.cc and
  // docs/nui/architecture.md, "Composing with title-specific hooks").
  // HookSet::kNone means the function stands alone.
  HookSet set;
  // Fixed address for builds we know exactly (0 = search by pattern).
  uint32_t fixed_address;
  // Instruction words from the function entry, in natural (big-endian
  // decoded) form, with per-word masks: (memory & mask) == (word & mask).
  std::vector<uint32_t> words;
  std::vector<uint32_t> masks;
};

struct NuiLibrarySignatures {
  // Static library name and version as listed in the XEX header, e.g.
  // "NUI" and "2.0.11775.6". An empty version matches any version of the
  // library when |module_name| matches instead.
  const char* library;
  const char* version;
  // Optional module (executable) name, for builds identified by file.
  const char* module_name;
  NuiHleQuirks quirks;
  std::vector<NuiFunctionSignature> functions;
};

const std::vector<NuiLibrarySignatures>& GetBuiltinNuiSignatures();

// Installs the NUI runtime replacements into |module| if it links a
// supported NUI library version. Called once per loaded executable before
// any of its code runs.
void AttachNuiHle(KernelState* kernel_state, UserModule* module);

// Currently active quirks (valid after AttachNuiHle found a match).
const NuiHleQuirks& GetActiveNuiHleQuirks();

}  // namespace nui
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_NUI_NUI_HLE_H_
