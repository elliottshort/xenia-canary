/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/nui_source.h"

#include "xenia/base/logging.h"
#include "xenia/nui/sources/null_nui_source.h"
#include "xenia/nui/sources/recorded_nui_source.h"
#include "xenia/nui/sources/virtual_nui_source.h"
#include "xenia/nui/sources/webcam_nui_source.h"

namespace xe {
namespace nui {

// The input system is handed to sources that need it through this hook so
// that the factory stays free of emulator dependencies.
static hid::InputSystem* g_input_system = nullptr;

void SetNuiSourceInputSystem(hid::InputSystem* input_system) {
  g_input_system = input_system;
}

std::unique_ptr<NuiSource> CreateNuiSource(std::string_view name) {
  if (name == "virtual") {
    return std::make_unique<VirtualNuiSource>(g_input_system);
  }
  if (name == "none") {
    return std::make_unique<NullNuiSource>();
  }
  if (name == "webcam") {
    return std::make_unique<WebcamNuiSource>();
  }
  if (name == "playback") {
    return std::make_unique<RecordedNuiSource>();
  }
  return nullptr;
}

}  // namespace nui
}  // namespace xe
