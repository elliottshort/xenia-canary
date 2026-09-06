/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_TITLE_HOOKS_MILO_NUI_HOOKS_H_
#define XENIA_KERNEL_TITLE_HOOKS_MILO_NUI_HOOKS_H_

#include <cstdint>

namespace xe {
namespace kernel {

class KernelState;
class UserModule;

namespace hooks {
namespace milo {

// "Project Milo" (Lionhead, Kinect technology demo, XDK 11427 devkit builds).
constexpr uint32_t kTitleId = 0x4D5308D8;

// Installs host replacements for the Kinect (NUI) runtime entry points the
// title uses, providing a "virtual Kinect": skeleton frames synthesised from
// controller/keyboard input so the game can be played without the sensor.
void Install(KernelState* kernel_state, UserModule* module);

}  // namespace milo
}  // namespace hooks
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_TITLE_HOOKS_MILO_NUI_HOOKS_H_
