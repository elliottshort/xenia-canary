/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_TITLE_HOOKS_H_
#define XENIA_KERNEL_TITLE_HOOKS_H_

namespace xe {
namespace kernel {

class KernelState;
class UserModule;

// Title-specific host replacements for guest functions.
//
// Some titles statically link runtime libraries that talk to hardware xenia
// cannot emulate at the kernel boundary in a useful way (the Kinect/NUI
// runtime is the main example: the skeletal tracker runs inside the title and
// only raw sensor transfers cross into the kernel). For those, the cleanest
// place to substitute emulation is the library's public entry points inside
// the title itself. Hooks are keyed on the title ID and validated against the
// instruction bytes of the target build, and are installed once the main
// module's imports are resolved and before any of its code runs.
void ApplyTitleHooks(KernelState* kernel_state, UserModule* module);

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_TITLE_HOOKS_H_
