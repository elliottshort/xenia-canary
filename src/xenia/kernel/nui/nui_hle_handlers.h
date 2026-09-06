/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_NUI_NUI_HLE_HANDLERS_H_
#define XENIA_KERNEL_NUI_NUI_HLE_HANDLERS_H_

#include <string_view>

#include "xenia/cpu/function.h"

namespace xe {
namespace kernel {

class KernelState;

namespace nui {

// Host implementation of a public NUI API function, by name
// (e.g. "NuiSkeletonGetNextFrame"). Returns nullptr for unknown names.
cpu::GuestFunction::ExternHandler LookupNuiHleHandler(std::string_view name);

// A stub for NUI API functions we know about but do not emulate: logs once
// and fails with E_NOTIMPL. Returns nullptr if |name| has no tripwire slot.
cpu::GuestFunction::ExternHandler LookupNuiTripwireHandler(
    std::string_view name);

// Drops every guest resource the handlers retained (events, stream
// tables). Called when hooks are (re)installed for a new title.
void ResetNuiHleState(KernelState* kernel_state);

}  // namespace nui
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_NUI_NUI_HLE_HANDLERS_H_
