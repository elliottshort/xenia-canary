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

// Drops every guest resource the handlers retained (events, stream tables,
// the frame listener). Called when the title is terminated and when hooks
// are (re)installed for a new title.
//
// |free_guest_memory| releases the pooled guest allocations that back the
// image streams as well; only pass true while the title that owns them is
// still the one whose memory |kernel_state| addresses, i.e. from the
// termination path.
void ResetNuiHleState(KernelState* kernel_state,
                      bool free_guest_memory = false);

// Declares that a title-specific hook layer owns a stateful part of the NUI
// runtime (see nui_hle.cc). Our NuiShutdown then drops only our own guest
// resources: tearing the device model down under an owner that never asked
// for it costs a camera reopen and a model reload on its next frame.
void SetNuiHleExternalOwner(bool external_owner);

}  // namespace nui
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_NUI_NUI_HLE_HANDLERS_H_
