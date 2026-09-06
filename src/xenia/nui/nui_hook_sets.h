/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_NUI_HOOK_SETS_H_
#define XENIA_NUI_NUI_HOOK_SETS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xe {
namespace nui {

// Groups of NUI runtime functions that must be replaced all-or-nothing,
// because the functions of a group share state that a foreign implementation
// (a title-specific hook layer) cannot see: stream handles, the skeleton
// frame cursor, the tilt motor model. Mixing our implementation of one
// function of a group with somebody else's implementation of another breaks
// both, so whoever owns one function of a group owns all of them.
//
// This lives in xenia-nui rather than in the kernel so that the (pure)
// ownership decision can be unit tested without the kernel.
enum class NuiHookSet {
  // No shared state: the function stands alone and is decided by itself
  // (pure helpers, tripwire stubs).
  kNone = 0,
  // NuiInitialize / NuiShutdown: the device model's initialized flag.
  kLifecycle,
  // Skeleton tracking: enable/disable flags, the frame cursor, the tracked
  // skeleton selection and the smoother state.
  kSkeleton,
  // Image streams: the stream handles handed out by NuiImageStreamOpen and
  // the guest memory backing their frames.
  kImage,
  // The simulated tilt motor and its gravity vector.
  kCamera,
  // Pure coordinate transforms (no state, but they must agree with whoever
  // produced the image they are applied to).
  kTransform,
};

// Short human-readable name used in logs, e.g. "image stream".
const char* NuiHookSetName(NuiHookSet set);

// One resolved (or unresolved) NUI runtime function, as the input of the
// ownership decision.
struct NuiHookCandidate {
  std::string name;
  NuiHookSet set = NuiHookSet::kNone;
  // Where the function was found (0 when it was not); only used for logging.
  uint32_t address = 0;
  // The function was located in the module image.
  bool resolved = false;
  // Somebody else already replaced this function (its entry is the syscall
  // trampoline). Implies |resolved|.
  bool already_hooked = false;
  // A function we must have to serve the group; optional entries (tripwires)
  // that are not found do not spoil their group.
  bool required = true;
};

enum class NuiHookSetStatus {
  // Every function of the group is ours: hook them all.
  kHook,
  // At least one function of the group is already hooked by somebody else:
  // hook none of them and let the other owner drive.
  kCeded,
  // Nobody owns the group but a required function was not found: hook none
  // of them (an incompletely replaced group is worse than none).
  kMissing,
};

struct NuiHookSetDecision {
  NuiHookSet set = NuiHookSet::kNone;
  // NuiHookSetName(set), or the function name for standalone (kNone) entries.
  std::string label;
  NuiHookSetStatus status = NuiHookSetStatus::kHook;
  // The function that caused kCeded / kMissing, and where it was found.
  std::string blocking_function;
  uint32_t blocking_address = 0;
  // Indices into the candidate list this decision covers.
  std::vector<size_t> members;
};

struct NuiHookPlan {
  // One entry per group, in the order the groups first appear in the input.
  std::vector<NuiHookSetDecision> decisions;
  // Parallel to the candidate list: whether to install our hook.
  std::vector<bool> hook;
  uint32_t hooked_count = 0;
  // Hooked functions that were required, i.e. real emulation rather than
  // tripwire stubs. Zero means we serve nothing and the title should be told
  // there is no sensor.
  uint32_t required_hooked_count = 0;
  uint32_t ceded_set_count = 0;
  uint32_t missing_set_count = 0;

  // Whether the candidate at |index| is one of ours to install.
  bool ShouldHook(size_t index) const {
    return index < hook.size() && hook[index];
  }
  bool IsSetCeded(NuiHookSet set) const;
  bool IsSetHooked(NuiHookSet set) const;
};

// Decides, per group, whether we hook it, cede it to whoever hooked part of
// it first, or drop it because a required function is missing. Pure: no
// module, no memory, no logging.
NuiHookPlan PlanNuiHooks(const std::vector<NuiHookCandidate>& candidates);

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_NUI_HOOK_SETS_H_
