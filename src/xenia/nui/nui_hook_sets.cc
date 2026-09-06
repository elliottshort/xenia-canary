/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/nui_hook_sets.h"

namespace xe {
namespace nui {

const char* NuiHookSetName(NuiHookSet set) {
  switch (set) {
    case NuiHookSet::kLifecycle:
      return "lifecycle";
    case NuiHookSet::kSkeleton:
      return "skeleton";
    case NuiHookSet::kImage:
      return "image stream";
    case NuiHookSet::kCamera:
      return "camera tilt";
    case NuiHookSet::kTransform:
      return "coordinate transform";
    case NuiHookSet::kNone:
    default:
      return "standalone";
  }
}

bool IsStatefulNuiHookSet(NuiHookSet set) {
  switch (set) {
    case NuiHookSet::kLifecycle:
    case NuiHookSet::kSkeleton:
    case NuiHookSet::kImage:
      return true;
    default:
      return false;
  }
}

bool NuiHookPlan::IsSetCeded(NuiHookSet set) const {
  for (const auto& decision : decisions) {
    if (decision.set == set && decision.status == NuiHookSetStatus::kCeded) {
      return true;
    }
  }
  return false;
}

bool NuiHookPlan::IsSetHooked(NuiHookSet set) const {
  for (const auto& decision : decisions) {
    if (decision.set == set && decision.status == NuiHookSetStatus::kHook) {
      return true;
    }
  }
  return false;
}

NuiHookPlan PlanNuiHooks(const std::vector<NuiHookCandidate>& candidates) {
  NuiHookPlan plan;
  plan.hook.assign(candidates.size(), false);

  // Group the candidates. Standalone (kNone) entries each get their own
  // group so that one missing helper never takes anything else down.
  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& candidate = candidates[i];
    size_t index = plan.decisions.size();
    if (candidate.set != NuiHookSet::kNone) {
      for (size_t d = 0; d < plan.decisions.size(); ++d) {
        if (plan.decisions[d].set == candidate.set) {
          index = d;
          break;
        }
      }
    }
    if (index == plan.decisions.size()) {
      NuiHookSetDecision decision;
      decision.set = candidate.set;
      decision.label = candidate.set == NuiHookSet::kNone
                           ? candidate.name
                           : NuiHookSetName(candidate.set);
      plan.decisions.push_back(std::move(decision));
    }
    plan.decisions[index].members.push_back(i);
  }

  for (auto& decision : plan.decisions) {
    // Somebody else owning any function of the group wins over everything
    // else: they see state we do not, so we hook none of it.
    for (size_t i : decision.members) {
      if (candidates[i].already_hooked) {
        decision.status = NuiHookSetStatus::kCeded;
        decision.blocking_function = candidates[i].name;
        decision.blocking_address = candidates[i].address;
        break;
      }
    }
    if (decision.status == NuiHookSetStatus::kHook) {
      for (size_t i : decision.members) {
        if (!candidates[i].resolved && candidates[i].required) {
          decision.status = NuiHookSetStatus::kMissing;
          decision.blocking_function = candidates[i].name;
          break;
        }
      }
    }
    switch (decision.status) {
      case NuiHookSetStatus::kHook:
        for (size_t i : decision.members) {
          if (!candidates[i].resolved) {
            continue;  // Optional and not found; nothing to replace.
          }
          plan.hook[i] = true;
          ++plan.hooked_count;
          if (candidates[i].required) {
            ++plan.required_hooked_count;
          }
        }
        if (IsStatefulNuiHookSet(decision.set)) {
          ++plan.hooked_stateful_set_count;
        }
        break;
      case NuiHookSetStatus::kCeded:
        ++plan.ceded_set_count;
        if (IsStatefulNuiHookSet(decision.set)) {
          ++plan.ceded_stateful_set_count;
        }
        break;
      case NuiHookSetStatus::kMissing:
        ++plan.missing_set_count;
        break;
    }
  }
  return plan;
}

}  // namespace nui
}  // namespace xe
