/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/title_hooks.h"

#include "xenia/base/logging.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/title_hooks/milo_nui_hooks.h"
#include "xenia/kernel/user_module.h"

namespace xe {
namespace kernel {

void ApplyTitleHooks(KernelState* kernel_state, UserModule* module) {
  if (!module || !module->xex_module() || !module->is_executable()) {
    return;
  }

  switch (module->title_id()) {
    case hooks::milo::kTitleId:
      hooks::milo::Install(kernel_state, module);
      break;
    default:
      break;
  }
}

}  // namespace kernel
}  // namespace xe
