/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/sources/null_nui_source.h"

namespace xe {
namespace nui {

bool NullNuiSource::Start(const DeviceState& initial_state) {
  frame_ = std::make_shared<SourceFrame>();
  frame_->sequence = 1;
  return true;
}

std::shared_ptr<const SourceFrame> NullNuiSource::AcquireLatest(
    uint64_t last_sequence) {
  if (frame_ && frame_->sequence > last_sequence) {
    return frame_;
  }
  return nullptr;
}

void NullNuiSource::GetStats(SourceStats* out_stats) const {
  *out_stats = SourceStats();
  out_stats->status = "no sensor input";
}

}  // namespace nui
}  // namespace xe
