/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_SOURCES_NULL_NUI_SOURCE_H_
#define XENIA_NUI_SOURCES_NULL_NUI_SOURCE_H_

#include <memory>

#include "xenia/nui/nui_source.h"

namespace xe {
namespace nui {

// A sensor that works but never sees anybody.
class NullNuiSource : public NuiSource {
 public:
  std::string_view name() const override { return "none"; }
  bool Start(const DeviceState& initial_state) override;
  void Stop() override {}
  void SetDeviceState(const DeviceState& state) override {}
  std::shared_ptr<const SourceFrame> AcquireLatest(
      uint64_t last_sequence) override;
  void GetStats(SourceStats* out_stats) const override;

 private:
  std::shared_ptr<SourceFrame> frame_;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_SOURCES_NULL_NUI_SOURCE_H_
