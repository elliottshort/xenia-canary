/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_NUI_PREVIEW_DIALOG_H_
#define XENIA_APP_NUI_PREVIEW_DIALOG_H_

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "xenia/nui/nui_system.h"
#include "xenia/nui/sources/webcam_nui_source.h"
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"
#include "xenia/ui/immediate_drawer.h"

namespace xe {
namespace app {

class EmulatorWindow;

// Live view of what the emulated Kinect sees: the webcam image with the
// 2D pose landmarks (webcam source) or the skeleton frame projected onto a
// depth-image canvas with the player mask (other sources), plus statistics.
class NuiPreviewDialog final : public ui::ImGuiDialog {
 public:
  NuiPreviewDialog(ui::ImGuiDrawer* imgui_drawer,
                   EmulatorWindow& emulator_window);
  ~NuiPreviewDialog() override = default;

 protected:
  void OnDraw(ImGuiIO& io) override;

 private:
  struct Canvas {
    ImVec2 origin;
    ImVec2 size;
    ImVec2 ToScreen(float u, float v) const {
      return ImVec2(origin.x + u * size.x, origin.y + v * size.y);
    }
  };

  void DrawWebcamPreview(nui::NuiSystem* nui_system,
                         nui::WebcamNuiSource* webcam,
                         const nui::SkeletonFrame* skeleton,
                         const nui::SourceFrame* source);
  void DrawSkeletonCanvas(const nui::SkeletonFrame* skeleton,
                          const nui::SourceFrame* source);
  void DrawStatus(nui::NuiSystem* nui_system,
                  const nui::SkeletonFrame* skeleton);

  // Recreates |texture_| from a camera frame, resampled to the preview
  // width, when the frame is new and the update interval has elapsed.
  void UpdateWebcamTexture(const nui::CameraFrame& frame);
  // Recreates |texture_| from the player mask of a source frame.
  void UpdateMaskTexture(const nui::SourceFrame& source);
  void ReplaceTexture(uint32_t width, uint32_t height);

  EmulatorWindow& emulator_window_;

  std::unique_ptr<ui::ImmediateTexture> texture_;
  uint32_t texture_width_ = 0;
  uint32_t texture_height_ = 0;
  // Sequence of the camera frame / source frame the texture was built from.
  uint64_t texture_sequence_ = 0;
  bool texture_is_mask_ = false;
  std::chrono::steady_clock::time_point last_texture_update_{};
  std::vector<uint8_t> scratch_rgba_;

  nui::WebcamNuiSource::Preview preview_;
  bool preview_valid_ = false;
};

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_NUI_PREVIEW_DIALOG_H_
