/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_NUI_SETTINGS_DIALOG_H_
#define XENIA_APP_NUI_SETTINGS_DIALOG_H_

#include <filesystem>
#include <string>
#include <vector>

#include "xenia/nui/camera_capture.h"
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"

namespace xe {
namespace app {

class EmulatorWindow;

// Kinect (NUI) emulation settings: source selection, webcam configuration,
// tracking options, runtime/model status, playback and recording. Edits are
// written to the nui_* cvars; settings that need the source to be recreated
// take effect on "Apply" (NuiSystem::RestartSource).
class NuiSettingsDialog final : public ui::ImGuiDialog {
 public:
  NuiSettingsDialog(ui::ImGuiDrawer* imgui_drawer,
                    EmulatorWindow& emulator_window);
  ~NuiSettingsDialog() override = default;

 protected:
  void OnDraw(ImGuiIO& io) override;

 private:
  struct FileStatus {
    std::string name;
    std::filesystem::path path;
    bool exists = false;
  };

  void LoadFromCvars();
  void RefreshCameras();
  void RefreshFileStatus();
  // Writes the settings that only take effect when the source is recreated
  // and restarts it.
  void ApplyAndRestart();
  // Writes the settings sources read live.
  void ApplyLiveSettings();

  void DrawSourceSection();
  void DrawCameraSection();
  void DrawGeometrySection();
  void DrawTrackingSection();
  void DrawRuntimeStatusSection();
  void DrawPlaybackSection();

  std::filesystem::path runtime_folder() const;
  std::filesystem::path models_folder() const;
  std::filesystem::path nui_folder() const;

  EmulatorWindow& emulator_window_;

  // Restart-needed settings.
  bool enabled_ = false;
  int source_index_ = 0;
  std::vector<nui::CameraDeviceInfo> cameras_;
  // -1: first camera (empty nui_camera); -2: custom text in camera_text_.
  int camera_index_ = -1;
  char camera_text_[256] = {};
  int capture_size_index_ = 0;
  int capture_fps_index_ = 0;
  int capture_width_ = 640;
  int capture_height_ = 480;
  int capture_fps_ = 30;
  int model_quality_index_ = 0;
  int execution_provider_index_ = 0;
  bool segmentation_ = true;
  char playback_path_[1024] = {};
  bool playback_loop_ = true;
  bool playback_realtime_ = true;

  // Live settings.
  float camera_hfov_ = 70.0f;
  float camera_height_ = 1.0f;
  float camera_pitch_ = 0.0f;
  float user_scale_ = 1.0f;
  bool camera_mirrored_ = false;
  bool auto_bind_user0_ = true;
  bool smoothing_override_ = false;
  float smoothing_ = 0.5f;
  float correction_ = 0.5f;
  float prediction_ = 0.5f;
  float jitter_radius_ = 0.05f;
  float max_deviation_radius_ = 0.04f;
  int tilt_mode_index_ = 0;
  int max_players_ = 2;

  // Recording.
  char record_path_[1024] = {};

  // Status panel.
  std::vector<FileStatus> file_status_;
  bool runtime_complete_ = false;
  bool models_complete_ = false;
  std::string setup_command_;
  double last_status_refresh_ = 0.0;

  bool dirty_restart_ = false;
  std::string apply_message_;
  double apply_message_until_ = 0.0;
};

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_NUI_SETTINGS_DIALOG_H_
