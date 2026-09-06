/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/nui_settings_dialog.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "third_party/fmt/include/fmt/format.h"
#include "third_party/imgui/imgui.h"
#include "xenia/app/emulator_window.h"
#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/string.h"
#include "xenia/base/string_util.h"
#include "xenia/base/system.h"
#include "xenia/config.h"
#include "xenia/emulator.h"
#include "xenia/nui/nui_flags.h"
#include "xenia/nui/nui_system.h"

namespace xe {
namespace app {

namespace {

constexpr const char* kSourceNames[] = {"virtual", "webcam", "none",
                                        "playback"};
constexpr const char* kSourceLabels[] = {
    "virtual (gamepad-driven player)", "webcam (pose estimation)",
    "none (empty play space)", "playback (recording)"};
constexpr const char* kModelQualities[] = {"auto", "lite", "full", "heavy"};
constexpr const char* kExecutionProviders[] = {"auto", "dml", "cpu"};
constexpr const char* kTiltModes[] = {"simulate", "ignore"};
constexpr const char* kTiltModeLabels[] = {"simulate (rotate the sensor)",
                                           "ignore (report the angle only)"};

struct CaptureSize {
  int width;
  int height;
  const char* label;
};
constexpr CaptureSize kCaptureSizes[] = {
    {320, 240, "320x240"},   {640, 480, "640x480"},   {800, 600, "800x600"},
    {1280, 720, "1280x720"}, {1280, 960, "1280x960"}, {1920, 1080, "1920x1080"},
};
constexpr int kCaptureFpsValues[] = {15, 24, 30, 60};

constexpr double kStatusRefreshSeconds = 2.0;
constexpr double kApplyMessageSeconds = 4.0;

constexpr const char* kRuntimeFiles[] = {"onnxruntime.dll", "DirectML.dll"};
constexpr const char* kModelFiles[] = {
    "pose_detection.onnx", "pose_landmark_lite.onnx", "pose_landmark_full.onnx",
    "pose_landmark_heavy.onnx", "models.json"};

int IndexOf(const char* const* names, size_t count, const std::string& value,
            int fallback) {
  for (size_t i = 0; i < count; ++i) {
    if (value == names[i]) {
      return static_cast<int>(i);
    }
  }
  return fallback;
}

bool IsAllDigits(const std::string& value) {
  if (value.empty()) {
    return false;
  }
  return std::all_of(value.begin(), value.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::string LowerAscii(std::string value) {
  for (auto& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

bool Combo(const char* label, int* index, const char* const* items,
           size_t count) {
  return ImGui::Combo(label, index, items, static_cast<int>(count));
}

// The cvars are defined in nui_flags.cc, so the OVERRIDE_* macros (which
// need the defining translation unit's cv::cv_* handle) cannot be used here;
// the registry is looked up by name instead. Same semantics as
// OVERRIDE_*: the value takes effect now and is what SaveConfig writes.
template <typename T>
void OverrideCvar(const char* name, const T& value) {
  if (!cvar::ConfigVars) {
    return;
  }
  auto it = cvar::ConfigVars->find(name);
  if (it == cvar::ConfigVars->end()) {
    XELOGW("NUI settings: unknown cvar '{}'", name);
    return;
  }
  auto* config_var = dynamic_cast<cvar::ConfigVar<T>*>(it->second);
  if (!config_var) {
    XELOGW("NUI settings: cvar '{}' has an unexpected type", name);
    return;
  }
  config_var->OverrideConfigValue(value);
}

}  // namespace

NuiSettingsDialog::NuiSettingsDialog(ui::ImGuiDrawer* imgui_drawer,
                                     EmulatorWindow& emulator_window)
    : ui::ImGuiDialog(imgui_drawer), emulator_window_(emulator_window) {
  RefreshCameras();
  LoadFromCvars();
  RefreshFileStatus();
}

void NuiSettingsDialog::LoadFromCvars() {
  enabled_ = cvars::nui;
  source_index_ =
      IndexOf(kSourceNames, std::size(kSourceNames), cvars::nui_source, 0);

  // Camera: an index, a name fragment, or empty (first camera).
  xe::string_util::copy_truncating(camera_text_, cvars::nui_camera,
                                   sizeof(camera_text_));
  camera_index_ = -1;
  if (!cvars::nui_camera.empty()) {
    camera_index_ = -2;
    if (IsAllDigits(cvars::nui_camera)) {
      const int index = std::atoi(cvars::nui_camera.c_str());
      if (index >= 0 && index < static_cast<int>(cameras_.size())) {
        camera_index_ = index;
      }
    } else {
      const std::string needle = LowerAscii(cvars::nui_camera);
      for (size_t i = 0; i < cameras_.size(); ++i) {
        if (LowerAscii(cameras_[i].name).find(needle) != std::string::npos) {
          camera_index_ = static_cast<int>(i);
          break;
        }
      }
    }
  }

  capture_width_ = cvars::nui_capture_width;
  capture_height_ = cvars::nui_capture_height;
  capture_fps_ = cvars::nui_capture_fps;
  capture_size_index_ = static_cast<int>(std::size(kCaptureSizes));
  for (size_t i = 0; i < std::size(kCaptureSizes); ++i) {
    if (kCaptureSizes[i].width == capture_width_ &&
        kCaptureSizes[i].height == capture_height_) {
      capture_size_index_ = static_cast<int>(i);
      break;
    }
  }
  capture_fps_index_ = static_cast<int>(std::size(kCaptureFpsValues));
  for (size_t i = 0; i < std::size(kCaptureFpsValues); ++i) {
    if (kCaptureFpsValues[i] == capture_fps_) {
      capture_fps_index_ = static_cast<int>(i);
      break;
    }
  }

  model_quality_index_ = IndexOf(kModelQualities, std::size(kModelQualities),
                                 cvars::nui_model_quality, 0);
  execution_provider_index_ =
      IndexOf(kExecutionProviders, std::size(kExecutionProviders),
              cvars::nui_execution_provider, 0);
  segmentation_ = cvars::nui_segmentation;
  xe::string_util::copy_truncating(playback_path_,
                                   xe::path_to_utf8(cvars::nui_playback_path),
                                   sizeof(playback_path_));
  playback_loop_ = cvars::nui_playback_loop;
  playback_realtime_ = cvars::nui_playback_realtime;

  camera_hfov_ = static_cast<float>(cvars::nui_camera_hfov);
  camera_height_ = static_cast<float>(cvars::nui_camera_height);
  camera_pitch_ = static_cast<float>(cvars::nui_camera_pitch);
  user_scale_ = static_cast<float>(cvars::nui_user_scale);
  camera_mirrored_ = cvars::nui_camera_mirrored;
  auto_bind_user0_ = cvars::nui_auto_bind_user0;
  smoothing_override_ = cvars::nui_smoothing_override;
  smoothing_ = static_cast<float>(cvars::nui_smoothing);
  correction_ = static_cast<float>(cvars::nui_correction);
  prediction_ = static_cast<float>(cvars::nui_prediction);
  jitter_radius_ = static_cast<float>(cvars::nui_jitter_radius);
  max_deviation_radius_ = static_cast<float>(cvars::nui_max_deviation_radius);
  tilt_mode_index_ =
      IndexOf(kTiltModes, std::size(kTiltModes), cvars::nui_tilt_mode, 0);
  max_players_ = std::clamp(cvars::nui_max_players, 1,
                            static_cast<int>(nui::kMaxTrackedSkeletons));

  xe::string_util::copy_truncating(record_path_,
                                   xe::path_to_utf8(cvars::nui_record_path),
                                   sizeof(record_path_));
  dirty_restart_ = false;
}

void NuiSettingsDialog::RefreshCameras() {
  cameras_ = nui::CameraCapture::EnumerateDevices();
}

std::filesystem::path NuiSettingsDialog::nui_folder() const {
  return xe::filesystem::GetExecutableFolder() / "nui";
}

std::filesystem::path NuiSettingsDialog::runtime_folder() const {
  if (!cvars::nui_runtime_path.empty()) {
    return cvars::nui_runtime_path;
  }
  return nui_folder() / "runtime";
}

std::filesystem::path NuiSettingsDialog::models_folder() const {
  if (!cvars::nui_model_path.empty()) {
    return cvars::nui_model_path;
  }
  return nui_folder() / "models";
}

void NuiSettingsDialog::RefreshFileStatus() {
  file_status_.clear();
  runtime_complete_ = true;
  models_complete_ = true;
  std::error_code ec;
  const auto runtime = runtime_folder();
  for (const char* name : kRuntimeFiles) {
    FileStatus status;
    status.name = name;
    status.path = runtime / name;
    status.exists = std::filesystem::is_regular_file(status.path, ec);
    runtime_complete_ = runtime_complete_ && status.exists;
    file_status_.push_back(std::move(status));
  }
  const auto models = models_folder();
  bool any_landmark = false;
  for (const char* name : kModelFiles) {
    FileStatus status;
    status.name = name;
    status.path = models / name;
    status.exists = std::filesystem::is_regular_file(status.path, ec);
    if (std::strncmp(name, "pose_landmark_", 14) == 0) {
      any_landmark = any_landmark || status.exists;
    } else {
      models_complete_ = models_complete_ && status.exists;
    }
    file_status_.push_back(std::move(status));
  }
  models_complete_ = models_complete_ && any_landmark;
  setup_command_ =
      fmt::format("python tools/nui/setup_nui.py --exe \"{}\"",
                  xe::path_to_utf8(xe::filesystem::GetExecutablePath()));
  last_status_refresh_ = ImGui::GetTime();
}

void NuiSettingsDialog::ApplyLiveSettings() {
  OverrideCvar<double>("nui_camera_hfov", static_cast<double>(camera_hfov_));
  OverrideCvar<double>("nui_camera_height",
                       static_cast<double>(camera_height_));
  OverrideCvar<double>("nui_camera_pitch", static_cast<double>(camera_pitch_));
  OverrideCvar<double>("nui_user_scale", static_cast<double>(user_scale_));
  OverrideCvar<bool>("nui_camera_mirrored", camera_mirrored_);
  OverrideCvar<bool>("nui_auto_bind_user0", auto_bind_user0_);
  OverrideCvar<bool>("nui_smoothing_override", smoothing_override_);
  OverrideCvar<double>("nui_smoothing", static_cast<double>(smoothing_));
  OverrideCvar<double>("nui_correction", static_cast<double>(correction_));
  OverrideCvar<double>("nui_prediction", static_cast<double>(prediction_));
  OverrideCvar<double>("nui_jitter_radius",
                       static_cast<double>(jitter_radius_));
  OverrideCvar<double>("nui_max_deviation_radius",
                       static_cast<double>(max_deviation_radius_));
  OverrideCvar<std::string>("nui_tilt_mode",
                            std::string(kTiltModes[tilt_mode_index_]));
  OverrideCvar<int32_t>("nui_max_players", max_players_);
  if (auto* nui_system = emulator_window_.emulator()->nui_system()) {
    nui_system->RefreshDeviceState();
  }
}

void NuiSettingsDialog::ApplyAndRestart() {
  OverrideCvar<bool>("nui", enabled_);
  OverrideCvar<std::string>("nui_source",
                            std::string(kSourceNames[source_index_]));
  std::string camera;
  if (camera_index_ >= 0 && camera_index_ < static_cast<int>(cameras_.size())) {
    // Prefer the name (stable across re-enumeration); fall back to the index
    // when several cameras share it.
    const std::string& name = cameras_[camera_index_].name;
    const size_t same_name = std::count_if(
        cameras_.begin(), cameras_.end(),
        [&](const nui::CameraDeviceInfo& info) { return info.name == name; });
    camera =
        same_name > 1 || name.empty() ? std::to_string(camera_index_) : name;
  } else if (camera_index_ == -2) {
    camera = camera_text_;
  }
  OverrideCvar<std::string>("nui_camera", camera);
  if (capture_size_index_ >= 0 &&
      capture_size_index_ < static_cast<int>(std::size(kCaptureSizes))) {
    capture_width_ = kCaptureSizes[capture_size_index_].width;
    capture_height_ = kCaptureSizes[capture_size_index_].height;
  }
  if (capture_fps_index_ >= 0 &&
      capture_fps_index_ < static_cast<int>(std::size(kCaptureFpsValues))) {
    capture_fps_ = kCaptureFpsValues[capture_fps_index_];
  }
  OverrideCvar<int32_t>("nui_capture_width", capture_width_);
  OverrideCvar<int32_t>("nui_capture_height", capture_height_);
  OverrideCvar<int32_t>("nui_capture_fps", capture_fps_);
  OverrideCvar<std::string>("nui_model_quality",
                            std::string(kModelQualities[model_quality_index_]));
  OverrideCvar<std::string>(
      "nui_execution_provider",
      std::string(kExecutionProviders[execution_provider_index_]));
  OverrideCvar<bool>("nui_segmentation", segmentation_);
  OverrideCvar<std::filesystem::path>("nui_playback_path",
                                      xe::to_path(std::string(playback_path_)));
  OverrideCvar<bool>("nui_playback_loop", playback_loop_);
  OverrideCvar<bool>("nui_playback_realtime", playback_realtime_);
  ApplyLiveSettings();

  auto* nui_system = emulator_window_.emulator()->nui_system();
  if (nui_system) {
    nui_system->RestartSource();
    apply_message_ = fmt::format(
        "Applied: source '{}' {}", cvars::nui_source,
        nui_system->is_device_present() ? "ready" : "(no sensor: see the log)");
  } else {
    apply_message_ = "Applied (Kinect emulation not available)";
  }
  apply_message_until_ = ImGui::GetTime() + kApplyMessageSeconds;
  dirty_restart_ = false;
  XELOGI("NUI settings: {}", apply_message_);
}

void NuiSettingsDialog::DrawSourceSection() {
  if (ImGui::Checkbox("Enable Kinect emulation (nui)", &enabled_)) {
    dirty_restart_ = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Titles started after applying see a connected sensor.");
  }
  ImGui::SetNextItemWidth(-1.0f);
  if (Combo("##Source", &source_index_, kSourceLabels,
            std::size(kSourceLabels))) {
    dirty_restart_ = true;
  }
}

void NuiSettingsDialog::DrawCameraSection() {
  if (!ImGui::CollapsingHeader("Webcam", ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  // Camera combo: first camera, every enumerated device, custom text.
  std::string preview;
  if (camera_index_ == -1) {
    preview = "(first camera)";
  } else if (camera_index_ == -2) {
    preview = fmt::format("custom: {}", camera_text_);
  } else if (camera_index_ < static_cast<int>(cameras_.size())) {
    preview = cameras_[camera_index_].name;
  }
  ImGui::SetNextItemWidth(-90.0f);
  if (ImGui::BeginCombo("##Camera", preview.c_str())) {
    if (ImGui::Selectable("(first camera)", camera_index_ == -1)) {
      camera_index_ = -1;
      dirty_restart_ = true;
    }
    for (size_t i = 0; i < cameras_.size(); ++i) {
      const std::string label =
          fmt::format("{}: {}", i, cameras_[i].name.c_str());
      if (ImGui::Selectable(label.c_str(),
                            camera_index_ == static_cast<int>(i))) {
        camera_index_ = static_cast<int>(i);
        dirty_restart_ = true;
      }
      if (ImGui::IsItemHovered() && !cameras_[i].id.empty()) {
        ImGui::SetTooltip("%s", cameras_[i].id.c_str());
      }
    }
    if (ImGui::Selectable("custom (index or name fragment)",
                          camera_index_ == -2)) {
      camera_index_ = -2;
      dirty_restart_ = true;
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::Button("Refresh", ImVec2(-1.0f, 0.0f))) {
    RefreshCameras();
    if (camera_index_ >= static_cast<int>(cameras_.size())) {
      camera_index_ = -1;
    }
  }
  if (camera_index_ == -2) {
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##CameraText",
                                 "0, 1, ... or part of the name", camera_text_,
                                 sizeof(camera_text_))) {
      dirty_restart_ = true;
    }
  }
  if (cameras_.empty()) {
    ImGui::TextDisabled("No cameras found.");
  }

  // Capture mode.
  const char* size_labels[std::size(kCaptureSizes) + 1];
  for (size_t i = 0; i < std::size(kCaptureSizes); ++i) {
    size_labels[i] = kCaptureSizes[i].label;
  }
  const std::string custom_size =
      fmt::format("custom ({}x{})", capture_width_, capture_height_);
  size_labels[std::size(kCaptureSizes)] = custom_size.c_str();
  ImGui::SetNextItemWidth(140.0f);
  if (Combo("Size", &capture_size_index_, size_labels,
            std::size(size_labels))) {
    dirty_restart_ = true;
  }
  ImGui::SameLine();
  const char* fps_labels[std::size(kCaptureFpsValues) + 1];
  std::string fps_strings[std::size(kCaptureFpsValues) + 1];
  for (size_t i = 0; i < std::size(kCaptureFpsValues); ++i) {
    fps_strings[i] = std::to_string(kCaptureFpsValues[i]);
    fps_labels[i] = fps_strings[i].c_str();
  }
  fps_strings[std::size(kCaptureFpsValues)] =
      fmt::format("custom ({})", capture_fps_);
  fps_labels[std::size(kCaptureFpsValues)] =
      fps_strings[std::size(kCaptureFpsValues)].c_str();
  ImGui::SetNextItemWidth(110.0f);
  if (Combo("fps", &capture_fps_index_, fps_labels, std::size(fps_labels))) {
    dirty_restart_ = true;
  }

  // Model.
  ImGui::SetNextItemWidth(140.0f);
  if (Combo("Model quality", &model_quality_index_, kModelQualities,
            std::size(kModelQualities))) {
    dirty_restart_ = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("auto: full on DirectML, lite on CPU");
  }
  ImGui::SameLine();
  ImGui::SetNextItemWidth(110.0f);
  if (Combo("Provider", &execution_provider_index_, kExecutionProviders,
            std::size(kExecutionProviders))) {
    dirty_restart_ = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("dml: DirectML on the GPU; cpu: ONNX Runtime CPU");
  }
  if (ImGui::Checkbox("Person segmentation for the player silhouette",
                      &segmentation_)) {
    dirty_restart_ = true;
  }
}

void NuiSettingsDialog::DrawGeometrySection() {
  if (!ImGui::CollapsingHeader("Camera geometry (live)",
                               ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  bool changed = false;
  ImGui::SetNextItemWidth(220.0f);
  changed |= ImGui::SliderFloat("Horizontal FOV (deg)", &camera_hfov_, 40.0f,
                                120.0f, "%.1f");
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Field of view of the webcam. Wrong values make you appear nearer "
        "or farther than you are.");
  }
  ImGui::SetNextItemWidth(220.0f);
  changed |= ImGui::SliderFloat("Camera height (m)", &camera_height_, 0.0f,
                                2.0f, "%.2f");
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Height above the floor; 0 = unknown (no floor plane).");
  }
  ImGui::SetNextItemWidth(220.0f);
  changed |= ImGui::SliderFloat("Camera pitch (deg)", &camera_pitch_, -30.0f,
                                30.0f, "%.1f");
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Positive = looking up.");
  }
  ImGui::SetNextItemWidth(220.0f);
  changed |=
      ImGui::SliderFloat("Distance scale", &user_scale_, 0.5f, 2.0f, "%.2f");
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Multiplies the estimated distance; calibrate so 2 m reads as 2 m.");
  }
  changed |=
      ImGui::Checkbox("Camera image is already mirrored", &camera_mirrored_);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Set when a raised right hand shows on the wrong side of the "
        "preview.");
  }
  if (changed) {
    ApplyLiveSettings();
  }
}

void NuiSettingsDialog::DrawTrackingSection() {
  if (!ImGui::CollapsingHeader("Tracking (live)")) {
    return;
  }
  bool changed = false;
  ImGui::SetNextItemWidth(120.0f);
  changed |= ImGui::SliderInt("Tracked players", &max_players_, 1,
                              static_cast<int>(nui::kMaxTrackedSkeletons));
  ImGui::SetNextItemWidth(220.0f);
  changed |= Combo("Tilt requests", &tilt_mode_index_, kTiltModeLabels,
                   std::size(kTiltModeLabels));
  changed |= ImGui::Checkbox("Bind the first tracked player to user 0",
                             &auto_bind_user0_);
  changed |= ImGui::Checkbox("Override the title's smoothing parameters",
                             &smoothing_override_);
  ImGui::BeginDisabled(!smoothing_override_);
  ImGui::Indent();
  ImGui::SetNextItemWidth(200.0f);
  changed |= ImGui::SliderFloat("Smoothing", &smoothing_, 0.0f, 1.0f, "%.2f");
  ImGui::SetNextItemWidth(200.0f);
  changed |= ImGui::SliderFloat("Correction", &correction_, 0.0f, 1.0f, "%.2f");
  ImGui::SetNextItemWidth(200.0f);
  changed |= ImGui::SliderFloat("Prediction", &prediction_, 0.0f, 1.0f, "%.2f");
  ImGui::SetNextItemWidth(200.0f);
  changed |= ImGui::SliderFloat("Jitter radius (m)", &jitter_radius_, 0.0f,
                                0.5f, "%.3f");
  ImGui::SetNextItemWidth(200.0f);
  changed |= ImGui::SliderFloat("Max deviation radius (m)",
                                &max_deviation_radius_, 0.0f, 0.5f, "%.3f");
  ImGui::Unindent();
  ImGui::EndDisabled();
  if (changed) {
    ApplyLiveSettings();
  }
}

void NuiSettingsDialog::DrawRuntimeStatusSection() {
  if (!ImGui::CollapsingHeader("Pose estimation runtime",
                               ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  if (ImGui::GetTime() - last_status_refresh_ > kStatusRefreshSeconds) {
    RefreshFileStatus();
  }
  const ImVec4 ok(0.4f, 0.9f, 0.4f, 1.0f);
  const ImVec4 missing(0.95f, 0.4f, 0.4f, 1.0f);
  ImGui::TextColored(runtime_complete_ ? ok : missing, "%s",
                     runtime_complete_ ? "Runtime: found" : "Runtime: missing");
  ImGui::SameLine();
  ImGui::TextDisabled("%s", xe::path_to_utf8(runtime_folder()).c_str());
  ImGui::TextColored(models_complete_ ? ok : missing, "%s",
                     models_complete_ ? "Models: found" : "Models: missing");
  ImGui::SameLine();
  ImGui::TextDisabled("%s", xe::path_to_utf8(models_folder()).c_str());
  if (ImGui::TreeNode("Files")) {
    for (const auto& status : file_status_) {
      ImGui::TextColored(status.exists ? ok : missing, "%s  %s",
                         status.exists ? "[ok]     " : "[missing]",
                         status.name.c_str());
    }
    ImGui::TreePop();
  }
  if (ImGui::Button("Open NUI folder")) {
    std::error_code ec;
    const auto folder = nui_folder();
    LaunchFileExplorer(std::filesystem::is_directory(folder, ec)
                           ? folder
                           : xe::filesystem::GetExecutableFolder());
  }
  ImGui::SameLine();
  if (ImGui::Button("Re-check")) {
    RefreshFileStatus();
  }
  if (!runtime_complete_ || !models_complete_) {
    ImGui::TextWrapped(
        "Install the ONNX Runtime and the BlazePose models with (needs "
        "Python 3):");
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputText("##SetupCommand", setup_command_.data(),
                     setup_command_.size() + 1, ImGuiInputTextFlags_ReadOnly);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Select and copy (Ctrl+C).");
    }
  }
}

void NuiSettingsDialog::DrawPlaybackSection() {
  if (!ImGui::CollapsingHeader("Playback and recording")) {
    return;
  }
  ImGui::Text("Playback file (.nuirec)");
  ImGui::SetNextItemWidth(-1.0f);
  if (ImGui::InputTextWithHint("##PlaybackPath", "path to a recording",
                               playback_path_, sizeof(playback_path_))) {
    dirty_restart_ = true;
  }
  if (ImGui::Checkbox("Loop", &playback_loop_)) {
    dirty_restart_ = true;
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Real time", &playback_realtime_)) {
    dirty_restart_ = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Off: one recorded frame per 30 Hz tick (deterministic).");
  }
  ImGui::Spacing();
  ImGui::Text("Record to (.nuirec)");
  ImGui::SetNextItemWidth(-1.0f);
  ImGui::InputTextWithHint("##RecordPath",
                           "empty: <storage>/nui_recording.nuirec",
                           record_path_, sizeof(record_path_));
  const bool recording = !cvars::nui_record_path.empty();
  ImGui::BeginDisabled(recording);
  if (ImGui::Button("Start recording")) {
    std::string path = record_path_;
    if (path.empty()) {
      path = xe::path_to_utf8(emulator_window_.emulator()->storage_root() /
                              "nui_recording.nuirec");
      xe::string_util::copy_truncating(record_path_, path,
                                       sizeof(record_path_));
    }
    OverrideCvar<std::filesystem::path>("nui_record_path", xe::to_path(path));
    XELOGI("NUI settings: recording requested to {}", path);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(!recording);
  if (ImGui::Button("Stop recording")) {
    OverrideCvar<std::filesystem::path>("nui_record_path",
                                        std::filesystem::path());
    XELOGI("NUI settings: recording stopped");
  }
  ImGui::EndDisabled();
  if (recording) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f), "Recording: %s",
                       xe::path_to_utf8(cvars::nui_record_path).c_str());
  }
}

void NuiSettingsDialog::OnDraw(ImGuiIO& io) {
  ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(20, 20), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSizeConstraints(ImVec2(460, 0), ImVec2(FLT_MAX, FLT_MAX));
  ImGui::SetNextWindowBgAlpha(0.85f);
  bool dialog_open = true;
  if (!ImGui::Begin(
          "Kinect settings", &dialog_open,
          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::End();
    Close();
    return;
  }

  DrawSourceSection();
  DrawCameraSection();
  DrawGeometrySection();
  DrawTrackingSection();
  DrawRuntimeStatusSection();
  DrawPlaybackSection();

  ImGui::Separator();
  if (ImGui::Button("Apply", ImVec2(120.0f, 0.0f))) {
    ApplyAndRestart();
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Writes the settings and restarts the source (camera, model, "
        "playback file). Live settings are applied as you change them.");
  }
  ImGui::SameLine();
  if (ImGui::Button("Save to config", ImVec2(120.0f, 0.0f))) {
    config::SaveConfig();
    apply_message_ = "Configuration saved";
    apply_message_until_ = ImGui::GetTime() + kApplyMessageSeconds;
  }
  ImGui::SameLine();
  if (ImGui::Button("Reload", ImVec2(90.0f, 0.0f))) {
    LoadFromCvars();
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Discard unapplied edits and reread the settings.");
  }
  if (dirty_restart_) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.95f, 0.8f, 0.3f, 1.0f),
                       "Apply to restart the source");
  } else if (ImGui::GetTime() < apply_message_until_) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", apply_message_.c_str());
  }

  ImGui::End();

  if (!dialog_open) {
    Close();
    emulator_window_.ToggleNuiSettingsDialog();
    // `this` might have been destroyed by ToggleNuiSettingsDialog.
    return;
  }
}

}  // namespace app
}  // namespace xe
