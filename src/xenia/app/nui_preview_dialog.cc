/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/nui_preview_dialog.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "third_party/fmt/include/fmt/format.h"
#include "third_party/imgui/imgui.h"
#include "xenia/app/emulator_window.h"
#include "xenia/emulator.h"
#include "xenia/nui/camera_model.h"
#include "xenia/nui/nui_flags.h"
#include "xenia/nui/pose_estimator.h"

namespace xe {
namespace app {

namespace {

constexpr float kPreviewWidth = 640.0f;
constexpr float kCanvasWidth = 320.0f;
constexpr float kCanvasHeight = 240.0f;
constexpr auto kTextureUpdateInterval = std::chrono::milliseconds(66);
constexpr float kDegreesToRadians = 3.14159265358979323846f / 180.0f;

constexpr ImU32 kColorTracked = IM_COL32(80, 230, 80, 255);
constexpr ImU32 kColorInferred = IM_COL32(240, 200, 40, 255);
constexpr ImU32 kColorNotTracked = IM_COL32(200, 60, 60, 255);
constexpr ImU32 kColorFov = IM_COL32(80, 160, 255, 200);
constexpr ImU32 kColorLabel = IM_COL32(255, 255, 255, 255);
constexpr ImU32 kColorLabelShadow = IM_COL32(0, 0, 0, 200);
constexpr ImU32 kColorCanvasBackground = IM_COL32(24, 24, 32, 255);
constexpr ImU32 kColorCanvasBorder = IM_COL32(90, 90, 110, 255);

// Kinect skeleton bones (NUI_SKELETON_POSITION_INDEX pairs).
constexpr std::pair<nui::Joint, nui::Joint> kKinectBones[] = {
    {nui::Joint::kHipCenter, nui::Joint::kSpine},
    {nui::Joint::kSpine, nui::Joint::kShoulderCenter},
    {nui::Joint::kShoulderCenter, nui::Joint::kHead},
    {nui::Joint::kShoulderCenter, nui::Joint::kShoulderLeft},
    {nui::Joint::kShoulderLeft, nui::Joint::kElbowLeft},
    {nui::Joint::kElbowLeft, nui::Joint::kWristLeft},
    {nui::Joint::kWristLeft, nui::Joint::kHandLeft},
    {nui::Joint::kShoulderCenter, nui::Joint::kShoulderRight},
    {nui::Joint::kShoulderRight, nui::Joint::kElbowRight},
    {nui::Joint::kElbowRight, nui::Joint::kWristRight},
    {nui::Joint::kWristRight, nui::Joint::kHandRight},
    {nui::Joint::kHipCenter, nui::Joint::kHipLeft},
    {nui::Joint::kHipLeft, nui::Joint::kKneeLeft},
    {nui::Joint::kKneeLeft, nui::Joint::kAnkleLeft},
    {nui::Joint::kAnkleLeft, nui::Joint::kFootLeft},
    {nui::Joint::kHipCenter, nui::Joint::kHipRight},
    {nui::Joint::kHipRight, nui::Joint::kKneeRight},
    {nui::Joint::kKneeRight, nui::Joint::kAnkleRight},
    {nui::Joint::kAnkleRight, nui::Joint::kFootRight},
};

// MediaPipe POSE_CONNECTIONS (BlazePose 33 landmarks).
constexpr std::pair<uint8_t, uint8_t> kBlazePoseBones[] = {
    {0, 1},   {1, 2},   {2, 3},   {3, 7},   {0, 4},   {4, 5},   {5, 6},
    {6, 8},   {9, 10},  {11, 12}, {11, 13}, {13, 15}, {15, 17}, {15, 19},
    {15, 21}, {17, 19}, {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22},
    {18, 20}, {11, 23}, {12, 24}, {23, 24}, {23, 25}, {24, 26}, {25, 27},
    {26, 28}, {27, 29}, {28, 30}, {29, 31}, {30, 32}, {27, 31}, {28, 32},
};

// Landmarks below this visibility are drawn as "inferred".
constexpr float kLandmarkTrackedVisibility = 0.5f;

// Tint per player index in the mask (1-based), 0xAABBGGRR for ImGui.
constexpr ImU32 kPlayerTints[nui::kMaxSkeletons] = {
    IM_COL32(80, 220, 80, 255),  IM_COL32(80, 140, 255, 255),
    IM_COL32(255, 200, 60, 255), IM_COL32(230, 80, 200, 255),
    IM_COL32(60, 220, 220, 255), IM_COL32(255, 120, 60, 255),
};

void DrawShadowedText(ImDrawList* draw_list, ImVec2 pos, const char* text) {
  draw_list->AddText(ImVec2(pos.x + 1.0f, pos.y + 1.0f), kColorLabelShadow,
                     text);
  draw_list->AddText(pos, kColorLabel, text);
}

void DrawJointSquare(ImDrawList* draw_list, ImVec2 center, ImU32 color) {
  draw_list->AddRectFilled(ImVec2(center.x - 2.0f, center.y - 2.0f),
                           ImVec2(center.x + 3.0f, center.y + 3.0f), color);
}

ImU32 ColorForJointState(nui::JointState state) {
  switch (state) {
    case nui::JointState::kTracked:
      return kColorTracked;
    case nui::JointState::kInferred:
      return kColorInferred;
    default:
      return kColorNotTracked;
  }
}

float Distance(const nui::Vec4& a, const nui::Vec4& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Tracking id the title sees for a source person key: the NuiSystem copies
// the body into a skeleton slot unchanged except for the ids, so the
// positions match.
uint32_t TrackingIdForPersonKey(const nui::SkeletonFrame* skeleton,
                                const nui::SourceFrame* source,
                                uint32_t person_key) {
  if (!skeleton || !source || person_key == nui::kInvalidTrackingId) {
    return nui::kInvalidTrackingId;
  }
  const uint32_t body_count = std::min(source->body_count, nui::kMaxSkeletons);
  for (uint32_t b = 0; b < body_count; ++b) {
    const nui::Skeleton& body = source->bodies[b];
    if (body.tracking_id != person_key) {
      continue;
    }
    uint32_t best = nui::kInvalidTrackingId;
    float best_distance = 1e-3f;
    for (const auto& candidate : skeleton->skeletons) {
      if (candidate.state == nui::SkeletonState::kNotTracked) {
        continue;
      }
      const float distance = Distance(candidate.position, body.position);
      if (distance < best_distance) {
        best_distance = distance;
        best = candidate.tracking_id;
      }
    }
    return best;
  }
  return nui::kInvalidTrackingId;
}

}  // namespace

NuiPreviewDialog::NuiPreviewDialog(ui::ImGuiDrawer* imgui_drawer,
                                   EmulatorWindow& emulator_window)
    : ui::ImGuiDialog(imgui_drawer), emulator_window_(emulator_window) {}

void NuiPreviewDialog::ReplaceTexture(uint32_t width, uint32_t height) {
  // Immediate textures are immutable: a new one replaces the old one.
  texture_ = imgui_drawer()->CreateRgbaTexture(
      width, height, scratch_rgba_.data(), ui::ImmediateTextureFilter::kLinear);
  texture_width_ = texture_ ? width : 0;
  texture_height_ = texture_ ? height : 0;
  last_texture_update_ = std::chrono::steady_clock::now();
}

void NuiPreviewDialog::UpdateWebcamTexture(const nui::CameraFrame& frame) {
  if (!frame.width || !frame.height || frame.rgba.empty()) {
    return;
  }
  const bool same =
      texture_ && !texture_is_mask_ && texture_sequence_ == frame.sequence;
  if (same) {
    return;
  }
  if (texture_ && std::chrono::steady_clock::now() - last_texture_update_ <
                      kTextureUpdateInterval) {
    return;
  }
  const uint32_t stride = frame.stride ? frame.stride : frame.width * 4;
  if (frame.rgba.size() < static_cast<size_t>(stride) * frame.height) {
    return;
  }
  // Resample (nearest) to the preview width, keeping the aspect ratio.
  const uint32_t width = static_cast<uint32_t>(kPreviewWidth);
  const uint32_t height = std::max<uint32_t>(
      1, static_cast<uint32_t>(kPreviewWidth * frame.height / frame.width));
  scratch_rgba_.resize(static_cast<size_t>(width) * height * 4);
  for (uint32_t y = 0; y < height; ++y) {
    const uint32_t sy = std::min(frame.height - 1, y * frame.height / height);
    const uint8_t* src_row =
        frame.rgba.data() + static_cast<size_t>(sy) * stride;
    uint8_t* dst_row =
        scratch_rgba_.data() + static_cast<size_t>(y) * width * 4;
    if (width == frame.width) {
      std::memcpy(dst_row, src_row, static_cast<size_t>(width) * 4);
      continue;
    }
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t sx = std::min(frame.width - 1, x * frame.width / width);
      std::memcpy(dst_row + static_cast<size_t>(x) * 4,
                  src_row + static_cast<size_t>(sx) * 4, 4);
    }
  }
  ReplaceTexture(width, height);
  texture_sequence_ = frame.sequence;
  texture_is_mask_ = false;
}

void NuiPreviewDialog::UpdateMaskTexture(const nui::SourceFrame& source) {
  const size_t pixel_count =
      static_cast<size_t>(nui::kDepthWidth) * nui::kDepthHeight;
  if (!source.has_player_mask || source.player_mask.size() < pixel_count) {
    return;
  }
  if (texture_ && texture_is_mask_ && texture_sequence_ == source.sequence) {
    return;
  }
  if (texture_ && std::chrono::steady_clock::now() - last_texture_update_ <
                      kTextureUpdateInterval) {
    return;
  }
  const bool have_depth =
      source.has_depth && source.depth_mm.size() >= pixel_count;
  scratch_rgba_.resize(pixel_count * 4);
  for (size_t i = 0; i < pixel_count; ++i) {
    uint8_t* px = scratch_rgba_.data() + i * 4;
    const uint8_t player = source.player_mask[i];
    if (player == 0 || player > nui::kMaxSkeletons) {
      // Background: faint depth shading when available.
      uint8_t shade = 0;
      if (have_depth && source.depth_mm[i] != nui::kDepthUnknown) {
        const int32_t mm = source.depth_mm[i];
        shade = static_cast<uint8_t>(std::clamp(
            96 - (mm - nui::kDepthMinMillimetres) * 80 /
                     (nui::kDepthMaxMillimetres - nui::kDepthMinMillimetres),
            16, 96));
      }
      px[0] = shade;
      px[1] = shade;
      px[2] = static_cast<uint8_t>(shade + 8);
      px[3] = 255;
      continue;
    }
    const ImU32 tint = kPlayerTints[player - 1];
    px[0] = static_cast<uint8_t>((tint >> IM_COL32_R_SHIFT) & 0xFF);
    px[1] = static_cast<uint8_t>((tint >> IM_COL32_G_SHIFT) & 0xFF);
    px[2] = static_cast<uint8_t>((tint >> IM_COL32_B_SHIFT) & 0xFF);
    px[3] = 255;
  }
  ReplaceTexture(nui::kDepthWidth, nui::kDepthHeight);
  texture_sequence_ = source.sequence;
  texture_is_mask_ = true;
}

void NuiPreviewDialog::DrawWebcamPreview(nui::NuiSystem* nui_system,
                                         nui::WebcamNuiSource* webcam,
                                         const nui::SkeletonFrame* skeleton,
                                         const nui::SourceFrame* source) {
  if (webcam->GetPreview(&preview_)) {
    preview_valid_ = true;
  }
  if (!preview_valid_) {
    ImGui::TextDisabled("Waiting for the first camera frame...");
    ImGui::Dummy(ImVec2(kPreviewWidth, kPreviewWidth * 3.0f / 4.0f));
    return;
  }
  const nui::CameraFrame& frame = preview_.frame;
  UpdateWebcamTexture(frame);

  Canvas canvas;
  canvas.size = ImVec2(kPreviewWidth, kPreviewWidth * 3.0f / 4.0f);
  if (frame.width && frame.height) {
    canvas.size.y = kPreviewWidth * frame.height / frame.width;
  }
  if (texture_ && !texture_is_mask_) {
    ImGui::Image(reinterpret_cast<ImTextureID>(texture_.get()), canvas.size);
  } else {
    ImGui::Dummy(canvas.size);
  }
  canvas.origin = ImGui::GetItemRectMin();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();

  // Kinect depth camera field of view inside the webcam's field of view.
  {
    const float hfov =
        std::clamp(static_cast<float>(cvars::nui_camera_hfov), 10.0f, 170.0f);
    const float tan_half_h = std::tan(hfov * 0.5f * kDegreesToRadians);
    const float aspect = canvas.size.y / canvas.size.x;
    const float tan_half_v = tan_half_h * aspect;
    const float fx = std::min(1.0f, std::tan(nui::kDepthHorizontalFovDegrees *
                                             0.5f * kDegreesToRadians) /
                                        tan_half_h);
    const float fy = std::min(1.0f, std::tan(nui::kDepthVerticalFovDegrees *
                                             0.5f * kDegreesToRadians) /
                                        tan_half_v);
    const ImVec2 p0 = canvas.ToScreen(0.5f - fx * 0.5f, 0.5f - fy * 0.5f);
    const ImVec2 p1 = canvas.ToScreen(0.5f + fx * 0.5f, 0.5f + fy * 0.5f);
    draw_list->AddRect(p0, p1, kColorFov, 0.0f, 0, 1.5f);
    DrawShadowedText(draw_list, ImVec2(p0.x + 3.0f, p0.y + 2.0f),
                     "Kinect depth FOV");
  }

  // Landmarks and bones of every person.
  for (size_t p = 0; p < preview_.poses.size(); ++p) {
    const nui::PoseResult& pose = preview_.poses[p];
    if (!pose.valid) {
      continue;
    }
    auto landmark_pos = [&](uint8_t index) {
      const nui::PoseLandmark& lm = pose.landmarks[index];
      return canvas.ToScreen(lm.x, lm.y);
    };
    auto landmark_tracked = [&](uint8_t index) {
      return pose.landmarks[index].visibility >= kLandmarkTrackedVisibility;
    };
    for (const auto& bone : kBlazePoseBones) {
      const bool tracked =
          landmark_tracked(bone.first) && landmark_tracked(bone.second);
      draw_list->AddLine(landmark_pos(bone.first), landmark_pos(bone.second),
                         tracked ? kColorTracked : kColorInferred, 1.5f);
    }
    for (uint32_t i = 0; i < nui::kPoseLandmarkCount; ++i) {
      const uint8_t index = static_cast<uint8_t>(i);
      DrawJointSquare(draw_list, landmark_pos(index),
                      landmark_tracked(index) ? kColorTracked : kColorInferred);
    }
    const uint32_t person_key =
        p < preview_.person_keys.size() ? preview_.person_keys[p] : 0;
    const uint32_t tracking_id =
        TrackingIdForPersonKey(skeleton, source, person_key);
    std::string label;
    if (tracking_id != nui::kInvalidTrackingId) {
      label = fmt::format("id {} (key {})", tracking_id, person_key);
    } else if (person_key) {
      label = fmt::format("key {}", person_key);
    } else {
      label = fmt::format("person {}", p + 1);
    }
    const ImVec2 head =
        landmark_pos(static_cast<uint8_t>(nui::PoseLandmarkIndex::kNose));
    DrawShadowedText(draw_list, ImVec2(head.x + 8.0f, head.y - 18.0f),
                     label.c_str());
  }

  ImGui::TextDisabled("Raise your right hand: it should appear on the right.");
  if (frame.width && frame.height) {
    ImGui::SameLine();
    ImGui::TextDisabled("(%ux%u)", frame.width, frame.height);
  }
}

void NuiPreviewDialog::DrawSkeletonCanvas(const nui::SkeletonFrame* skeleton,
                                          const nui::SourceFrame* source) {
  Canvas canvas;
  canvas.size = ImVec2(kCanvasWidth, kCanvasHeight);
  if (source) {
    UpdateMaskTexture(*source);
  }
  const bool have_mask =
      texture_ && texture_is_mask_ && source && source->has_player_mask;
  if (have_mask) {
    ImGui::Image(reinterpret_cast<ImTextureID>(texture_.get()), canvas.size);
  } else {
    ImGui::Dummy(canvas.size);
  }
  canvas.origin = ImGui::GetItemRectMin();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  const ImVec2 p1(canvas.origin.x + canvas.size.x,
                  canvas.origin.y + canvas.size.y);
  if (!have_mask) {
    draw_list->AddRectFilled(canvas.origin, p1, kColorCanvasBackground);
  }
  draw_list->AddRect(canvas.origin, p1, kColorCanvasBorder);

  if (!skeleton) {
    DrawShadowedText(draw_list,
                     ImVec2(canvas.origin.x + 6.0f, canvas.origin.y + 6.0f),
                     "No frame published yet");
    return;
  }

  nui::DepthCameraModel camera;
  auto project = [&](const nui::Vec4& point, ImVec2* out) {
    float u = 0.0f;
    float v = 0.0f;
    uint16_t depth = 0;
    if (!camera.Project(point, &u, &v, &depth)) {
      return false;
    }
    *out = canvas.ToScreen(u / static_cast<float>(camera.width),
                           v / static_cast<float>(camera.height));
    return true;
  };

  for (const auto& body : skeleton->skeletons) {
    if (body.state == nui::SkeletonState::kNotTracked) {
      continue;
    }
    ImVec2 position;
    if (body.state == nui::SkeletonState::kPositionOnly) {
      if (project(body.position, &position)) {
        draw_list->AddCircle(position, 6.0f, kColorInferred, 0, 1.5f);
        DrawShadowedText(
            draw_list, ImVec2(position.x + 8.0f, position.y - 8.0f),
            fmt::format("id {} (position only)", body.tracking_id).c_str());
      }
      continue;
    }
    for (const auto& bone : kKinectBones) {
      const auto a = static_cast<size_t>(bone.first);
      const auto b = static_cast<size_t>(bone.second);
      if (body.joint_states[a] == nui::JointState::kNotTracked ||
          body.joint_states[b] == nui::JointState::kNotTracked) {
        continue;
      }
      ImVec2 pa;
      ImVec2 pb;
      if (!project(body.joints[a], &pa) || !project(body.joints[b], &pb)) {
        continue;
      }
      const bool tracked = body.joint_states[a] == nui::JointState::kTracked &&
                           body.joint_states[b] == nui::JointState::kTracked;
      draw_list->AddLine(pa, pb, tracked ? kColorTracked : kColorInferred,
                         1.5f);
    }
    for (uint32_t j = 0; j < nui::kJointCount; ++j) {
      if (body.joint_states[j] == nui::JointState::kNotTracked) {
        continue;
      }
      ImVec2 pj;
      if (project(body.joints[j], &pj)) {
        DrawJointSquare(draw_list, pj,
                        ColorForJointState(body.joint_states[j]));
      }
    }
    const nui::Vec4& head = body.joints[static_cast<size_t>(nui::Joint::kHead)];
    if (project(head, &position)) {
      std::string label = fmt::format("id {}", body.tracking_id);
      if (body.user_index != nui::kInvalidUserIndex) {
        label += fmt::format(" user {}", body.user_index);
      }
      DrawShadowedText(draw_list, ImVec2(position.x + 8.0f, position.y - 18.0f),
                       label.c_str());
    }
  }
}

void NuiPreviewDialog::DrawStatus(nui::NuiSystem* nui_system,
                                  const nui::SkeletonFrame* skeleton) {
  const nui::NuiSystem::Stats stats = nui_system->GetStats();
  ImGui::Text("Source: %s   capture %.1f fps   inference %.1f ms",
              nui_system->source_name().c_str(), stats.source.capture_fps,
              stats.source.inference_ms);
  ImGui::Text("Frames published: %llu   tracked bodies: %u   %s",
              static_cast<unsigned long long>(stats.frames_published),
              stats.tracked_bodies,
              nui_system->initialized() ? "(title initialized)"
                                        : "(title not initialized)");
  if (!stats.source.status.empty()) {
    ImGui::TextWrapped("%s", stats.source.status.c_str());
  }
  if (!skeleton) {
    return;
  }
  for (const auto& body : skeleton->skeletons) {
    if (body.state != nui::SkeletonState::kTracked) {
      continue;
    }
    const nui::Vec4& hip =
        body.joints[static_cast<size_t>(nui::Joint::kHipCenter)];
    ImGui::Text("  id %u hip centre: x %+.2f  y %+.2f  z %.2f m%s",
                body.tracking_id, hip.x, hip.y, hip.z,
                body.quality_flags ? "  (clipped)" : "");
  }
}

void NuiPreviewDialog::OnDraw(ImGuiIO& io) {
  // Next to the menu bar it was opened from, like the display config dialog.
  ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(20, 20), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.85f);
  bool dialog_open = true;
  if (!ImGui::Begin(
          "Kinect camera preview", &dialog_open,
          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::End();
    Close();
    return;
  }

  nui::NuiSystem* nui_system = emulator_window_.emulator()->nui_system();
  if (!nui_system) {
    ImGui::TextDisabled("Kinect emulation is not available.");
  } else if (!nui_system->is_enabled()) {
    ImGui::TextDisabled(
        "Kinect emulation is disabled (nui=false). Enable it in NUI > "
        "Kinect settings and apply.");
  } else {
    nui::SkeletonFrame skeleton;
    std::shared_ptr<const nui::SourceFrame> source;
    const bool have_frames = nui_system->GetLatestFrames(&skeleton, &source);
    const nui::SkeletonFrame* skeleton_ptr = have_frames ? &skeleton : nullptr;

    // Only short, thread-safe calls on the source from the UI thread.
    nui::NuiSource* nui_source = nui_system->source_for_ui();
    auto* webcam = dynamic_cast<nui::WebcamNuiSource*>(nui_source);
    if (webcam) {
      DrawWebcamPreview(nui_system, webcam, skeleton_ptr, source.get());
    } else {
      if (!nui_source) {
        ImGui::TextDisabled(
            "The source starts when a title initializes the sensor.");
      }
      DrawSkeletonCanvas(skeleton_ptr, source.get());
    }
    ImGui::Separator();
    DrawStatus(nui_system, skeleton_ptr);
  }

  ImGui::End();

  if (!dialog_open) {
    Close();
    emulator_window_.ToggleNuiPreviewDialog();
    // `this` might have been destroyed by ToggleNuiPreviewDialog.
    return;
  }
}

}  // namespace app
}  // namespace xe
