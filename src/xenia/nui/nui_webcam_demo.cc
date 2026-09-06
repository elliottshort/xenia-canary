/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Standalone console tool for the webcam Kinect source: lists cameras,
// grabs frames, runs the pose estimator, or drives WebcamNuiSource through
// the NuiSource interface and dumps what a title would see.
//
//   xenia-nui-webcam-demo --list
//   xenia-nui-webcam-demo --capture [N] [--out DIR]
//   xenia-nui-webcam-demo --pose [SECONDS] [--out DIR]
//   xenia-nui-webcam-demo --source [SECONDS] [--out DIR]
//
// All nui_* cvars are accepted (--nui_camera=1, --nui_execution_provider=cpu,
// ...). Exit codes: 0 success, 1 usage, 2 camera error, 3 model error.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

#include "xenia/base/console_app_main.h"
#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/threading.h"
#include "xenia/nui/camera_capture.h"
#include "xenia/nui/nui_flags.h"
#include "xenia/nui/nui_source.h"
#include "xenia/nui/nui_types.h"
#include "xenia/nui/pose_estimator.h"
#include "xenia/nui/sources/webcam_nui_source.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#undef _CRT_SECURE_NO_WARNINGS
#undef _CRT_NONSTDC_NO_DEPRECATE
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "third_party/stb/stb_image_write.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace xe {
namespace nui {

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 1;
constexpr int kExitCamera = 2;
constexpr int kExitModel = 3;

enum class Mode {
  kNone,
  kList,
  kCapture,
  kPose,
  kSource,
};

struct DemoArgs {
  Mode mode = Mode::kNone;
  int count = -1;  // frames (capture) or seconds (pose/source)
  std::filesystem::path out;
  bool help = false;
};

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void PrintUsage(const std::string& exe) {
  fmt::print(
      "Usage: {} MODE [--out DIR] [--nui_*=VALUE ...]\n"
      "\n"
      "Modes:\n"
      "  --list               list the cameras Media Foundation exposes\n"
      "  --capture [N]        grab N frames (default 30), write "
      "frame_NNN.png,\n"
      "                       print the achieved frame rate and brightness\n"
      "  --pose [SECONDS]     run pose estimation live (default 10 s), print "
      "a\n"
      "                       summary every second and write pose_NNN.png\n"
      "  --source [SECONDS]   run the webcam NuiSource as the emulator does\n"
      "                       (default 10 s), print the bodies every second\n"
      "                       and write depth_NNN.png / color_NNN.png\n"
      "\n"
      "Options:\n"
      "  --usage, /?          this text (--help prints the cvar list instead)\n"
      "  --out DIR            folder for the images (default: current folder)\n"
      "  --nui_camera=NAME    camera index or part of its name\n"
      "  --nui_capture_width=W --nui_capture_height=H --nui_capture_fps=F\n"
      "  --nui_camera_hfov=DEG --nui_camera_mirrored=true|false\n"
      "  --nui_model_quality=lite|full|heavy|auto\n"
      "  --nui_execution_provider=auto|dml|cpu --nui_inference_threads=N\n"
      "  --nui_model_path=DIR --nui_runtime_path=DIR --nui_segmentation=BOOL\n"
      "  --nui_user_scale=F --nui_camera_height=M --nui_camera_pitch=DEG\n"
      "  --nui_max_players=1|2\n"
      "\n"
      "Exit codes: 0 ok, 1 usage, 2 camera error, 3 model error.\n",
      exe);
}

bool IsInteger(const std::string& s) {
  if (s.empty()) {
    return false;
  }
  for (char c : s) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return true;
}

// cvars are parsed by the console app framework (cxxopts with unrecognised
// options allowed); this picks the demo's own switches out of argv and skips
// everything that looks like a cvar.
bool ParseDemoArgs(const std::vector<std::string>& args, DemoArgs* out) {
  auto set_mode = [&](Mode mode) {
    if (out->mode != Mode::kNone) {
      fmt::print(stderr, "error: more than one mode given\n");
      return false;
    }
    out->mode = mode;
    return true;
  };
  for (size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    auto take_int = [&]() {
      if (i + 1 < args.size() && IsInteger(args[i + 1])) {
        out->count = std::atoi(args[++i].c_str());
      }
    };
    if (a == "--help" || a == "-h" || a == "/?" || a == "-?" ||
        a == "--usage") {
      // --help/-h are normally consumed by the cvar parser (which prints the
      // cvar list, or nothing when stdout is redirected); --usage and /?
      // reach this tool untouched.
      out->help = true;
    } else if (a == "--list") {
      if (!set_mode(Mode::kList)) {
        return false;
      }
    } else if (a == "--capture") {
      if (!set_mode(Mode::kCapture)) {
        return false;
      }
      take_int();
    } else if (a == "--pose") {
      if (!set_mode(Mode::kPose)) {
        return false;
      }
      take_int();
    } else if (a == "--source") {
      if (!set_mode(Mode::kSource)) {
        return false;
      }
      take_int();
    } else if (a == "--out") {
      if (i + 1 >= args.size()) {
        fmt::print(stderr, "error: --out needs a folder\n");
        return false;
      }
      out->out = xe::to_path(args[++i]);
    } else if (a.rfind("--out=", 0) == 0) {
      out->out = xe::to_path(a.substr(6));
    } else if (a.rfind("--", 0) == 0) {
      // A cvar. "--name value" (no '=') consumes the next token unless it
      // is another option.
      if (a.find('=') == std::string::npos && i + 1 < args.size() &&
          args[i + 1].rfind("--", 0) != 0) {
        ++i;
      }
    } else {
      fmt::print(stderr, "error: unexpected argument '{}'\n", a);
      return false;
    }
  }
  return true;
}

bool WritePng(const std::filesystem::path& path, int width, int height,
              int components, const void* data, int stride) {
  FILE* file = xe::filesystem::OpenFile(path, "wb");
  if (!file) {
    fmt::print(stderr, "error: cannot write {}\n", xe::path_to_utf8(path));
    return false;
  }
  auto write = [](void* context, void* bytes, int size) {
    fwrite(bytes, 1, static_cast<size_t>(size), static_cast<FILE*>(context));
  };
  int ok = stbi_write_png_to_func(write, file, width, height, components, data,
                                  stride);
  fclose(file);
  if (!ok) {
    fmt::print(stderr, "error: PNG encoding failed for {}\n",
               xe::path_to_utf8(path));
    return false;
  }
  return true;
}

bool PrepareOutputFolder(const std::filesystem::path& out) {
  if (out.empty()) {
    return true;
  }
  std::error_code ec;
  std::filesystem::create_directories(out, ec);
  if (ec && !std::filesystem::is_directory(out)) {
    fmt::print(stderr, "error: cannot create folder {}: {}\n",
               xe::path_to_utf8(out), ec.message());
    return false;
  }
  return true;
}

std::filesystem::path OutputPath(const std::filesystem::path& out,
                                 const std::string& name) {
  return out.empty() ? xe::to_path(name) : out / name;
}

CameraCapture::Options CameraOptionsFromCvars() {
  CameraCapture::Options options;
  options.device = cvars::nui_camera;
  options.width =
      static_cast<uint32_t>(std::max(cvars::nui_capture_width, 160));
  options.height =
      static_cast<uint32_t>(std::max(cvars::nui_capture_height, 120));
  options.fps = static_cast<uint32_t>(std::max(cvars::nui_capture_fps, 1));
  return options;
}

// Opens the camera selected by the cvars; prints the reason on failure.
std::unique_ptr<CameraCapture> OpenCamera() {
  auto camera = CameraCapture::Create();
  if (!camera) {
    fmt::print(stderr,
               "error: camera capture is not available on this "
               "platform\n");
    return nullptr;
  }
  std::string error;
  if (!camera->Open(CameraOptionsFromCvars(), &error)) {
    fmt::print(
        stderr, "error: cannot open camera '{}': {}\n",
        cvars::nui_camera.empty() ? std::string("<first>") : cvars::nui_camera,
        error);
    return nullptr;
  }
  fmt::print("camera: {} ({}x{} @ {:.1f} fps)\n", camera->device_name(),
             camera->width(), camera->height(), camera->fps());
  return camera;
}

std::filesystem::path ModelDirFromCvars() {
  if (!cvars::nui_model_path.empty()) {
    return cvars::nui_model_path;
  }
  return xe::filesystem::GetExecutableFolder() / "nui" / "models";
}

std::unique_ptr<PoseEstimator> CreateEstimator() {
  PoseEstimator::Options options;
  options.model_dir = ModelDirFromCvars();
  options.quality = cvars::nui_model_quality;
  options.execution_provider = cvars::nui_execution_provider;
  options.threads = cvars::nui_inference_threads;
  options.max_persons =
      static_cast<uint32_t>(std::clamp(cvars::nui_max_players, 1, 2));
  options.want_segmentation = cvars::nui_segmentation;
  std::string error;
  auto estimator = PoseEstimator::Create(options, &error);
  if (!estimator) {
    fmt::print(stderr,
               "error: pose estimator unavailable: {}\n"
               "  models:  {}\n"
               "  runtime: {}\n"
               "  Run tools/nui/setup_nui.py to download the runtime and "
               "models.\n",
               error, xe::path_to_utf8(options.model_dir),
               cvars::nui_runtime_path.empty()
                   ? xe::path_to_utf8(xe::filesystem::GetExecutableFolder() /
                                      "nui" / "runtime")
                   : xe::path_to_utf8(cvars::nui_runtime_path));
    return nullptr;
  }
  fmt::print("estimator: {} on {} (models in {})\n", estimator->model_name(),
             estimator->backend_name(), xe::path_to_utf8(options.model_dir));
  return estimator;
}

void FlipHorizontal(CameraFrame* frame) {
  if (!frame->width || !frame->height) {
    return;
  }
  const size_t stride = frame->stride ? frame->stride : frame->width * 4;
  for (uint32_t y = 0; y < frame->height; ++y) {
    uint32_t* row =
        reinterpret_cast<uint32_t*>(frame->rgba.data() + y * stride);
    std::reverse(row, row + frame->width);
  }
}

double MeanBrightness(const CameraFrame& frame) {
  if (!frame.width || !frame.height) {
    return 0.0;
  }
  const size_t stride = frame.stride ? frame.stride : frame.width * 4;
  uint64_t sum = 0;
  for (uint32_t y = 0; y < frame.height; ++y) {
    const uint8_t* row = frame.rgba.data() + y * stride;
    for (uint32_t x = 0; x < frame.width; ++x) {
      sum += row[x * 4 + 0] + row[x * 4 + 1] + row[x * 4 + 2];
    }
  }
  return static_cast<double>(sum) /
         (3.0 * static_cast<double>(frame.width) * frame.height);
}

// Simple RGBA drawing helpers for the pose images.
void PutPixel(CameraFrame* frame, int x, int y, uint32_t rgb) {
  if (x < 0 || y < 0 || x >= static_cast<int>(frame->width) ||
      y >= static_cast<int>(frame->height)) {
    return;
  }
  uint8_t* px = frame->rgba.data() + static_cast<size_t>(y) * frame->stride +
                static_cast<size_t>(x) * 4;
  px[0] = static_cast<uint8_t>(rgb >> 16);
  px[1] = static_cast<uint8_t>(rgb >> 8);
  px[2] = static_cast<uint8_t>(rgb);
  px[3] = 0xFF;
}

void DrawSquare(CameraFrame* frame, int cx, int cy, int half, uint32_t rgb) {
  for (int y = cy - half; y <= cy + half; ++y) {
    for (int x = cx - half; x <= cx + half; ++x) {
      PutPixel(frame, x, y, rgb);
    }
  }
}

void DrawLine(CameraFrame* frame, int x0, int y0, int x1, int y1,
              uint32_t rgb) {
  int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  for (int guard = 0; guard < 10000; ++guard) {
    PutPixel(frame, x0, y0, rgb);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

// BlazePose skeleton connections.
constexpr uint8_t kPoseBones[][2] = {
    {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}, {11, 23}, {12, 24},
    {23, 24}, {23, 25}, {25, 27}, {24, 26}, {26, 28}, {27, 29}, {29, 31},
    {27, 31}, {28, 30}, {30, 32}, {28, 32}, {15, 17}, {15, 19}, {15, 21},
    {17, 19}, {16, 18}, {16, 20}, {16, 22}, {18, 20}, {0, 1},   {1, 2},
    {2, 3},   {3, 7},   {0, 4},   {4, 5},   {5, 6},   {6, 8},   {9, 10},
};

constexpr uint32_t kPersonColors[] = {0x00FF40, 0xFF4040, 0x4080FF,
                                      0xFFFF00, 0xFF00FF, 0x00FFFF};

void DrawPose(CameraFrame* frame, const PoseResult& pose, uint32_t rgb) {
  const float w = static_cast<float>(frame->width);
  const float h = static_cast<float>(frame->height);
  auto px = [&](uint32_t i) {
    return static_cast<int>(std::lround(pose.landmarks[i].x * w));
  };
  auto py = [&](uint32_t i) {
    return static_cast<int>(std::lround(pose.landmarks[i].y * h));
  };
  for (const auto& bone : kPoseBones) {
    const PoseLandmark& a = pose.landmarks[bone[0]];
    const PoseLandmark& b = pose.landmarks[bone[1]];
    if (a.visibility < 0.3f || b.visibility < 0.3f) {
      continue;
    }
    DrawLine(frame, px(bone[0]), py(bone[0]), px(bone[1]), py(bone[1]), rgb);
  }
  for (uint32_t i = 0; i < kPoseLandmarkCount; ++i) {
    const PoseLandmark& lm = pose.landmarks[i];
    const uint32_t color = lm.visibility >= 0.65f  ? rgb
                           : lm.visibility >= 0.3f ? 0xFFFFFF
                                                   : 0x808080;
    DrawSquare(frame, px(i), py(i), 2, color);
  }
  // Region box.
  const int rx0 =
      static_cast<int>((pose.region.center_x - pose.region.width * 0.5f) * w);
  const int rx1 =
      static_cast<int>((pose.region.center_x + pose.region.width * 0.5f) * w);
  const int ry0 =
      static_cast<int>((pose.region.center_y - pose.region.height * 0.5f) * h);
  const int ry1 =
      static_cast<int>((pose.region.center_y + pose.region.height * 0.5f) * h);
  DrawLine(frame, rx0, ry0, rx1, ry0, 0x404040);
  DrawLine(frame, rx1, ry0, rx1, ry1, 0x404040);
  DrawLine(frame, rx1, ry1, rx0, ry1, 0x404040);
  DrawLine(frame, rx0, ry1, rx0, ry0, 0x404040);
}

int RunList() {
  auto devices = CameraCapture::EnumerateDevices();
  if (devices.empty()) {
    fmt::print("no cameras found\n");
    return kExitOk;
  }
  for (size_t i = 0; i < devices.size(); ++i) {
    fmt::print("[{}] {}\n    {}\n", i, devices[i].name, devices[i].id);
  }
  return kExitOk;
}

int RunCapture(const DemoArgs& args) {
  const int frame_count = args.count > 0 ? args.count : 30;
  auto camera = OpenCamera();
  if (!camera) {
    return kExitCamera;
  }
  if (!PrepareOutputFolder(args.out)) {
    return kExitUsage;
  }
  CameraFrame frame;
  std::string error;
  int captured = 0;
  int timeouts = 0;
  double brightness_sum = 0.0;
  Clock::time_point first_frame_time;
  Clock::time_point last_frame_time;
  const auto start = Clock::now();
  while (captured < frame_count) {
    if (SecondsSince(start) > 10.0 + frame_count / 5.0) {
      fmt::print(stderr, "error: timed out waiting for frames\n");
      break;
    }
    error.clear();
    if (!camera->ReadFrame(&frame, 1000, &error)) {
      if (!error.empty()) {
        fmt::print(stderr, "error: capture failed: {}\n", error);
        camera->Close();
        return kExitCamera;
      }
      timeouts++;
      continue;
    }
    if (captured == 0) {
      first_frame_time = Clock::now();
      fmt::print("first frame: {}x{} stride {} after {:.0f} ms\n", frame.width,
                 frame.height, frame.stride, SecondsSince(start) * 1000.0);
    }
    last_frame_time = Clock::now();
    const double brightness = MeanBrightness(frame);
    brightness_sum += brightness;
    const std::string name = fmt::format("frame_{:03}.png", captured);
    if (!WritePng(OutputPath(args.out, name), static_cast<int>(frame.width),
                  static_cast<int>(frame.height), 4, frame.rgba.data(),
                  static_cast<int>(frame.stride))) {
      camera->Close();
      return kExitUsage;
    }
    captured++;
  }
  camera->Close();
  if (!captured) {
    fmt::print(stderr, "error: no frames received ({} timeouts)\n", timeouts);
    return kExitCamera;
  }
  const double span =
      std::chrono::duration<double>(last_frame_time - first_frame_time).count();
  const double fps = captured > 1 && span > 0.0 ? (captured - 1) / span : 0.0;
  fmt::print(
      "captured {} frames, {:.1f} fps ({} timeouts), mean brightness "
      "{:.1f}/255\n",
      captured, fps, timeouts, brightness_sum / captured);
  fmt::print("wrote {}\n",
             xe::path_to_utf8(OutputPath(args.out, "frame_000.png")));
  return captured == frame_count ? kExitOk : kExitCamera;
}

int RunPose(const DemoArgs& args) {
  const double seconds = args.count > 0 ? args.count : 10;
  auto camera = OpenCamera();
  if (!camera) {
    return kExitCamera;
  }
  auto estimator = CreateEstimator();
  if (!estimator) {
    camera->Close();
    return kExitModel;
  }
  if (!PrepareOutputFolder(args.out)) {
    return kExitUsage;
  }
  fmt::print("running for {:.0f} s; images: pose_NNN.png\n", seconds);

  CameraFrame frame;
  std::vector<PoseResult> poses;
  std::string error;
  int frames_in_window = 0;
  int total_frames = 0;
  int images = 0;
  double inference_sum_ms = 0.0;
  const auto start = Clock::now();
  auto window_start = start;
  bool have_pose_this_window = false;
  CameraFrame window_frame;
  std::vector<PoseResult> window_poses;
  int exit_code = kExitOk;
  // A lost inference device is recoverable: the estimator backs off, rebuilds
  // and falls back to the CPU, which takes seconds. Exiting on the first
  // failed frame would mean none of that could ever be exercised here.
  constexpr int kFailedFrameBudget = 300;
  int failed_frames = 0;
  bool printed_failure = false;
  auto last_failure_print = start;
  while (SecondsSince(start) < seconds) {
    error.clear();
    if (!camera->ReadFrame(&frame, 1000, &error)) {
      if (!error.empty()) {
        fmt::print(stderr, "error: capture failed: {}\n", error);
        exit_code = kExitCamera;
        break;
      }
      continue;
    }
    if (!frame.stride) {
      frame.stride = frame.width * 4;
    }
    if (!cvars::nui_camera_mirrored) {
      FlipHorizontal(&frame);
    }
    error.clear();
    if (!estimator->Process(frame.rgba.data(), frame.width, frame.height,
                            frame.stride, &poses, &error)) {
      ++failed_frames;
      if (!printed_failure || SecondsSince(last_failure_print) >= 2.0) {
        printed_failure = true;
        last_failure_print = Clock::now();
        fmt::print(stderr, "warning: pose estimation failed: {} ({} frame{})\n",
                   error, failed_frames, failed_frames == 1 ? "" : "s");
      }
      if (estimator->backend_health() ==
              PoseEstimator::BackendHealth::kFailed ||
          failed_frames > kFailedFrameBudget) {
        fmt::print(stderr,
                   "error: pose estimation failed {} frames in a row: "
                   "{}\n",
                   failed_frames, error);
        exit_code = kExitModel;
        break;
      }
      continue;
    }
    failed_frames = 0;
    printed_failure = false;
    frames_in_window++;
    total_frames++;
    inference_sum_ms += estimator->last_inference_ms();
    if (!have_pose_this_window) {
      window_frame = frame;
      window_poses = poses;
      have_pose_this_window = true;
    }

    const double window =
        std::chrono::duration<double>(Clock::now() - window_start).count();
    if (window >= 1.0) {
      fmt::print(
          "[{:5.1f}s] {:.1f} fps, {:.1f} ms/frame, {} on {}, {} "
          "person(s)\n",
          SecondsSince(start), frames_in_window / window,
          inference_sum_ms / std::max(frames_in_window, 1),
          estimator->model_name(), estimator->backend_name(), poses.size());
      if (!poses.empty() && poses[0].valid) {
        const PoseResult& p = poses[0];
        const auto& nose =
            p.landmarks[static_cast<size_t>(PoseLandmarkIndex::kNose)];
        const auto& lw =
            p.landmarks[static_cast<size_t>(PoseLandmarkIndex::kLeftWrist)];
        const auto& rw =
            p.landmarks[static_cast<size_t>(PoseLandmarkIndex::kRightWrist)];
        const auto& lh =
            p.world_landmarks[static_cast<size_t>(PoseLandmarkIndex::kLeftHip)];
        const auto& rh = p.world_landmarks[static_cast<size_t>(
            PoseLandmarkIndex::kRightHip)];
        fmt::print(
            "   score {:.2f}  nose ({:.3f},{:.3f} v{:.2f})  "
            "left_wrist ({:.3f},{:.3f} v{:.2f})  right_wrist "
            "({:.3f},{:.3f} v{:.2f})  world hip z {:.3f}  "
            "region {:.2f}x{:.2f} seg={}\n",
            p.pose_score, nose.x, nose.y, nose.visibility, lw.x, lw.y,
            lw.visibility, rw.x, rw.y, rw.visibility, (lh.z + rh.z) * 0.5f,
            p.region.width, p.region.height, p.has_segmentation ? "yes" : "no");
      }
      // Annotated image of the first frame of this window.
      for (size_t i = 0; i < window_poses.size(); ++i) {
        if (window_poses[i].valid) {
          DrawPose(&window_frame, window_poses[i],
                   kPersonColors[i % std::size(kPersonColors)]);
        }
      }
      const std::string name = fmt::format("pose_{:03}.png", images++);
      WritePng(OutputPath(args.out, name), static_cast<int>(window_frame.width),
               static_cast<int>(window_frame.height), 4,
               window_frame.rgba.data(), static_cast<int>(window_frame.stride));
      window_start = Clock::now();
      frames_in_window = 0;
      inference_sum_ms = 0.0;
      have_pose_this_window = false;
    }
  }
  camera->Close();
  fmt::print("processed {} frames, wrote {} image(s)\n", total_frames, images);
  return exit_code;
}

DeviceState DeviceStateFromCvars() {
  DeviceState state;
  state.init_flags = kInitDepthAndPlayerIndex | kInitColor | kInitSkeleton;
  state.skeleton_tracking = true;
  state.want_player_mask = true;
  state.want_depth = true;
  state.want_color = true;
  state.camera_height_m = static_cast<float>(cvars::nui_camera_height);
  state.camera_pitch_degrees = static_cast<float>(cvars::nui_camera_pitch);
  state.max_tracked_players =
      static_cast<uint32_t>(std::clamp(cvars::nui_max_players, 1, 2));
  return state;
}

const char* JointStateName(JointState state) {
  switch (state) {
    case JointState::kTracked:
      return "T";
    case JointState::kInferred:
      return "I";
    default:
      return "-";
  }
}

std::string FormatJoint(const Skeleton& body, Joint joint) {
  const Vec4& p = body.joints[static_cast<size_t>(joint)];
  return fmt::format(
      "({:+.2f},{:+.2f},{:.2f}){}", p.x, p.y, p.z,
      JointStateName(body.joint_states[static_cast<size_t>(joint)]));
}

std::string FormatQuality(uint32_t flags) {
  std::string s;
  if (flags & kQualityClippedLeft) {
    s += "L";
  }
  if (flags & kQualityClippedRight) {
    s += "R";
  }
  if (flags & kQualityClippedTop) {
    s += "T";
  }
  if (flags & kQualityClippedBottom) {
    s += "B";
  }
  return s.empty() ? "none" : "clipped:" + s;
}

void WriteDepthImage(const SourceFrame& frame,
                     const std::filesystem::path& path) {
  std::vector<uint8_t> rgb(static_cast<size_t>(kDepthWidth) * kDepthHeight * 3);
  for (size_t i = 0; i < static_cast<size_t>(kDepthWidth) * kDepthHeight; ++i) {
    const uint16_t mm = frame.has_depth ? frame.depth_mm[i] : 0;
    uint8_t gray = 0;
    if (mm) {
      // 0.5 m bright .. 4.5 m dark.
      const float t = std::clamp((mm - 500.0f) / 4000.0f, 0.0f, 1.0f);
      gray = static_cast<uint8_t>(255.0f * (1.0f - t) * 0.85f + 30.0f);
    }
    const uint8_t player = frame.has_player_mask ? frame.player_mask[i] : 0;
    uint8_t r = gray, g = gray, b = gray;
    if (player) {
      const uint32_t tint =
          kPersonColors[(player - 1) % std::size(kPersonColors)];
      r = static_cast<uint8_t>((gray + ((tint >> 16) & 0xFF)) / 2);
      g = static_cast<uint8_t>((gray + ((tint >> 8) & 0xFF)) / 2);
      b = static_cast<uint8_t>((gray + (tint & 0xFF)) / 2);
    }
    rgb[i * 3 + 0] = r;
    rgb[i * 3 + 1] = g;
    rgb[i * 3 + 2] = b;
  }
  WritePng(path, kDepthWidth, kDepthHeight, 3, rgb.data(), kDepthWidth * 3);
}

void WriteColorImage(const SourceFrame& frame,
                     const std::filesystem::path& path) {
  std::vector<uint8_t> rgb(static_cast<size_t>(kColorWidth) * kColorHeight * 3);
  for (size_t i = 0; i < static_cast<size_t>(kColorWidth) * kColorHeight; ++i) {
    const uint32_t argb = frame.color_argb[i];
    rgb[i * 3 + 0] = static_cast<uint8_t>(argb >> 16);
    rgb[i * 3 + 1] = static_cast<uint8_t>(argb >> 8);
    rgb[i * 3 + 2] = static_cast<uint8_t>(argb);
  }
  WritePng(path, kColorWidth, kColorHeight, 3, rgb.data(), kColorWidth * 3);
}

// Worst observed wake-up latency of a 1 ms sleep, in milliseconds: ~1 on a
// system with a 1 ms timer period, ~16 with the default 15.6 ms period.
double MeasureSleepGranularityMs() {
  double worst = 0.0;
  for (int i = 0; i < 4; ++i) {
    const auto t0 = Clock::now();
    xe::threading::NanoSleepPrecise(1000000);
    worst = std::max(
        worst,
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  return worst;
}

// Waits for the asynchronous start of the source, printing every status
// change. Returns the settled state (kRunning or kCameraOnly) or, on
// timeout, the state it was stuck in.
WebcamNuiSource::State WaitForSourceStart(WebcamNuiSource* source) {
  // One retry of the camera (3 s interval) plus its own 10 s open timeout.
  constexpr double kStartTimeoutSeconds = 25.0;
  // A camera that reports an error gets one retry before the demo gives up;
  // the emulator itself keeps retrying forever.
  constexpr double kCameraErrorGraceSeconds = 5.0;
  const auto start = Clock::now();
  Clock::time_point first_camera_error;
  bool camera_error_seen = false;
  std::string last_status;
  SourceStats stats;
  while (true) {
    source->GetStats(&stats);
    if (stats.status != last_status) {
      fmt::print("source: {}\n", stats.status);
      last_status = stats.status;
    }
    const WebcamNuiSource::State state = source->state();
    if (state == WebcamNuiSource::State::kRunning ||
        state == WebcamNuiSource::State::kCameraOnly ||
        state == WebcamNuiSource::State::kStopped) {
      return state;
    }
    if (stats.status.rfind("camera error", 0) == 0) {
      if (!camera_error_seen) {
        camera_error_seen = true;
        first_camera_error = Clock::now();
      } else if (SecondsSince(first_camera_error) > kCameraErrorGraceSeconds) {
        return state;
      }
    }
    if (SecondsSince(start) > kStartTimeoutSeconds) {
      return state;
    }
    xe::threading::Sleep(std::chrono::milliseconds(50));
  }
}

int RunSource(const DemoArgs& args) {
  const double seconds = args.count > 0 ? args.count : 10;
  if (!PrepareOutputFolder(args.out)) {
    return kExitUsage;
  }
  WebcamNuiSource source;
  const DeviceState state = DeviceStateFromCvars();
  const auto start_call = Clock::now();
  if (!source.Start(state)) {
    SourceStats stats;
    source.GetStats(&stats);
    fmt::print(stderr, "error: source failed to start: {}\n", stats.status);
    return kExitCamera;
  }
  fmt::print("Start() returned after {:.1f} ms\n",
             SecondsSince(start_call) * 1000.0);
  SourceStats stats;
  const WebcamNuiSource::State settled = WaitForSourceStart(&source);
  source.GetStats(&stats);
  if (settled != WebcamNuiSource::State::kRunning &&
      settled != WebcamNuiSource::State::kCameraOnly) {
    fmt::print(stderr, "error: the source did not start ({:.1f} s): {}\n",
               SecondsSince(start_call), stats.status);
    source.Stop();
    return settled == WebcamNuiSource::State::kLoadingModels ? kExitModel
                                                             : kExitCamera;
  }
  fmt::print("source ready after {:.1f} s: {}\n", SecondsSince(start_call),
             stats.status);
  const bool model_missing = settled == WebcamNuiSource::State::kCameraOnly;
  if (model_missing) {
    fmt::print(stderr,
               "warning: running camera-only, no bodies will be "
               "reported\n");
  }
  fmt::print("running for {:.0f} s; images: depth_NNN.png, color_NNN.png\n",
             seconds);

  // AcquireLatest only hands out the newest frame, so the loop must look
  // more often than the source produces. A plain Sleep wakes at the OS timer
  // period (~16 ms by default), which at 30 fps skips every third or fourth
  // frame; instead the loop sleeps 1 ms at a time while the next frame is
  // comfortably far away and yields in a spin inside the last timer period
  // before it is due. Sequence gaps are still counted so the reported rate
  // is the source's even if a frame is missed.
  const double sleep_granularity_ms = MeasureSleepGranularityMs();
  const auto spin_margin = std::chrono::microseconds(
      static_cast<int64_t>((sleep_granularity_ms + 1.0) * 1000.0));
  auto frame_interval = std::chrono::microseconds(1000000 / kFrameRateHz);

  const auto start = Clock::now();
  auto next_report = start + std::chrono::seconds(1);
  auto last_arrival = start;
  uint64_t last_sequence = 0;
  uint64_t new_frames = 0;       // frames this loop received
  uint64_t produced_window = 0;  // frames the source published (by sequence)
  uint64_t received_window = 0;
  uint32_t max_bodies = 0;
  double inference_sum_ms = 0.0;  // of the source's per-frame EWMA
  int images = 0;
  std::shared_ptr<const SourceFrame> latest;
  while (SecondsSince(start) < seconds) {
    auto frame = source.AcquireLatest(last_sequence);
    if (frame) {
      const auto now = Clock::now();
      if (last_sequence) {
        const auto delta =
            std::chrono::duration_cast<std::chrono::microseconds>(now -
                                                                  last_arrival);
        if (delta > std::chrono::milliseconds(5) &&
            delta < std::chrono::milliseconds(200)) {
          frame_interval = (frame_interval * 7 + delta) / 8;
        }
      }
      last_arrival = now;
      latest = frame;
      produced_window += frame->sequence - last_sequence;
      last_sequence = frame->sequence;
      new_frames++;
      received_window++;
      max_bodies = std::max(max_bodies, frame->body_count);
      source.GetStats(&stats);
      inference_sum_ms += stats.inference_ms;
    } else {
      const auto now = Clock::now();
      const auto expected = last_arrival + frame_interval;
      if (now + spin_margin < expected || now > expected + frame_interval) {
        // Far from the next frame, or the source has gone quiet: cheap.
        xe::threading::NanoSleepPrecise(1000000);
      } else {
        xe::threading::MaybeYield();
      }
    }
    if (Clock::now() < next_report) {
      continue;
    }
    next_report += std::chrono::seconds(1);
    source.GetStats(&stats);
    fmt::print(
        "[{:5.1f}s] source {} fps ({} received; capture {:.1f} fps, {:.1f} ms "
        "inference, {} produced, {} dropped) {}\n",
        SecondsSince(start), produced_window, received_window,
        stats.capture_fps, stats.inference_ms, stats.frames_produced,
        stats.frames_dropped, stats.status);
    produced_window = 0;
    received_window = 0;
    if (!latest) {
      fmt::print("   no frame yet\n");
      continue;
    }
    const SourceFrame& f = *latest;
    fmt::print(
        "   seq {} bodies {} floor ({:.2f},{:.2f},{:.2f},{:.2f}) "
        "mask={} depth={} color={}\n",
        f.sequence, f.body_count, f.floor_clip_plane.x, f.floor_clip_plane.y,
        f.floor_clip_plane.z, f.floor_clip_plane.w,
        f.has_player_mask ? "y" : "n", f.has_depth ? "y" : "n",
        f.has_color ? "y" : "n");
    for (uint32_t i = 0; i < f.body_count && i < kMaxSkeletons; ++i) {
      const Skeleton& body = f.bodies[i];
      const char* state_name = body.state == SkeletonState::kTracked ? "tracked"
                               : body.state == SkeletonState::kPositionOnly
                                   ? "position"
                                   : "none";
      fmt::print("   body {} id {} {} pos ({:+.2f},{:+.2f},{:.2f}) {}\n", i + 1,
                 body.tracking_id, state_name, body.position.x, body.position.y,
                 body.position.z, FormatQuality(body.quality_flags));
      if (body.state == SkeletonState::kTracked) {
        fmt::print("      hip {} head {} handL {} handR {}\n",
                   FormatJoint(body, Joint::kHipCenter),
                   FormatJoint(body, Joint::kHead),
                   FormatJoint(body, Joint::kHandLeft),
                   FormatJoint(body, Joint::kHandRight));
      }
    }
    if (f.has_player_mask) {
      size_t covered = 0;
      for (uint8_t p : f.player_mask) {
        covered += p ? 1 : 0;
      }
      fmt::print("   player mask covers {:.1f}% of the image\n",
                 100.0 * covered / f.player_mask.size());
    }
    if (f.has_depth || f.has_player_mask) {
      WriteDepthImage(
          f, OutputPath(args.out, fmt::format("depth_{:03}.png", images)));
    }
    if (f.has_color) {
      WriteColorImage(
          f, OutputPath(args.out, fmt::format("color_{:03}.png", images)));
    }
    images++;
  }
  source.GetStats(&stats);
  source.Stop();
  const uint64_t produced = stats.frames_produced;
  fmt::print(
      "summary: {} frames produced, {} received ({} missed by this poll, "
      "sleep granularity {:.1f} ms), {} person(s) max, mean inference {:.1f} "
      "ms, {} dropped by the source, wrote {} image pair(s)\n",
      produced, new_frames, produced > new_frames ? produced - new_frames : 0,
      sleep_granularity_ms, max_bodies,
      new_frames ? inference_sum_ms / static_cast<double>(new_frames) : 0.0,
      stats.frames_dropped, images);
  if (model_missing) {
    return kExitModel;
  }
  if (!new_frames) {
    fmt::print(stderr, "error: the source never produced a frame\n");
    return kExitCamera;
  }
  return kExitOk;
}

}  // namespace

int nui_webcam_demo_main(const std::vector<std::string>& args) {
  DemoArgs demo_args;
  if (!ParseDemoArgs(args, &demo_args)) {
    PrintUsage(args.empty() ? "xenia-nui-webcam-demo" : args[0]);
    return kExitUsage;
  }
  if (demo_args.help || demo_args.mode == Mode::kNone) {
    PrintUsage(args.empty() ? "xenia-nui-webcam-demo" : args[0]);
    return demo_args.help ? kExitOk : kExitUsage;
  }
  switch (demo_args.mode) {
    case Mode::kList:
      return RunList();
    case Mode::kCapture:
      return RunCapture(demo_args);
    case Mode::kPose:
      return RunPose(demo_args);
    case Mode::kSource:
      return RunSource(demo_args);
    default:
      return kExitUsage;
  }
}

}  // namespace nui
}  // namespace xe

XE_DEFINE_CONSOLE_APP("xenia-nui-webcam-demo", xe::nui::nui_webcam_demo_main,
                      "--list | --capture [N] | --pose [SECONDS] | --source "
                      "[SECONDS]  [--out DIR] [--nui_*=VALUE]");
