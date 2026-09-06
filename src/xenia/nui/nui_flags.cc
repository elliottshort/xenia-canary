/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/nui_flags.h"

DEFINE_bool(nui, false,
            "Emulate a Kinect (NUI) sensor. Titles see a connected, ready "
            "sensor whose data comes from the source selected with "
            "nui_source.",
            "NUI");
DEFINE_string(nui_source, "virtual",
              "Where Kinect data comes from:\n"
              " virtual  - a standing player driven by the gamepad/keyboard\n"
              " playback - a recording made with nui_record_path\n"
              " webcam   - a camera with host-side pose estimation\n"
              " none     - a sensor that never sees anybody",
              "NUI");
DEFINE_bool(nui_hle, true,
            "Replace the NUI runtime entry points statically linked into "
            "titles with host implementations (requires a signature for the "
            "title's NUI library version).",
            "NUI");
DEFINE_bool(nui_trace, false,
            "Log every call into the emulated NUI runtime with its arguments.",
            "NUI");
DEFINE_bool(nui_log_stats, false,
            "Log Kinect emulation statistics every 5 seconds.", "NUI");
DEFINE_double(nui_camera_height, 1.0,
              "Height of the (virtual) sensor above the floor in metres. "
              "0 means unknown: no floor plane is reported.",
              "NUI");
DEFINE_double(nui_camera_pitch, 0.0,
              "Physical pitch of the webcam in degrees, positive looking up.",
              "NUI");
DEFINE_string(nui_tilt_mode, "simulate",
              "How title requests to tilt the sensor are handled:\n"
              " simulate - rotate the virtual sensor (affects skeletons, "
              "images and the floor plane)\n"
              " ignore   - only report the requested angle back",
              "NUI");
DEFINE_int32(nui_max_players, 2, "Number of fully tracked skeletons (1 or 2).",
             "NUI");
DEFINE_bool(nui_smoothing_override, false,
            "Ignore the smoothing parameters titles pass to "
            "NuiTransformSmooth and use the nui_smoothing_* values instead.",
            "NUI");
DEFINE_double(nui_smoothing, 0.5, "Override: smoothing [0..1].", "NUI");
DEFINE_double(nui_correction, 0.5, "Override: correction [0..1].", "NUI");
DEFINE_double(nui_prediction, 0.5, "Override: prediction (frames).", "NUI");
DEFINE_double(nui_jitter_radius, 0.05, "Override: jitter radius (metres).",
              "NUI");
DEFINE_double(nui_max_deviation_radius, 0.04,
              "Override: maximum deviation radius (metres).", "NUI");
DEFINE_path(nui_record_path, "",
            "Record every frame the emulated sensor publishes (skeletons, "
            "player mask, depth and colour when the source produces them) to "
            "this .nuirec file. Can be set or cleared while running; an "
            "existing file is overwritten. Play back with "
            "nui_source=playback.",
            "NUI");
DEFINE_path(nui_playback_path, "",
            "Recording (.nuirec, made with nui_record_path) to play back "
            "when nui_source is playback.",
            "NUI");
DEFINE_bool(nui_playback_loop, true,
            "Restart the playback recording when it ends (false: the last "
            "frame stays).",
            "NUI");
DEFINE_bool(nui_playback_realtime, true,
            "Play recordings by their timestamps, skipping frames if needed "
            "(false: one recorded frame per 30 Hz tick, deterministic).",
            "NUI");
DEFINE_bool(nui_auto_bind_user0, true,
            "Bind the first tracked player to the signed-in user 0.", "NUI");

DEFINE_string(nui_camera, "",
              "Webcam to use: a device index (0, 1, ...) or part of its "
              "name; empty = the first camera.",
              "NUI");
DEFINE_int32(nui_capture_width, 640, "Requested capture width.", "NUI");
DEFINE_int32(nui_capture_height, 480, "Requested capture height.", "NUI");
DEFINE_int32(nui_capture_fps, 30, "Requested capture frame rate.", "NUI");
DEFINE_double(nui_camera_hfov, 70.0,
              "Horizontal field of view of the webcam in degrees (used to "
              "estimate distances; typical laptop/USB webcams are 60-78).",
              "NUI");
DEFINE_bool(nui_camera_mirrored, false,
            "The camera driver already delivers a mirrored image (raise "
            "your right hand: it should appear on the right of the preview).",
            "NUI");
DEFINE_double(nui_user_scale, 1.0,
              "Multiplies the estimated distance to the player (calibrate "
              "so that standing 2 m away reads as 2 m).",
              "NUI");
DEFINE_string(nui_model_quality, "auto",
              "Pose model: lite | full | heavy | auto (full on GPU, lite on "
              "CPU).",
              "NUI");
DEFINE_string(nui_execution_provider, "auto",
              "ONNX Runtime execution provider: auto | dml | cpu.", "NUI");
DEFINE_int32(nui_inference_threads, 0, "CPU threads for inference (0 = auto).",
             "NUI");
DEFINE_path(nui_model_path, "",
            "Folder with the pose models (default <storage>/nui/models).",
            "NUI");
DEFINE_path(nui_runtime_path, "",
            "Folder with onnxruntime.dll and DirectML.dll (default "
            "<storage>/nui/runtime, then the emulator folder).",
            "NUI");
DEFINE_bool(nui_segmentation, true,
            "Use the pose model's person segmentation for the player "
            "silhouette in depth images.",
            "NUI");
DEFINE_path(nui_dump_module_image, "",
            "Developer: write the loaded image of every executable that links "
            "a NUI library into this folder (<module>_<base>.bin plus a .json "
            "sidecar with code ranges and imports) for making signatures.",
            "NUI");
