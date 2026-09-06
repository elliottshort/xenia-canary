/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_NUI_FLAGS_H_
#define XENIA_NUI_NUI_FLAGS_H_

#include "xenia/base/cvar.h"

DECLARE_bool(nui);
DECLARE_string(nui_source);
DECLARE_bool(nui_hle);
DECLARE_bool(nui_trace);
DECLARE_bool(nui_log_stats);
DECLARE_double(nui_camera_height);
DECLARE_double(nui_camera_pitch);
DECLARE_string(nui_tilt_mode);
DECLARE_int32(nui_max_players);
DECLARE_bool(nui_smoothing_override);
DECLARE_double(nui_smoothing);
DECLARE_double(nui_correction);
DECLARE_double(nui_prediction);
DECLARE_double(nui_jitter_radius);
DECLARE_double(nui_max_deviation_radius);
DECLARE_path(nui_record_path);
DECLARE_path(nui_playback_path);
DECLARE_bool(nui_playback_loop);
DECLARE_bool(nui_playback_realtime);
DECLARE_bool(nui_auto_bind_user0);
DECLARE_string(nui_camera);
DECLARE_int32(nui_capture_width);
DECLARE_int32(nui_capture_height);
DECLARE_int32(nui_capture_fps);
DECLARE_double(nui_camera_hfov);
DECLARE_bool(nui_camera_mirrored);
DECLARE_double(nui_user_scale);
DECLARE_string(nui_model_quality);
DECLARE_string(nui_execution_provider);
DECLARE_int32(nui_inference_threads);
DECLARE_path(nui_model_path);
DECLARE_path(nui_runtime_path);
DECLARE_bool(nui_segmentation);
DECLARE_path(nui_dump_module_image);

#endif  // XENIA_NUI_NUI_FLAGS_H_
