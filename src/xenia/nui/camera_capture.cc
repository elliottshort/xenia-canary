/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/camera_capture.h"

#include "xenia/base/platform.h"

namespace xe {
namespace nui {

std::vector<CameraDeviceInfo> CameraCapture::EnumerateDevices() {
#if XE_PLATFORM_WIN32
  return EnumerateCameraDevicesWin();
#else
  return {};
#endif
}

std::unique_ptr<CameraCapture> CameraCapture::Create() {
#if XE_PLATFORM_WIN32
  return CreateCameraCaptureWin();
#else
  return nullptr;
#endif
}

}  // namespace nui
}  // namespace xe
