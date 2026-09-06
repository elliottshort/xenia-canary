/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_CAMERA_CAPTURE_H_
#define XENIA_NUI_CAMERA_CAPTURE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xenia/base/platform.h"

namespace xe {
namespace nui {

struct CameraDeviceInfo {
  std::string name;  // human readable
  std::string id;    // platform device id (symbolic link / path)
};

// One captured frame, converted to 8-bit RGBA, top-down rows.
struct CameraFrame {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride = 0;  // bytes per row in |rgba|
  std::vector<uint8_t> rgba;
  // Host steady-clock time of the capture in microseconds.
  uint64_t timestamp_us = 0;
  uint64_t sequence = 0;
};

// A webcam. Implementations exist per platform (Media Foundation on
// Windows); Create() returns nullptr where none is available.
class CameraCapture {
 public:
  struct Options {
    // Device index ("0", "1", ...) or a case-insensitive substring of the
    // device name; empty selects the first camera.
    std::string device;
    uint32_t width = 640;
    uint32_t height = 480;
    uint32_t fps = 30;
  };

  static std::vector<CameraDeviceInfo> EnumerateDevices();
  static std::unique_ptr<CameraCapture> Create();

  virtual ~CameraCapture() = default;

  // Opens the device. The actual mode may differ from the request (query
  // width()/height()/fps()). |out_error| receives a human readable reason
  // on failure (missing device, access denied by privacy settings, ...).
  virtual bool Open(const Options& options, std::string* out_error) = 0;
  virtual void Close() = 0;
  virtual bool is_open() const = 0;

  // Blocks until the next frame arrives or |timeout_ms| passes. Returns
  // false on timeout (|out_error| empty) or on a device error.
  virtual bool ReadFrame(CameraFrame* frame, uint32_t timeout_ms,
                         std::string* out_error) = 0;

  virtual std::string device_name() const = 0;
  virtual uint32_t width() const = 0;
  virtual uint32_t height() const = 0;
  virtual double fps() const = 0;
};

#if XE_PLATFORM_WIN32
std::vector<CameraDeviceInfo> EnumerateCameraDevicesWin();
std::unique_ptr<CameraCapture> CreateCameraCaptureWin();
#endif

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_CAMERA_CAPTURE_H_
