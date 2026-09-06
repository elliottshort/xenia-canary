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

#if XE_PLATFORM_WIN32

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "xenia/base/logging.h"
#include "xenia/base/platform_win.h"
#include "xenia/base/threading.h"
#include "xenia/nui/nui_types.h"

// Media Foundation headers must come after platform_win.h (windows.h with
// WIN32_LEAN_AND_MEAN / NOMINMAX already applied).
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>

// Media Foundation webcam capture.
//
// Threading model: one dedicated capture thread ("NUI Camera") owns every
// Media Foundation object for the lifetime of an Open()/Close() pair. It
// initialises COM (MTA) and Media Foundation, enumerates and activates the
// device, negotiates the media type, runs the synchronous ReadSample loop,
// converts each sample to top-down RGBA, publishes it under a mutex and
// finally tears everything down in reverse order (reader, source shutdown,
// activate shutdown, MFShutdown, CoUninitialize) on that same thread.
//
// Bounded waits: IMFSourceReader::ReadSample in synchronous mode has no
// timeout and blocks for about one frame interval (~33 ms at 30 fps, up to
// ~300 ms for the very first call while the camera starts streaming). By
// keeping that call on the capture thread, ReadFrame() itself is only ever a
// condition-variable wait bounded exactly by |timeout_ms|: it returns the
// newest converted frame when one arrives, false with an empty error on
// timeout, or false with the device error. Open() waits at most
// kOpenTimeoutMs for the first frame (a busy camera surfaces its error on the
// first ReadSample, ~250 ms in; a healthy one delivers ~500 ms in). Close()
// sets a stop flag and joins; the in-flight ReadSample returns within one
// frame interval. If a driver ever misbehaves and the thread has not exited
// after kCloseJoinTimeoutMs, IMFSourceReader::Flush is called from the
// closing thread to cancel the pending request before joining for good.
//
// Pixel orientation: the video processor delivers RGB32 (B,G,R,X in memory)
// through IMF2DBuffer. Lock2D's pitch sign is the single source of truth for
// the row order (its scanline0 is defined as the TOP image row whatever the
// sign), so walking scanline0 + y * pitch yields top-down rows for both
// signs. MF_MT_DEFAULT_STRIDE and MFGetStrideForBitmapInfoHeader are not
// trusted for orientation: on the reference camera they are absent /
// negative while the delivered buffers are top-down with a positive pitch.
// The frames are the raw (non-mirrored) camera view; the Kinect mirror flip
// is applied once by the pipeline, not here.

namespace xe {
namespace nui {
namespace {

using Microsoft::WRL::ComPtr;

// MF_SOURCE_READER_FIRST_VIDEO_STREAM is a negative enumerator; cast once.
constexpr DWORD kVideoStream =
    static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM);

// Upper bound for the first frame after Open() (observed ~520 ms on a warm
// camera; USB cameras waking from power-save or re-enumerating can take
// several seconds, and some withhold frames until auto-exposure and white
// balance have converged, which in a dark room takes 2-4 s). This is the
// only first-frame wait: ReadFrame() is bounded by the caller's timeout_ms
// and reports a plain timeout (no error) until the first sample lands, so a
// slow camera never turns into a device error. Device errors still surface
// immediately through error_.
constexpr uint32_t kOpenTimeoutMs = 10000;
// Time Close() gives the capture thread to leave ReadSample on its own before
// forcing the reader to flush.
constexpr uint32_t kCloseJoinTimeoutMs = 1500;
// Sample-time correction is only applied when the sample is at most this old
// (guards against drivers reporting bogus timestamps).
constexpr uint64_t kMaxSampleAgeUs = 1000000;

std::string ToUtf8(const wchar_t* wide, size_t length) {
  if (!wide || !length) {
    return std::string();
  }
  int needed = WideCharToMultiByte(CP_UTF8, 0, wide, static_cast<int>(length),
                                   nullptr, 0, nullptr, nullptr);
  if (needed <= 0) {
    return std::string();
  }
  std::string result(static_cast<size_t>(needed), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide, static_cast<int>(length), result.data(),
                      needed, nullptr, nullptr);
  return result;
}

std::string ToLower(std::string text) {
  for (char& c : text) {
    c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
  }
  return text;
}

uint64_t SteadyNowMicroseconds() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Human readable subtype name: FOURCC for video formats, else a GUID.
std::string SubtypeName(const GUID& subtype) {
  if (subtype == MFVideoFormat_RGB32) {
    return "RGB32";
  }
  if (subtype == MFVideoFormat_ARGB32) {
    return "ARGB32";
  }
  if (subtype == MFVideoFormat_RGB24) {
    return "RGB24";
  }
  // Every other MF video subtype is a FOURCC in Data1.
  char fourcc[5] = {static_cast<char>(subtype.Data1 & 0xFF),
                    static_cast<char>((subtype.Data1 >> 8) & 0xFF),
                    static_cast<char>((subtype.Data1 >> 16) & 0xFF),
                    static_cast<char>((subtype.Data1 >> 24) & 0xFF), 0};
  for (char& c : fourcc) {
    if (c && (c < 0x20 || c > 0x7E)) {
      c = '?';
    }
  }
  return std::string(fourcc);
}

// Maps the HRESULTs a capture pipeline commonly meets to user messages.
// FormatMessage knows only the generic codes (the MF_E_* family is absent
// from the system table), so the code is always appended.
std::string DescribeHresult(HRESULT hr) {
  char buffer[512];
  const char* reason = nullptr;
  switch (static_cast<uint32_t>(hr)) {
    case static_cast<uint32_t>(E_ACCESSDENIED):
      reason =
          "camera access is blocked by Windows privacy settings (Settings > "
          "Privacy & security > Camera > allow desktop apps)";
      break;
    case static_cast<uint32_t>(MF_E_HW_MFT_FAILED_START_STREAMING):
      reason =
          "the camera is in use by another application (close other apps "
          "using it)";
      break;
    case static_cast<uint32_t>(MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED):
      reason = "the camera was disconnected";
      break;
    case static_cast<uint32_t>(MF_E_INVALIDMEDIATYPE):
      reason = "the camera does not support the requested format";
      break;
    case static_cast<uint32_t>(MF_E_TOPO_CODEC_NOT_FOUND):
      reason =
          "no decoder is available for the camera format (install the Media "
          "Feature Pack on N editions of Windows)";
      break;
    case static_cast<uint32_t>(MF_E_NOT_FOUND):
      reason = "camera not found";
      break;
    case static_cast<uint32_t>(MF_E_INVALIDREQUEST):
      reason = "invalid request (camera reader state)";
      break;
    case static_cast<uint32_t>(MF_E_INVALIDSTREAMNUMBER):
      reason = "the device has no video stream";
      break;
    case static_cast<uint32_t>(E_OUTOFMEMORY):
      reason = "out of memory";
      break;
    default:
      break;
  }
  if (reason) {
    snprintf(buffer, sizeof(buffer), "%s (HRESULT 0x%08X)", reason,
             static_cast<unsigned>(hr));
    return buffer;
  }
  wchar_t* system_message = nullptr;
  DWORD n = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, static_cast<DWORD>(hr), 0,
      reinterpret_cast<LPWSTR>(&system_message), 0, nullptr);
  std::string text;
  if (n && system_message) {
    while (n &&
           (system_message[n - 1] == L'\r' || system_message[n - 1] == L'\n' ||
            system_message[n - 1] == L' ')) {
      --n;
    }
    text = ToUtf8(system_message, n);
    LocalFree(system_message);
  }
  if (text.empty()) {
    text = "unknown error";
  }
  snprintf(buffer, sizeof(buffer), "%s (HRESULT 0x%08X)", text.c_str(),
           static_cast<unsigned>(hr));
  return buffer;
}

// COM apartment for the calling thread. RPC_E_CHANGED_MODE means the thread
// is already initialised in another mode (e.g. STA UI thread): that is usable
// but must not be balanced with CoUninitialize.
class ScopedComApartment {
 public:
  ScopedComApartment() : hr_(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) {}
  ~ScopedComApartment() {
    if (SUCCEEDED(hr_)) {
      CoUninitialize();
    }
  }
  bool ok() const { return SUCCEEDED(hr_) || hr_ == RPC_E_CHANGED_MODE; }
  HRESULT hr() const { return hr_; }

 private:
  HRESULT hr_;
};

// Process-wide, ref-counted Media Foundation startup.
class ScopedMediaFoundation {
 public:
  ScopedMediaFoundation() : hr_(MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET)) {}
  ~ScopedMediaFoundation() {
    if (SUCCEEDED(hr_)) {
      MFShutdown();
    }
  }
  bool ok() const { return SUCCEEDED(hr_); }
  HRESULT hr() const { return hr_; }

 private:
  HRESULT hr_;
};

struct DeviceEntry {
  std::string name;
  std::string symbolic_link;
  ComPtr<IMFActivate> activate;
};

// Lists video capture devices. Zero devices is S_OK with an empty list.
HRESULT EnumerateDeviceEntries(std::vector<DeviceEntry>* out) {
  ComPtr<IMFAttributes> attributes;
  HRESULT hr = MFCreateAttributes(&attributes, 1);
  if (FAILED(hr)) {
    return hr;
  }
  hr = attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                           MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
  if (FAILED(hr)) {
    return hr;
  }
  IMFActivate** devices = nullptr;
  UINT32 count = 0;
  hr = MFEnumDeviceSources(attributes.Get(), &devices, &count);
  if (FAILED(hr)) {
    return hr;
  }
  for (UINT32 i = 0; i < count; ++i) {
    DeviceEntry entry;
    entry.activate.Attach(devices[i]);  // adopts the reference
    wchar_t* text = nullptr;
    UINT32 length = 0;
    if (SUCCEEDED(entry.activate->GetAllocatedString(
            MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &text, &length))) {
      entry.name = ToUtf8(text, length);
      CoTaskMemFree(text);
    }
    text = nullptr;
    length = 0;
    if (SUCCEEDED(entry.activate->GetAllocatedString(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &text,
            &length))) {
      entry.symbolic_link = ToUtf8(text, length);
      CoTaskMemFree(text);
    }
    out->push_back(std::move(entry));
  }
  CoTaskMemFree(devices);  // the array itself; elements were adopted above
  return S_OK;
}

// Options::device: all digits = index, otherwise a case-insensitive substring
// of the friendly name, empty = first device.
bool SelectDevice(const std::vector<DeviceEntry>& devices,
                  const std::string& selector, size_t* out_index,
                  std::string* out_error) {
  if (devices.empty()) {
    *out_error = "no camera found";
    return false;
  }
  if (selector.empty()) {
    *out_index = 0;
    return true;
  }
  bool all_digits = std::all_of(selector.begin(), selector.end(),
                                [](char c) { return c >= '0' && c <= '9'; });
  if (all_digits) {
    size_t index = static_cast<size_t>(strtoul(selector.c_str(), nullptr, 10));
    if (index >= devices.size()) {
      *out_error = fmt::format("camera index {} is out of range ({} found)",
                               index, devices.size());
      return false;
    }
    *out_index = index;
    return true;
  }
  std::string needle = ToLower(selector);
  for (size_t i = 0; i < devices.size(); ++i) {
    if (ToLower(devices[i].name).find(needle) != std::string::npos) {
      *out_index = i;
      return true;
    }
  }
  std::string available;
  for (const DeviceEntry& entry : devices) {
    if (!available.empty()) {
      available += ", ";
    }
    available += "\"" + entry.name + "\"";
  }
  *out_error = fmt::format("no camera name contains \"{}\" (available: {})",
                           selector, available);
  return false;
}

struct NativeFormat {
  DWORD index = 0;
  UINT32 width = 0;
  UINT32 height = 0;
  UINT32 fps_num = 0;
  UINT32 fps_den = 1;
  GUID subtype = GUID_NULL;
  ComPtr<IMFMediaType> type;
  double fps() const {
    return fps_den ? double(fps_num) / double(fps_den) : 0.0;
  }
};

HRESULT EnumerateNativeFormats(IMFSourceReader* reader,
                               std::vector<NativeFormat>* out) {
  for (DWORD i = 0;; ++i) {
    ComPtr<IMFMediaType> type;
    HRESULT hr = reader->GetNativeMediaType(kVideoStream, i, &type);
    if (hr == MF_E_NO_MORE_TYPES) {
      break;
    }
    if (FAILED(hr)) {
      return hr;
    }
    NativeFormat f;
    f.index = i;
    f.type = type;
    MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &f.width, &f.height);
    MFGetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, &f.fps_num, &f.fps_den);
    type->GetGUID(MF_MT_SUBTYPE, &f.subtype);
    out->push_back(std::move(f));
  }
  return S_OK;
}

// Lower is better. Resolution distance dominates, then frame rate, then a
// preference for uncompressed formats (no MJPEG decode latency).
double ScoreFormat(const NativeFormat& f, uint32_t want_w, uint32_t want_h,
                   double want_fps) {
  double score = std::abs(double(f.width) - double(want_w)) +
                 std::abs(double(f.height) - double(want_h));
  score += std::abs(f.fps() - want_fps) * 10.0;
  if (f.subtype == MFVideoFormat_MJPG || f.subtype == MFVideoFormat_H264) {
    score += 1.0;
  }
  return score;
}

class MediaFoundationCameraCapture : public CameraCapture {
 public:
  MediaFoundationCameraCapture() = default;
  ~MediaFoundationCameraCapture() override { Close(); }

  bool Open(const Options& options, std::string* out_error) override;
  void Close() override;
  bool is_open() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return open_;
  }
  bool ReadFrame(CameraFrame* frame, uint32_t timeout_ms,
                 std::string* out_error) override;

  std::string device_name() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return device_name_;
  }
  uint32_t width() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return width_;
  }
  uint32_t height() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return height_;
  }
  double fps() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return fps_;
  }

 private:
  // Everything owned by the capture thread for one Open()/Close() cycle.
  struct Session {
    ComPtr<IMFActivate> activate;
    ComPtr<IMFMediaSource> source;
    ComPtr<IMFSourceReader> reader;
    uint32_t width = 0;
    uint32_t height = 0;
    // Row pitch (bytes, positive) and row order used when only the flat
    // IMFMediaBuffer view is available (MF_MT_DEFAULT_STRIDE < 0 means the
    // flat buffer is stored bottom-up).
    uint32_t fallback_stride = 0;
    bool fallback_bottom_up = false;
    bool warned_flat_lock = false;
    std::vector<uint8_t> scratch;
  };

  void CaptureThreadMain(const Options& options);
  bool SetupSession(const Options& options, Session* session,
                    std::string* out_error);
  bool NegotiateFormat(const Options& options, Session* session,
                       std::string* out_error);
  bool ApplyOutputType(Session* session, uint32_t width, uint32_t height,
                       HRESULT* out_hr);
  bool ReadCurrentType(Session* session, double* out_fps,
                       std::string* out_error);
  void RunReadLoop(Session* session);
  bool ConvertAndPublish(Session* session, IMFSample* sample,
                         LONGLONG sample_time, uint64_t now_us,
                         LONGLONG mf_now);
  void TeardownSession(Session* session);
  void StopThread();
  void PublishError(std::string error);

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::unique_ptr<xe::threading::Thread> thread_;
  bool open_ = false;
  bool stop_requested_ = false;
  bool thread_exited_ = false;
  std::string error_;
  std::string device_name_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  double fps_ = 0.0;
  CameraFrame latest_;
  uint64_t latest_sequence_ = 0;
  uint64_t consumed_sequence_ = 0;
  uint64_t last_timestamp_us_ = 0;

  // The live reader, shared only so that Close() can Flush() a stuck
  // ReadSample as a last resort.
  std::mutex reader_mutex_;
  ComPtr<IMFSourceReader> shared_reader_;
};

bool MediaFoundationCameraCapture::Open(const Options& options,
                                        std::string* out_error) {
  std::string local_error;
  if (!out_error) {
    out_error = &local_error;
  }
  out_error->clear();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_) {
      *out_error = "camera is already open";
      return false;
    }
    open_ = false;
    stop_requested_ = false;
    thread_exited_ = false;
    error_.clear();
    device_name_.clear();
    width_ = 0;
    height_ = 0;
    fps_ = 0.0;
    latest_ = CameraFrame();
    latest_sequence_ = 0;
    consumed_sequence_ = 0;
    last_timestamp_us_ = 0;
  }

  Options thread_options = options;
  if (!thread_options.width || !thread_options.height) {
    thread_options.width = kColorWidth;
    thread_options.height = kColorHeight;
  }
  if (!thread_options.fps) {
    thread_options.fps = kFrameRateHz;
  }

  xe::threading::Thread::CreationParameters params;
  params.stack_size = 512 * 1024;
  std::unique_ptr<xe::threading::Thread> thread =
      xe::threading::Thread::Create(params, [this, thread_options]() {
        xe::threading::set_name("NUI Camera");
        CaptureThreadMain(thread_options);
      });
  if (!thread) {
    *out_error = "failed to create the camera capture thread";
    XELOGE("NUI camera: {}", *out_error);
    return false;
  }
  thread->set_name("NUI Camera");

  bool success = false;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    thread_ = std::move(thread);
    cv_.wait_for(lock, std::chrono::milliseconds(kOpenTimeoutMs), [this]() {
      return latest_sequence_ > 0 || !error_.empty() || thread_exited_;
    });
    if (latest_sequence_ > 0 && error_.empty()) {
      open_ = true;
      success = true;
    } else if (!error_.empty()) {
      *out_error = error_;
    } else if (thread_exited_) {
      *out_error = "the camera capture thread exited unexpectedly";
    } else {
      *out_error = fmt::format(
          "the camera did not deliver a frame within {} ms", kOpenTimeoutMs);
    }
  }
  if (!success) {
    XELOGE("NUI camera: open failed: {}", *out_error);
    StopThread();
  }
  return success;
}

void MediaFoundationCameraCapture::Close() {
  StopThread();
  std::lock_guard<std::mutex> lock(mutex_);
  open_ = false;
  latest_ = CameraFrame();
}

void MediaFoundationCameraCapture::StopThread() {
  std::unique_ptr<xe::threading::Thread> thread;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_requested_ = true;
    open_ = false;
    thread = std::move(thread_);
  }
  cv_.notify_all();
  if (!thread) {
    return;
  }
  auto result = xe::threading::Wait(
      thread.get(), false, std::chrono::milliseconds(kCloseJoinTimeoutMs));
  if (result != xe::threading::WaitResult::kSuccess) {
    // The driver did not return from ReadSample within a frame interval.
    // Flush cancels pending sample requests; the reader is free-threaded so
    // this is legal from another thread.
    ComPtr<IMFSourceReader> reader;
    {
      std::lock_guard<std::mutex> lock(reader_mutex_);
      reader = shared_reader_;
    }
    if (reader) {
      XELOGW(
          "NUI camera: capture thread still blocked after {} ms, flushing "
          "the source reader",
          kCloseJoinTimeoutMs);
      reader->Flush(kVideoStream);
      // Drop this reference before joining: the capture thread releases its
      // own on the way out and then calls MFShutdown / CoUninitialize, so
      // ours must not be the last one (released here, after MFShutdown, on
      // a thread that never initialised COM).
      reader.Reset();
    }
    xe::threading::Wait(thread.get(), false);
  }
  thread.reset();
}

void MediaFoundationCameraCapture::PublishError(std::string error) {
  XELOGE("NUI camera: {}", error);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_.empty()) {
      error_ = std::move(error);
    }
  }
  cv_.notify_all();
}

bool MediaFoundationCameraCapture::ReadFrame(CameraFrame* frame,
                                             uint32_t timeout_ms,
                                             std::string* out_error) {
  std::string local_error;
  if (!out_error) {
    out_error = &local_error;
  }
  out_error->clear();
  if (!frame) {
    *out_error = "no frame provided";
    return false;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (!open_) {
    *out_error = "camera is not open";
    return false;
  }
  // Bounded by |timeout_ms| only: the blocking ReadSample lives on the
  // capture thread, which wakes this wait as soon as a frame is converted.
  cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
    return latest_sequence_ > consumed_sequence_ || !error_.empty() ||
           stop_requested_ || thread_exited_;
  });
  if (latest_sequence_ > consumed_sequence_) {
    // Newest frame only; anything older was already overwritten (never
    // queued).
    frame->width = latest_.width;
    frame->height = latest_.height;
    frame->stride = latest_.stride;
    frame->rgba = latest_.rgba;
    frame->timestamp_us = latest_.timestamp_us;
    frame->sequence = latest_.sequence;
    consumed_sequence_ = latest_sequence_;
    return true;
  }
  if (!error_.empty()) {
    *out_error = error_;
    return false;
  }
  if (stop_requested_ || thread_exited_) {
    *out_error = "camera closed";
    return false;
  }
  // Timeout: no error.
  return false;
}

void MediaFoundationCameraCapture::CaptureThreadMain(const Options& options) {
  // Scoped objects unwind in reverse: session objects are released before
  // MFShutdown and CoUninitialize run.
  ScopedComApartment com;
  if (!com.ok()) {
    PublishError("CoInitializeEx failed: " + DescribeHresult(com.hr()));
    std::lock_guard<std::mutex> lock(mutex_);
    thread_exited_ = true;
    cv_.notify_all();
    return;
  }
  ScopedMediaFoundation mf;
  if (!mf.ok()) {
    PublishError("MFStartup failed: " + DescribeHresult(mf.hr()));
    std::lock_guard<std::mutex> lock(mutex_);
    thread_exited_ = true;
    cv_.notify_all();
    return;
  }
  {
    Session session;
    std::string error;
    if (SetupSession(options, &session, &error)) {
      {
        std::lock_guard<std::mutex> lock(reader_mutex_);
        shared_reader_ = session.reader;
      }
      RunReadLoop(&session);
      {
        std::lock_guard<std::mutex> lock(reader_mutex_);
        shared_reader_.Reset();
      }
    } else {
      PublishError(std::move(error));
    }
    TeardownSession(&session);
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    thread_exited_ = true;
  }
  cv_.notify_all();
}

bool MediaFoundationCameraCapture::SetupSession(const Options& options,
                                                Session* session,
                                                std::string* out_error) {
  std::vector<DeviceEntry> devices;
  HRESULT hr = EnumerateDeviceEntries(&devices);
  if (FAILED(hr)) {
    *out_error = "camera enumeration failed: " + DescribeHresult(hr);
    return false;
  }
  size_t selected = 0;
  if (!SelectDevice(devices, options.device, &selected, out_error)) {
    return false;
  }
  DeviceEntry& device = devices[selected];
  {
    std::lock_guard<std::mutex> lock(mutex_);
    device_name_ = device.name;
  }
  XELOGI("NUI camera: opening [{}] \"{}\" ({})", selected, device.name,
         device.symbolic_link);

  session->activate = device.activate;
  hr = session->activate->ActivateObject(IID_PPV_ARGS(&session->source));
  if (FAILED(hr)) {
    *out_error = "activating the camera failed: " + DescribeHresult(hr);
    return false;
  }

  ComPtr<IMFAttributes> reader_attributes;
  hr = MFCreateAttributes(&reader_attributes, 3);
  if (SUCCEEDED(hr)) {
    // Lets the reader insert the MJPEG decoder and the video processor so
    // any native type (NV12/YUY2/MJPG) can be delivered as RGB32, scaled if
    // requested.
    hr = reader_attributes->SetUINT32(
        MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, TRUE);
  }
  if (SUCCEEDED(hr)) {
    // CPU-only conversion: no hidden D3D11 device next to the emulator's
    // own D3D12/Vulkan devices; Lock() is a plain memory pointer.
    hr = reader_attributes->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, TRUE);
  }
  if (SUCCEEDED(hr)) {
    hr = MFCreateSourceReaderFromMediaSource(
        session->source.Get(), reader_attributes.Get(), &session->reader);
  }
  if (FAILED(hr)) {
    *out_error = "creating the camera reader failed: " + DescribeHresult(hr);
    return false;
  }

  if (!NegotiateFormat(options, session, out_error)) {
    return false;
  }

  hr = session->reader->SetStreamSelection(kVideoStream, TRUE);
  if (FAILED(hr)) {
    *out_error = "selecting the video stream failed: " + DescribeHresult(hr);
    return false;
  }
  return true;
}

bool MediaFoundationCameraCapture::ApplyOutputType(Session* session,
                                                   uint32_t width,
                                                   uint32_t height,
                                                   HRESULT* out_hr) {
  ComPtr<IMFMediaType> output_type;
  HRESULT hr = MFCreateMediaType(&output_type);
  if (SUCCEEDED(hr)) {
    hr = output_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
  }
  if (SUCCEEDED(hr)) {
    hr = output_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
  }
  // Frame size and rate are inherited from the native type unless a size is
  // given, in which case the video processor scales.
  if (SUCCEEDED(hr) && width && height) {
    hr = MFSetAttributeSize(output_type.Get(), MF_MT_FRAME_SIZE, width, height);
  }
  if (SUCCEEDED(hr)) {
    hr = session->reader->SetCurrentMediaType(kVideoStream, nullptr,
                                              output_type.Get());
  }
  *out_hr = hr;
  return SUCCEEDED(hr);
}

bool MediaFoundationCameraCapture::ReadCurrentType(Session* session,
                                                   double* out_fps,
                                                   std::string* out_error) {
  ComPtr<IMFMediaType> current;
  HRESULT hr = session->reader->GetCurrentMediaType(kVideoStream, &current);
  if (FAILED(hr)) {
    *out_error =
        "querying the camera output type failed: " + DescribeHresult(hr);
    return false;
  }
  UINT32 width = 0, height = 0, fps_num = 0, fps_den = 1;
  MFGetAttributeSize(current.Get(), MF_MT_FRAME_SIZE, &width, &height);
  MFGetAttributeRatio(current.Get(), MF_MT_FRAME_RATE, &fps_num, &fps_den);
  if (!width || !height) {
    *out_error = "the camera output type has no frame size";
    return false;
  }
  GUID subtype = GUID_NULL;
  current->GetGUID(MF_MT_SUBTYPE, &subtype);
  if (subtype != MFVideoFormat_RGB32) {
    *out_error = fmt::format("the camera reader produced {} instead of RGB32",
                             SubtypeName(subtype));
    return false;
  }
  // MF_MT_DEFAULT_STRIDE is an INT32 stored as UINT32. Its magnitude is the
  // pitch of a flat buffer and its sign the row order of that flat buffer
  // (negative = bottom-up, the classic RGB32 convention); neither matters on
  // the IMF2DBuffer path (see the orientation note at the top of the file).
  UINT32 stride_attr = 0;
  uint32_t stride = width * 4;
  bool bottom_up = false;
  if (SUCCEEDED(current->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride_attr))) {
    INT32 signed_stride = static_cast<INT32>(stride_attr);
    uint32_t magnitude = static_cast<uint32_t>(
        signed_stride < 0 ? -signed_stride : signed_stride);
    if (magnitude >= width * 4) {
      stride = magnitude;
      bottom_up = signed_stride < 0;
    }
  }
  session->width = width;
  session->height = height;
  session->fallback_stride = stride;
  session->fallback_bottom_up = bottom_up;
  *out_fps = fps_den ? double(fps_num) / double(fps_den) : 0.0;
  return true;
}

bool MediaFoundationCameraCapture::NegotiateFormat(const Options& options,
                                                   Session* session,
                                                   std::string* out_error) {
  std::vector<NativeFormat> formats;
  HRESULT hr = EnumerateNativeFormats(session->reader.Get(), &formats);
  if (FAILED(hr)) {
    *out_error = "listing the camera formats failed: " + DescribeHresult(hr);
    return false;
  }
  if (formats.empty()) {
    *out_error = "the camera exposes no video formats";
    return false;
  }
  const double want_fps = double(options.fps);
  std::vector<size_t> order(formats.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return ScoreFormat(formats[a], options.width, options.height, want_fps) <
           ScoreFormat(formats[b], options.width, options.height, want_fps);
  });

  // Walk the candidates from best to worst until one negotiates. Pin the
  // native type first, then ask for RGB32 output.
  HRESULT last_hr = S_OK;
  for (size_t candidate : order) {
    const NativeFormat& native = formats[candidate];
    hr = session->reader->SetCurrentMediaType(kVideoStream, nullptr,
                                              native.type.Get());
    if (FAILED(hr)) {
      last_hr = hr;
      continue;
    }
    // Ask the video processor to scale when the native size differs but has
    // the same aspect ratio and is at least as large (so a camera without a
    // 640x480 mode still produces 640x480). Otherwise inherit the native
    // size and let the pipeline resample.
    bool scaled = false;
    bool applied = false;
    if (native.width != options.width || native.height != options.height) {
      double native_aspect = double(native.width) / double(native.height);
      double want_aspect = double(options.width) / double(options.height);
      bool same_aspect =
          std::abs(native_aspect - want_aspect) <= want_aspect * 0.02;
      if (same_aspect && native.width >= options.width &&
          native.height >= options.height) {
        applied = ApplyOutputType(session, options.width, options.height, &hr);
        scaled = applied;
      }
    }
    if (!applied) {
      applied = ApplyOutputType(session, 0, 0, &hr);
    }
    if (!applied) {
      last_hr = hr;
      continue;
    }
    double fps = 0.0;
    if (!ReadCurrentType(session, &fps, out_error)) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      width_ = session->width;
      height_ = session->height;
      fps_ = fps;
    }
    XELOGI(
        "NUI camera: native #{} {} {}x{} @ {:.2f} fps -> RGB32 {}x{} @ {:.2f} "
        "fps{} (requested {}x{} @ {} fps, {} native formats)",
        native.index, SubtypeName(native.subtype), native.width, native.height,
        native.fps(), session->width, session->height, fps,
        scaled ? " (scaled by the video processor)" : "", options.width,
        options.height, options.fps, formats.size());
    return true;
  }
  *out_error = "no camera format could be converted to RGB32: " +
               DescribeHresult(last_hr);
  return false;
}

void MediaFoundationCameraCapture::RunReadLoop(Session* session) {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_requested_) {
        break;
      }
    }
    DWORD actual_stream = 0;
    DWORD flags = 0;
    LONGLONG sample_time = 0;
    ComPtr<IMFSample> sample;
    HRESULT hr = session->reader->ReadSample(kVideoStream, 0, &actual_stream,
                                             &flags, &sample_time, &sample);
    // Host time and MF clock read back to back so the sample age can be
    // transferred between the two domains.
    uint64_t now_us = SteadyNowMicroseconds();
    LONGLONG mf_now = MFGetSystemTime();
    if (FAILED(hr)) {
      PublishError(DescribeHresult(hr));
      break;
    }
    if (flags & MF_SOURCE_READERF_ERROR) {
      PublishError("the camera reported a stream error");
      break;
    }
    if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
      PublishError("the camera was disconnected (end of stream)");
      break;
    }
    if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
      double fps = 0.0;
      std::string error;
      if (!ReadCurrentType(session, &fps, &error)) {
        PublishError(std::move(error));
        break;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        width_ = session->width;
        height_ = session->height;
        fps_ = fps;
      }
      XELOGW("NUI camera: output type changed to {}x{} @ {:.2f} fps",
             session->width, session->height, fps);
    }
    if (!sample) {
      // MF_SOURCE_READERF_STREAMTICK or a gap: nothing to convert.
      continue;
    }
    if (!ConvertAndPublish(session, sample.Get(), sample_time, now_us,
                           mf_now)) {
      break;
    }
  }
}

bool MediaFoundationCameraCapture::ConvertAndPublish(Session* session,
                                                     IMFSample* sample,
                                                     LONGLONG sample_time,
                                                     uint64_t now_us,
                                                     LONGLONG mf_now) {
  const uint32_t width = session->width;
  const uint32_t height = session->height;
  ComPtr<IMFMediaBuffer> buffer;
  HRESULT hr = sample->ConvertToContiguousBuffer(&buffer);
  if (FAILED(hr)) {
    PublishError("reading the camera sample failed: " + DescribeHresult(hr));
    return false;
  }

  ComPtr<IMF2DBuffer> buffer2d;
  BYTE* scanline0 = nullptr;
  LONG pitch = 0;
  bool locked_2d = false;
  if (SUCCEEDED(buffer.As(&buffer2d)) &&
      SUCCEEDED(buffer2d->Lock2D(&scanline0, &pitch))) {
    locked_2d = true;
  } else {
    // Flat view: stride and row order from the media type's default stride.
    buffer2d.Reset();
    BYTE* data = nullptr;
    DWORD max_length = 0, current_length = 0;
    hr = buffer->Lock(&data, &max_length, &current_length);
    if (FAILED(hr)) {
      PublishError("locking the camera sample failed: " + DescribeHresult(hr));
      return false;
    }
    if (current_length < uint64_t(session->fallback_stride) * height) {
      buffer->Unlock();
      XELOGW("NUI camera: dropped a short sample ({} bytes for {}x{})",
             current_length, width, height);
      return true;
    }
    if (!session->warned_flat_lock) {
      session->warned_flat_lock = true;
      XELOGW(
          "NUI camera: IMF2DBuffer unavailable, using the flat buffer with "
          "stride {} ({} rows; if the picture is upside down, report this "
          "camera)",
          session->fallback_stride,
          session->fallback_bottom_up ? "bottom-up" : "top-down");
    }
    // Express the flat buffer as (top scanline, signed pitch) so the walk
    // below is the same for both row orders.
    if (session->fallback_bottom_up) {
      scanline0 = data + size_t(session->fallback_stride) * (height - 1);
      pitch = -static_cast<LONG>(session->fallback_stride);
    } else {
      scanline0 = data;
      pitch = static_cast<LONG>(session->fallback_stride);
    }
  }

  // BGRX -> RGBA, top-down. scanline0 + y * pitch walks image rows from the
  // top for either pitch sign (IMF2DBuffer contract).
  std::vector<uint8_t>& rgba = session->scratch;
  rgba.resize(size_t(width) * height * 4);
  for (uint32_t y = 0; y < height; ++y) {
    const uint8_t* src = scanline0 + ptrdiff_t(y) * pitch;
    uint8_t* dst = rgba.data() + size_t(y) * width * 4;
    for (uint32_t x = 0; x < width; ++x) {
      dst[0] = src[2];
      dst[1] = src[1];
      dst[2] = src[0];
      dst[3] = 0xFF;
      src += 4;
      dst += 4;
    }
  }
  if (locked_2d) {
    buffer2d->Unlock2D();
  } else {
    buffer->Unlock();
  }

  // Timestamp mapping. |sample_time| is the sample's presentation time in
  // 100 ns units on the Media Foundation clock; for capture sources that is
  // MFGetSystemTime() at the moment the driver completed the frame (QPC
  // based). CameraFrame::timestamp_us wants the host steady clock (also QPC
  // on MSVC, but a different epoch), so instead of converting the absolute
  // value the sample's *age* is transferred: |mf_now| and |now_us| were read
  // back to back after ReadSample returned, so
  //   age     = mf_now - sample_time            (100 ns -> us: / 10)
  //   capture = now_us - age
  // approximates the exposure time in the steady-clock domain and cancels
  // the decode/convert latency of the reader. The correction is skipped for
  // ages beyond kMaxSampleAgeUs (drivers that report a bogus or zero
  // sample_time), leaving the return time. Kept strictly increasing so
  // consumers can use it as a sequence key.
  uint64_t capture_us = now_us;
  if (sample_time > 0 && mf_now >= sample_time) {
    uint64_t age_us = static_cast<uint64_t>(mf_now - sample_time) / 10;
    if (age_us < kMaxSampleAgeUs && age_us < now_us) {
      capture_us = now_us - age_us;
    }
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (capture_us <= last_timestamp_us_) {
      capture_us = last_timestamp_us_ + 1;
    }
    last_timestamp_us_ = capture_us;
    latest_.width = width;
    latest_.height = height;
    latest_.stride = width * 4;
    std::swap(latest_.rgba, rgba);
    latest_.timestamp_us = capture_us;
    latest_.sequence = ++latest_sequence_;
  }
  cv_.notify_all();
  return true;
}

void MediaFoundationCameraCapture::TeardownSession(Session* session) {
  // Reader first, then the source must be shut down explicitly or the device
  // stays claimed ("in use" on the next open), then the activation object.
  session->reader.Reset();
  if (session->source) {
    session->source->Shutdown();
    session->source.Reset();
  }
  if (session->activate) {
    session->activate->ShutdownObject();
    session->activate.Reset();
  }
  session->scratch.clear();
  session->scratch.shrink_to_fit();
}

}  // namespace

std::vector<CameraDeviceInfo> EnumerateCameraDevicesWin() {
  std::vector<CameraDeviceInfo> result;
  ScopedComApartment com;
  if (!com.ok()) {
    XELOGE("NUI camera: CoInitializeEx failed: {}", DescribeHresult(com.hr()));
    return result;
  }
  ScopedMediaFoundation mf;
  if (!mf.ok()) {
    XELOGE("NUI camera: MFStartup failed: {}", DescribeHresult(mf.hr()));
    return result;
  }
  {
    std::vector<DeviceEntry> devices;
    HRESULT hr = EnumerateDeviceEntries(&devices);
    if (FAILED(hr)) {
      XELOGE("NUI camera: enumeration failed: {}", DescribeHresult(hr));
      return result;
    }
    result.reserve(devices.size());
    for (DeviceEntry& entry : devices) {
      CameraDeviceInfo info;
      info.name = std::move(entry.name);
      info.id = std::move(entry.symbolic_link);
      result.push_back(std::move(info));
    }
    // |devices| (and its IMFActivate references) go away here, before
    // MFShutdown.
  }
  return result;
}

std::unique_ptr<CameraCapture> CreateCameraCaptureWin() {
  return std::make_unique<MediaFoundationCameraCapture>();
}

}  // namespace nui
}  // namespace xe

#endif  // XE_PLATFORM_WIN32
