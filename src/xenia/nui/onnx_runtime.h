/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_ONNX_RUNTIME_H_
#define XENIA_NUI_ONNX_RUNTIME_H_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// ONNX Runtime is loaded at runtime from a user-provided onnxruntime.dll
// (with DirectML.dll next to it for GPU inference); nothing links against it
// at build time. Only the C API header is vendored under
// third_party/onnxruntime/include.

struct OrtApi;

namespace xe {
namespace nui {

class OnnxRuntime {
 public:
  // Loads the runtime on first use, searching |explicit_dir| (may be empty),
  // then <storage_root>/nui/runtime, then the executable directory. Returns
  // nullptr when unavailable; load_error() explains why.
  static OnnxRuntime* Get(const std::filesystem::path& explicit_dir,
                          const std::filesystem::path& storage_root);
  // A copy, taken under the loader lock: Get() rewrites the stored string.
  static std::string load_error();

  const OrtApi* api() const { return api_; }
  bool has_directml() const { return has_directml_; }
  const std::string& version() const { return version_; }
  const std::filesystem::path& runtime_dir() const { return runtime_dir_; }

 private:
  friend class OnnxSession;
  OnnxRuntime() = default;
  const OrtApi* api_ = nullptr;
  void* library_ = nullptr;
  void* env_ = nullptr;                   // OrtEnv*, one per process
  void* directml_provider_fn_ = nullptr;  // const OrtDmlApi*
  bool has_directml_ = false;
  std::string version_;
  std::filesystem::path runtime_dir_;
};

// True when an ONNX Runtime / DirectML error message describes a lost GPU
// device (a TDR, a driver reset or an adapter that went away) rather than a
// model or input problem. Everything after such a failure fails the same way
// until the sessions are rebuilt, so callers use this to decide between
// "retry the frame" and "throw the sessions away and start over".
//
// The DirectML execution provider surfaces these as ORT_EP_FAIL or
// ORT_RUNTIME_EXCEPTION statuses whose message carries the DXGI name or
// HRESULT, so the message text is the only reliable signal. Free function so
// it can be tested without a GPU.
bool OnnxErrorIsDeviceLost(std::string_view message);

struct OnnxTensorInfo {
  std::string name;
  std::vector<int64_t> shape;  // -1 for dynamic dimensions
  int element_type = 0;        // ONNXTensorElementDataType
};

// One loaded model with pre-bound float32 inputs and outputs.
class OnnxSession {
 public:
  struct Options {
    std::string execution_provider = "auto";  // auto | dml | cpu
    int intra_op_threads = 0;                 // 0 = auto
    // DXGI adapter index for DirectML; -1 = the high-performance GPU.
    int device_id = -1;
  };

  static std::unique_ptr<OnnxSession> Create(OnnxRuntime* runtime,
                                             const std::filesystem::path& model,
                                             const Options& options,
                                             std::string* out_error);
  ~OnnxSession();

  const std::vector<OnnxTensorInfo>& inputs() const { return inputs_; }
  const std::vector<OnnxTensorInfo>& outputs() const { return outputs_; }
  // "DirectML" or "CPU".
  const std::string& provider_name() const { return provider_name_; }

  // Runs the model. |input_data| holds one float32 buffer per input (in
  // inputs() order) with the given concrete shapes; outputs are written to
  // |out_outputs| (one float32 vector per output, in outputs() order) with
  // their concrete shapes in |out_shapes|.
  bool Run(const std::vector<const float*>& input_data,
           const std::vector<std::vector<int64_t>>& input_shapes,
           std::vector<std::vector<float>>* out_outputs,
           std::vector<std::vector<int64_t>>* out_shapes,
           std::string* out_error);

  // Set once a Run() failure was classified as a lost device (see
  // OnnxErrorIsDeviceLost); never cleared, because this session cannot
  // recover - the owner has to destroy it and create a new one.
  bool device_lost() const { return device_lost_; }

 private:
  OnnxSession() = default;
  // Run() without the device-lost bookkeeping.
  bool RunInternal(const std::vector<const float*>& input_data,
                   const std::vector<std::vector<int64_t>>& input_shapes,
                   std::vector<std::vector<float>>* out_outputs,
                   std::vector<std::vector<int64_t>>* out_shapes,
                   std::string* out_error);

  OnnxRuntime* runtime_ = nullptr;
  void* session_ = nullptr;      // OrtSession*
  void* allocator_ = nullptr;    // OrtAllocator*
  void* memory_info_ = nullptr;  // OrtMemoryInfo*
  std::vector<OnnxTensorInfo> inputs_;
  std::vector<OnnxTensorInfo> outputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  // Constant for the life of the session, built once in Create() so that
  // Run() allocates nothing: the name arrays ORT wants and the OrtValue*
  // slots (void* so the C API types stay out of this header).
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;
  std::vector<void*> input_values_;
  std::vector<void*> output_values_;
  std::string provider_name_;
  bool device_lost_ = false;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_ONNX_RUNTIME_H_
