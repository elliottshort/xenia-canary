/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/onnx_runtime.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iterator>
#include <mutex>
#include <thread>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/base/string.h"

#if XE_PLATFORM_WIN32
// Windows headers first: dml_provider_factory.h needs <d3d12.h> and
// <DirectML.h> (from the Windows SDK) and #undefs OPTIONAL.
#include "xenia/base/platform_win.h"

#include <d3d12.h>
#include <dxgi.h>

#include "third_party/onnxruntime/include/dml_provider_factory.h"
#endif  // XE_PLATFORM_WIN32

#include "third_party/onnxruntime/include/onnxruntime_c_api.h"

namespace xe {
namespace nui {

namespace {

std::mutex g_load_mutex;
bool g_load_attempted = false;
OnnxRuntime* g_runtime = nullptr;
std::string g_load_error;

constexpr char kRuntimeDll[] = "onnxruntime.dll";
constexpr char kDirectMLDll[] = "DirectML.dll";

// Takes ownership of |status|; returns false (with the message in
// |out_error|) when it is an error.
bool CheckStatus(const OrtApi* api, OrtStatus* status, const char* what,
                 std::string* out_error) {
  if (!status) {
    return true;
  }
  const char* message = api->GetErrorMessage(status);
  if (out_error) {
    *out_error = fmt::format("{}: {}", what, message ? message : "unknown");
  }
  api->ReleaseStatus(status);
  return false;
}

std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) {
      out += ",";
    }
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

bool ElementCount(const std::vector<int64_t>& shape, size_t* out_count) {
  size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      return false;
    }
    count *= static_cast<size_t>(dim);
  }
  *out_count = count;
  return true;
}

#if XE_PLATFORM_WIN32

void ORT_API_CALL OrtLogCallback(void* /*param*/, OrtLoggingLevel severity,
                                 const char* /*category*/,
                                 const char* /*logid*/,
                                 const char* /*code_location*/,
                                 const char* message) {
  const char* text = message ? message : "";
  switch (severity) {
    case ORT_LOGGING_LEVEL_ERROR:
    case ORT_LOGGING_LEVEL_FATAL:
      XELOGE("NUI: ONNX Runtime: {}", text);
      break;
    case ORT_LOGGING_LEVEL_WARNING:
      XELOGW("NUI: ONNX Runtime: {}", text);
      break;
    default:
      XELOGI("NUI: ONNX Runtime: {}", text);
      break;
  }
}

std::filesystem::path ModulePath(HMODULE module) {
  wchar_t buffer[MAX_PATH * 2];
  DWORD length =
      GetModuleFileNameW(module, buffer, static_cast<DWORD>(std::size(buffer)));
  return std::filesystem::path(std::wstring(buffer, length));
}

std::string LastErrorString() {
  DWORD code = GetLastError();
  return fmt::format("Win32 error {} (0x{:08X})", code, code);
}

// Name of the DXGI adapter with the given LUID, for the log. dxgi.dll is
// resolved dynamically so xenia-nui needs no link-time dependency on it.
std::string AdapterNameForLuid(const LUID& luid) {
  HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
  if (!dxgi) {
    return "unknown adapter";
  }
  std::string name = "unknown adapter";
  using CreateDXGIFactory1Fn = HRESULT(WINAPI*)(REFIID, void**);
  auto create_factory = reinterpret_cast<CreateDXGIFactory1Fn>(
      GetProcAddress(dxgi, "CreateDXGIFactory1"));
  IDXGIFactory1* factory = nullptr;
  if (create_factory &&
      SUCCEEDED(create_factory(__uuidof(IDXGIFactory1),
                               reinterpret_cast<void**>(&factory)))) {
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0;
         factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
      DXGI_ADAPTER_DESC1 desc;
      if (SUCCEEDED(adapter->GetDesc1(&desc)) &&
          desc.AdapterLuid.LowPart == luid.LowPart &&
          desc.AdapterLuid.HighPart == luid.HighPart) {
        name = fmt::format(
            "dxgi#{} {}", i,
            xe::to_utf8(std::u16string_view(
                reinterpret_cast<const char16_t*>(desc.Description))));
        adapter->Release();
        break;
      }
      adapter->Release();
    }
    factory->Release();
  }
  FreeLibrary(dxgi);
  return name;
}

bool PathStartsWith(const std::filesystem::path& path,
                    const std::filesystem::path& prefix) {
  std::wstring a = path.lexically_normal().native();
  std::wstring b = prefix.lexically_normal().native();
  while (!b.empty() && (b.back() == L'\\' || b.back() == L'/')) {
    b.pop_back();
  }
  if (a.size() < b.size()) {
    return false;
  }
  return CompareStringOrdinal(a.c_str(), static_cast<int>(b.size()), b.c_str(),
                              static_cast<int>(b.size()), TRUE) == CSTR_EQUAL;
}

#endif  // XE_PLATFORM_WIN32

}  // namespace

const std::string& OnnxRuntime::load_error() { return g_load_error; }

#if XE_PLATFORM_WIN32

OnnxRuntime* OnnxRuntime::Get(const std::filesystem::path& explicit_dir,
                              const std::filesystem::path& storage_root) {
  std::lock_guard<std::mutex> lock(g_load_mutex);
  if (g_load_attempted) {
    return g_runtime;
  }
  g_load_attempted = true;

  // Search order; every candidate is made absolute so the loader never
  // consults the default search path (C:\Windows\System32 carries an old
  // WinML build of onnxruntime.dll/DirectML.dll that must not be used).
  std::vector<std::filesystem::path> candidates;
  if (!explicit_dir.empty()) {
    candidates.push_back(explicit_dir);
  }
  if (!storage_root.empty()) {
    candidates.push_back(storage_root / "nui" / "runtime");
  }
  std::filesystem::path exe_dir = xe::filesystem::GetExecutableFolder();
  candidates.push_back(exe_dir / "nui" / "runtime");
  candidates.push_back(exe_dir);

  std::filesystem::path runtime_dir;
  std::string searched;
  for (const auto& candidate : candidates) {
    std::error_code ec;
    std::filesystem::path absolute = std::filesystem::absolute(candidate, ec);
    if (ec) {
      absolute = candidate;
    }
    absolute = absolute.lexically_normal();
    if (!searched.empty()) {
      searched += "; ";
    }
    searched += xe::path_to_utf8(absolute);
    if (std::filesystem::exists(absolute / kRuntimeDll, ec)) {
      runtime_dir = absolute;
      break;
    }
  }
  if (runtime_dir.empty()) {
    g_load_error = fmt::format(
        "{} not found (searched: {}). Run tools/nui/setup_nui.py or copy "
        "onnxruntime.dll and DirectML.dll into <storage>/nui/runtime.",
        kRuntimeDll, searched);
    XELOGW("NUI: {}", g_load_error);
    return nullptr;
  }

  const DWORD load_flags =
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;

  // DirectML.dll first. It is a delay-load import of onnxruntime.dll; a module
  // already loaded under that name is what later resolutions by base name
  // return, so the copy in System32 can never win. Without it inference
  // still works on the CPU.
  bool directml_loaded = false;
  std::string directml_error;
  std::filesystem::path directml_path = runtime_dir / kDirectMLDll;
  {
    std::error_code ec;
    if (!std::filesystem::exists(directml_path, ec)) {
      directml_error = fmt::format("{} not found in {}", kDirectMLDll,
                                   xe::path_to_utf8(runtime_dir));
    } else if (!LoadLibraryExW(directml_path.c_str(), nullptr, load_flags)) {
      directml_error =
          fmt::format("{} failed to load: {}", kDirectMLDll, LastErrorString());
    } else {
      directml_loaded = true;
    }
  }

  std::filesystem::path runtime_path = runtime_dir / kRuntimeDll;
  HMODULE library = LoadLibraryExW(runtime_path.c_str(), nullptr, load_flags);
  if (!library) {
    g_load_error = fmt::format(
        "{} failed to load: {}. The Visual C++ 2015-2022 x64 redistributable "
        "is required.",
        xe::path_to_utf8(runtime_path), LastErrorString());
    XELOGE("NUI: {}", g_load_error);
    return nullptr;
  }
  std::filesystem::path loaded_path = ModulePath(library);
  if (!PathStartsWith(loaded_path, runtime_dir)) {
    XELOGW(
        "NUI: {} was already loaded from {} (expected {}); using the loaded "
        "copy",
        kRuntimeDll, xe::path_to_utf8(loaded_path),
        xe::path_to_utf8(runtime_dir));
  }

  using OrtGetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
  auto get_api_base = reinterpret_cast<OrtGetApiBaseFn>(
      GetProcAddress(library, "OrtGetApiBase"));
  if (!get_api_base) {
    g_load_error = fmt::format("{} does not export OrtGetApiBase",
                               xe::path_to_utf8(loaded_path));
    XELOGE("NUI: {}", g_load_error);
    return nullptr;
  }
  const OrtApiBase* api_base = get_api_base();
  if (!api_base) {
    g_load_error = "OrtGetApiBase returned null";
    XELOGE("NUI: {}", g_load_error);
    return nullptr;
  }
  const char* version_string =
      api_base->GetVersionString ? api_base->GetVersionString() : nullptr;
  std::string version = version_string ? version_string : "unknown";
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  if (!api) {
    g_load_error = fmt::format(
        "{} (version {}) does not provide ONNX Runtime API {}; a newer "
        "onnxruntime.dll is required",
        xe::path_to_utf8(loaded_path), version, ORT_API_VERSION);
    XELOGE("NUI: {}", g_load_error);
    return nullptr;
  }

  // The DirectML execution provider is compiled into the DirectML build of
  // onnxruntime.dll; the CPU-only build reports only CPUExecutionProvider.
  bool build_has_dml = false;
  {
    char** providers = nullptr;
    int provider_count = 0;
    std::string error;
    if (CheckStatus(api,
                    api->GetAvailableProviders(&providers, &provider_count),
                    "GetAvailableProviders", &error)) {
      for (int i = 0; i < provider_count; ++i) {
        if (providers[i] &&
            std::strcmp(providers[i], "DmlExecutionProvider") == 0) {
          build_has_dml = true;
        }
      }
      api->ReleaseAvailableProviders(providers, provider_count);
    } else {
      XELOGW("NUI: {}", error);
    }
  }
  const OrtDmlApi* dml_api = nullptr;
  if (build_has_dml && directml_loaded) {
    std::string error;
    if (!CheckStatus(api,
                     api->GetExecutionProviderApi(
                         "DML", ORT_API_VERSION,
                         reinterpret_cast<const void**>(&dml_api)),
                     "GetExecutionProviderApi(DML)", &error)) {
      dml_api = nullptr;
      directml_error = error;
    }
  } else if (!build_has_dml) {
    directml_error = fmt::format(
        "{} is not the DirectML build (providers do not include "
        "DmlExecutionProvider)",
        kRuntimeDll);
  }
  bool has_directml = dml_api != nullptr;

  OrtEnv* env = nullptr;
  {
    std::string error;
    if (!CheckStatus(api,
                     api->CreateEnvWithCustomLogger(OrtLogCallback, nullptr,
                                                    ORT_LOGGING_LEVEL_WARNING,
                                                    "xenia_nui", &env),
                     "CreateEnv", &error)) {
      g_load_error = error;
      XELOGE("NUI: {}", g_load_error);
      return nullptr;
    }
  }

  auto runtime = new OnnxRuntime();
  runtime->api_ = api;
  runtime->library_ = library;
  runtime->env_ = env;
  runtime->directml_provider_fn_ =
      const_cast<void*>(static_cast<const void*>(dml_api));
  runtime->has_directml_ = has_directml;
  runtime->version_ = version;
  runtime->runtime_dir_ = runtime_dir;
  g_runtime = runtime;
  g_load_error.clear();

  if (has_directml) {
    XELOGI("NUI: ONNX Runtime {} loaded from {} (DirectML available)", version,
           xe::path_to_utf8(runtime_dir));
  } else {
    XELOGW(
        "NUI: ONNX Runtime {} loaded from {}; DirectML unavailable ({}), "
        "inference will run on the CPU",
        version, xe::path_to_utf8(runtime_dir), directml_error);
  }
  return runtime;
}

std::unique_ptr<OnnxSession> OnnxSession::Create(
    OnnxRuntime* runtime, const std::filesystem::path& model,
    const Options& options, std::string* out_error) {
  if (!runtime || !runtime->api() || !runtime->env_) {
    if (out_error) {
      *out_error = "ONNX Runtime is not loaded";
    }
    return nullptr;
  }
  const OrtApi* api = runtime->api();
  std::string error;

  std::string provider = options.execution_provider;
  std::transform(provider.begin(), provider.end(), provider.begin(),
                 [](unsigned char c) { return static_cast<char>(tolower(c)); });
  bool want_dml = false;
  bool require_dml = false;
  if (provider.empty() || provider == "auto") {
    want_dml = runtime->has_directml();
  } else if (provider == "dml" || provider == "directml") {
    want_dml = true;
    require_dml = true;
    if (!runtime->has_directml()) {
      if (out_error) {
        *out_error =
            "DirectML execution provider requested but unavailable (see the "
            "NUI runtime log line); use nui_execution_provider=auto or cpu";
      }
      return nullptr;
    }
  } else if (provider == "cpu") {
    want_dml = false;
  } else {
    if (out_error) {
      *out_error = fmt::format(
          "unknown execution provider '{}' (expected auto, dml or cpu)",
          options.execution_provider);
    }
    return nullptr;
  }

  std::error_code ec;
  if (!std::filesystem::exists(model, ec)) {
    if (out_error) {
      *out_error = fmt::format("model not found: {}", xe::path_to_utf8(model));
    }
    return nullptr;
  }

  OrtSessionOptions* session_options = nullptr;
  if (!CheckStatus(api, api->CreateSessionOptions(&session_options),
                   "CreateSessionOptions", out_error)) {
    return nullptr;
  }
  auto fail = [&](const std::string& message) -> std::unique_ptr<OnnxSession> {
    if (out_error) {
      *out_error = message;
    }
    api->ReleaseSessionOptions(session_options);
    return nullptr;
  };

  int threads = options.intra_op_threads;
  if (threads <= 0) {
    unsigned int cores = std::thread::hardware_concurrency();
    threads = std::clamp(static_cast<int>(cores / 4), 1, 4);
    if (want_dml) {
      // Only CPU fallback nodes and pre/post processing run on this pool.
      threads = std::min(threads, 2);
    }
  }

  // DirectML requires sequential execution and no memory patterns; the rest
  // keeps inference from competing with the emulator's own threads.
  if (!CheckStatus(api,
                   api->SetSessionGraphOptimizationLevel(session_options,
                                                         ORT_ENABLE_ALL),
                   "SetSessionGraphOptimizationLevel", &error) ||
      !CheckStatus(
          api, api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL),
          "SetSessionExecutionMode", &error) ||
      !CheckStatus(api, api->DisableMemPattern(session_options),
                   "DisableMemPattern", &error) ||
      !CheckStatus(api, api->SetIntraOpNumThreads(session_options, threads),
                   "SetIntraOpNumThreads", &error) ||
      !CheckStatus(api, api->SetInterOpNumThreads(session_options, 1),
                   "SetInterOpNumThreads", &error) ||
      !CheckStatus(api,
                   api->AddSessionConfigEntry(
                       session_options, "session.intra_op.allow_spinning", "0"),
                   "AddSessionConfigEntry(allow_spinning)", &error) ||
      !CheckStatus(api, api->SetSessionLogId(session_options, "xenia_nui"),
                   "SetSessionLogId", &error) ||
      !CheckStatus(api,
                   api->SetSessionLogSeverityLevel(session_options,
                                                   ORT_LOGGING_LEVEL_WARNING),
                   "SetSessionLogSeverityLevel", &error)) {
    return fail(error);
  }

  std::string provider_name = "CPU";
  std::string adapter_name;
  if (want_dml) {
    auto dml_api =
        static_cast<const OrtDmlApi*>(runtime->directml_provider_fn_);
    OrtStatus* status = nullptr;
    if (options.device_id < 0) {
      // Let DirectML pick the high-performance GPU; adapter index 0 is only
      // the primary display adapter, which may be an iGPU.
      OrtDmlDeviceOptions device_options = {};
      device_options.Preference = HighPerformance;
      device_options.Filter = Gpu;
      status = dml_api->SessionOptionsAppendExecutionProvider_DML2(
          session_options, &device_options);
      if (status) {
        XELOGW(
            "NUI: DirectML high-performance adapter selection failed: {}; "
            "trying adapter 0",
            api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        status = dml_api->SessionOptionsAppendExecutionProvider_DML(
            session_options, 0);
      }
    } else {
      status = dml_api->SessionOptionsAppendExecutionProvider_DML(
          session_options, options.device_id);
    }
    if (status) {
      std::string message = fmt::format("DirectML execution provider: {}",
                                        api->GetErrorMessage(status));
      api->ReleaseStatus(status);
      if (require_dml) {
        return fail(message);
      }
      XELOGW("NUI: {}; falling back to CPU inference", message);
    } else {
      provider_name = "DirectML";
      // GetDMLDevice hands out a non-owning pointer: do not Release() it.
      IDMLDevice* dml_device = nullptr;
      OrtStatus* device_status =
          dml_api->GetDMLDevice(session_options, &dml_device);
      if (device_status) {
        api->ReleaseStatus(device_status);
      } else if (dml_device) {
        ID3D12Device* d3d_device = nullptr;
        if (SUCCEEDED(dml_device->GetParentDevice(
                __uuidof(ID3D12Device),
                reinterpret_cast<void**>(&d3d_device)))) {
          adapter_name = AdapterNameForLuid(d3d_device->GetAdapterLuid());
          d3d_device->Release();
        }
      }
    }
  }

  auto start_time = std::chrono::steady_clock::now();
  OrtSession* ort_session = nullptr;
  if (!CheckStatus(
          api,
          api->CreateSession(static_cast<OrtEnv*>(runtime->env_),
                             model.native().c_str(), session_options,
                             &ort_session),
          fmt::format("CreateSession({})", xe::path_to_utf8(model)).c_str(),
          &error)) {
    return fail(error);
  }
  api->ReleaseSessionOptions(session_options);
  session_options = nullptr;
  double create_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start_time)
                         .count();

  std::unique_ptr<OnnxSession> session(new OnnxSession());
  session->runtime_ = runtime;
  session->session_ = ort_session;
  session->provider_name_ = provider_name;

  OrtAllocator* allocator = nullptr;
  if (!CheckStatus(api, api->GetAllocatorWithDefaultOptions(&allocator),
                   "GetAllocatorWithDefaultOptions", out_error)) {
    return nullptr;
  }
  session->allocator_ = allocator;
  OrtMemoryInfo* memory_info = nullptr;
  if (!CheckStatus(api,
                   api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                            OrtMemTypeDefault, &memory_info),
                   "CreateCpuMemoryInfo", out_error)) {
    return nullptr;
  }
  session->memory_info_ = memory_info;

  // Enumerate inputs and outputs.
  auto describe = [&](bool is_input, size_t index, OnnxTensorInfo* info,
                      std::string* name) -> bool {
    char* raw_name = nullptr;
    OrtStatus* status =
        is_input
            ? api->SessionGetInputName(ort_session, index, allocator, &raw_name)
            : api->SessionGetOutputName(ort_session, index, allocator,
                                        &raw_name);
    if (!CheckStatus(api, status, "SessionGetInputName/OutputName",
                     out_error)) {
      return false;
    }
    *name = raw_name ? raw_name : "";
    if (raw_name) {
      allocator->Free(allocator, raw_name);
    }
    info->name = *name;

    OrtTypeInfo* type_info = nullptr;
    status =
        is_input
            ? api->SessionGetInputTypeInfo(ort_session, index, &type_info)
            : api->SessionGetOutputTypeInfo(ort_session, index, &type_info);
    if (!CheckStatus(api, status, "SessionGetInputTypeInfo/OutputTypeInfo",
                     out_error)) {
      return false;
    }
    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    bool ok =
        CheckStatus(api, api->CastTypeInfoToTensorInfo(type_info, &tensor_info),
                    "CastTypeInfoToTensorInfo", out_error);
    if (ok && tensor_info) {
      size_t dim_count = 0;
      ok = CheckStatus(api, api->GetDimensionsCount(tensor_info, &dim_count),
                       "GetDimensionsCount", out_error);
      if (ok) {
        info->shape.assign(dim_count, -1);
        ok = CheckStatus(
            api, api->GetDimensions(tensor_info, info->shape.data(), dim_count),
            "GetDimensions", out_error);
      }
      ONNXTensorElementDataType element_type =
          ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      if (ok) {
        ok = CheckStatus(api,
                         api->GetTensorElementType(tensor_info, &element_type),
                         "GetTensorElementType", out_error);
      }
      info->element_type = static_cast<int>(element_type);
    }
    api->ReleaseTypeInfo(type_info);
    return ok;
  };

  size_t input_count = 0;
  size_t output_count = 0;
  if (!CheckStatus(api, api->SessionGetInputCount(ort_session, &input_count),
                   "SessionGetInputCount", out_error) ||
      !CheckStatus(api, api->SessionGetOutputCount(ort_session, &output_count),
                   "SessionGetOutputCount", out_error)) {
    return nullptr;
  }
  session->inputs_.resize(input_count);
  session->input_names_.resize(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    if (!describe(true, i, &session->inputs_[i], &session->input_names_[i])) {
      return nullptr;
    }
    if (session->inputs_[i].element_type !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      if (out_error) {
        *out_error =
            fmt::format("input '{}' of {} is not float32 (element type {})",
                        session->inputs_[i].name, xe::path_to_utf8(model),
                        session->inputs_[i].element_type);
      }
      return nullptr;
    }
  }
  session->outputs_.resize(output_count);
  session->output_names_.resize(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    if (!describe(false, i, &session->outputs_[i],
                  &session->output_names_[i])) {
      return nullptr;
    }
  }

  std::string io_summary;
  for (const auto& input : session->inputs_) {
    io_summary +=
        fmt::format(" in {}{}", input.name, ShapeToString(input.shape));
  }
  for (const auto& output : session->outputs_) {
    io_summary +=
        fmt::format(" out {}{}", output.name, ShapeToString(output.shape));
  }
  XELOGI("NUI: loaded {} on {}{} in {:.0f} ms, {} threads;{}",
         xe::path_to_utf8(model.filename()), provider_name,
         adapter_name.empty() ? "" : fmt::format(" ({})", adapter_name),
         create_ms, threads, io_summary);
  return session;
}

OnnxSession::~OnnxSession() {
  if (!runtime_ || !runtime_->api()) {
    return;
  }
  const OrtApi* api = runtime_->api();
  if (session_) {
    api->ReleaseSession(static_cast<OrtSession*>(session_));
    session_ = nullptr;
  }
  if (memory_info_) {
    api->ReleaseMemoryInfo(static_cast<OrtMemoryInfo*>(memory_info_));
    memory_info_ = nullptr;
  }
  // allocator_ comes from GetAllocatorWithDefaultOptions and is owned by the
  // runtime.
  allocator_ = nullptr;
}

bool OnnxSession::Run(const std::vector<const float*>& input_data,
                      const std::vector<std::vector<int64_t>>& input_shapes,
                      std::vector<std::vector<float>>* out_outputs,
                      std::vector<std::vector<int64_t>>* out_shapes,
                      std::string* out_error) {
  if (!runtime_ || !runtime_->api() || !session_) {
    if (out_error) {
      *out_error = "session is not initialized";
    }
    return false;
  }
  const OrtApi* api = runtime_->api();
  const size_t input_count = inputs_.size();
  const size_t output_count = outputs_.size();
  if (input_data.size() != input_count || input_shapes.size() != input_count) {
    if (out_error) {
      *out_error =
          fmt::format("expected {} inputs, got {} buffers/{} shapes",
                      input_count, input_data.size(), input_shapes.size());
    }
    return false;
  }

  std::vector<OrtValue*> input_values(input_count, nullptr);
  std::vector<OrtValue*> output_values(output_count, nullptr);
  auto release_all = [&]() {
    for (auto& value : input_values) {
      if (value) {
        api->ReleaseValue(value);
        value = nullptr;
      }
    }
    for (auto& value : output_values) {
      if (value) {
        api->ReleaseValue(value);
        value = nullptr;
      }
    }
  };

  for (size_t i = 0; i < input_count; ++i) {
    size_t count = 0;
    if (!input_data[i] || !ElementCount(input_shapes[i], &count) ||
        count == 0) {
      if (out_error) {
        *out_error =
            fmt::format("input '{}' has no data or an invalid shape {}",
                        inputs_[i].name, ShapeToString(input_shapes[i]));
      }
      release_all();
      return false;
    }
    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        static_cast<OrtMemoryInfo*>(memory_info_),
        const_cast<float*>(input_data[i]), count * sizeof(float),
        input_shapes[i].data(), input_shapes[i].size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_values[i]);
    if (!CheckStatus(api, status, "CreateTensorWithDataAsOrtValue",
                     out_error)) {
      release_all();
      return false;
    }
  }

  std::vector<const char*> input_names(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    input_names[i] = input_names_[i].c_str();
  }
  std::vector<const char*> output_names(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    output_names[i] = output_names_[i].c_str();
  }

  OrtStatus* status =
      api->Run(static_cast<OrtSession*>(session_), nullptr, input_names.data(),
               input_values.data(), input_count, output_names.data(),
               output_count, output_values.data());
  if (!CheckStatus(api, status, "Run", out_error)) {
    release_all();
    return false;
  }

  out_outputs->resize(output_count);
  out_shapes->resize(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    OrtValue* value = output_values[i];
    std::vector<int64_t>& shape = (*out_shapes)[i];
    std::vector<float>& data = (*out_outputs)[i];
    if (!value) {
      shape.clear();
      data.clear();
      continue;
    }
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (!CheckStatus(api, api->GetTensorTypeAndShape(value, &info),
                     "GetTensorTypeAndShape", out_error)) {
      release_all();
      return false;
    }
    size_t dim_count = 0;
    ONNXTensorElementDataType element_type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    bool ok = CheckStatus(api, api->GetDimensionsCount(info, &dim_count),
                          "GetDimensionsCount", out_error);
    if (ok) {
      shape.assign(dim_count, 0);
      ok = CheckStatus(api, api->GetDimensions(info, shape.data(), dim_count),
                       "GetDimensions", out_error);
    }
    if (ok) {
      ok = CheckStatus(api, api->GetTensorElementType(info, &element_type),
                       "GetTensorElementType", out_error);
    }
    api->ReleaseTensorTypeAndShapeInfo(info);
    if (!ok) {
      release_all();
      return false;
    }
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      if (out_error) {
        *out_error =
            fmt::format("output '{}' is not float32 (element type {})",
                        outputs_[i].name, static_cast<int>(element_type));
      }
      release_all();
      return false;
    }
    size_t count = 0;
    ElementCount(shape, &count);
    void* tensor_data = nullptr;
    if (!CheckStatus(api, api->GetTensorMutableData(value, &tensor_data),
                     "GetTensorMutableData", out_error)) {
      release_all();
      return false;
    }
    data.resize(count);
    if (count) {
      std::memcpy(data.data(), tensor_data, count * sizeof(float));
    }
  }
  release_all();
  return true;
}

#else  // !XE_PLATFORM_WIN32

OnnxRuntime* OnnxRuntime::Get(const std::filesystem::path& explicit_dir,
                              const std::filesystem::path& storage_root) {
  std::lock_guard<std::mutex> lock(g_load_mutex);
  if (!g_load_attempted) {
    g_load_attempted = true;
    g_load_error =
        "ONNX Runtime loading is only implemented on Windows; webcam pose "
        "estimation is unavailable on this platform";
    XELOGW("NUI: {}", g_load_error);
  }
  return g_runtime;
}

std::unique_ptr<OnnxSession> OnnxSession::Create(
    OnnxRuntime* runtime, const std::filesystem::path& model,
    const Options& options, std::string* out_error) {
  if (out_error) {
    *out_error = "ONNX Runtime is unavailable on this platform";
  }
  return nullptr;
}

OnnxSession::~OnnxSession() = default;

bool OnnxSession::Run(const std::vector<const float*>& input_data,
                      const std::vector<std::vector<int64_t>>& input_shapes,
                      std::vector<std::vector<float>>* out_outputs,
                      std::vector<std::vector<int64_t>>* out_shapes,
                      std::string* out_error) {
  if (out_error) {
    *out_error = "ONNX Runtime is unavailable on this platform";
  }
  return false;
}

#endif  // XE_PLATFORM_WIN32

}  // namespace nui
}  // namespace xe
