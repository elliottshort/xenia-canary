# ONNX Runtime (DirectML) headers

Vendored C API headers only; nothing links against ONNX Runtime at build time.
`dml_provider_factory.h` includes `<DirectML.h>` and `<d3d12.h>` from the
Windows SDK, so no DirectML headers are vendored here.
The emulator loads `onnxruntime.dll` (with `DirectML.dll` next to it) at
runtime from `<storage_root>/nui/runtime` (see `src/xenia/nui/onnx_runtime.h`).

| File | Source package | Version |
|---|---|---|
| `include/onnxruntime_c_api.h`, `onnxruntime_ep_c_api.h`, `dml_provider_factory.h`, `onnxruntime_session_options_config_keys.h`, `onnxruntime_run_options_config_keys.h` | Microsoft.ML.OnnxRuntime.DirectML (NuGet) | 1.24.4 (`ORT_API_VERSION` 24, commit 2d924974) |
| `LICENSE`, `ThirdPartyNotices.txt` | Microsoft.ML.OnnxRuntime.DirectML | MIT |

Runtime binaries (not in the repository):

- `runtimes/win-x64/native/onnxruntime.dll` from
  https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.24.4
- `bin/x64-win/DirectML.dll` from
  https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4

Both go into `<storage_root>/nui/runtime/` (`tools/nui/setup_nui.py` downloads,
verifies and installs them together with the pose models). `onnxruntime.dll` needs the
VC++ 2015-2022 x64 redistributable (`MSVCP140.dll`, `MSVCP140_1.dll`,
`VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`).
