# Adding a NUI library version

Kinect titles statically link the NUI runtime; Xenia replaces its public entry
points with host code (`src/xenia/kernel/nui/`). Each NUI library version needs
an entry in `src/xenia/kernel/nui/nui_signatures_builtin.cc` describing where
those functions are.

## 1. Find the version

Run the title once. The log prints the static libraries from the XEX header:

```
NUI HLE: default.xex links Kinect libraries: NUI=2.0.11775.6 ST=2.0.11775.5 NUISP=2.0.11775.4
NUI HLE: no signatures for NUI library 2.0.11775.6 in default.xex; ...
```

## 2. Locate the functions

The functions to find, and how to recognise them without symbols:

| Function | Anchors |
|---|---|
| `NuiInitialize` | calls `XamNuiGetDeviceStatus`, `XamNuiCameraTiltSetCallback`, `XMsgInProcessCall(0xFE, 0x2B004)`, `XamContentOpenFile("stexemplar", ...)`; string `"database.gmsodf"` |
| `NuiShutdown` | the only public function calling the runtime shutdown that closes the "Natal" thread |
| `NuiSkeletonGetNextFrame` | `KeWaitForSingleObject` followed by a copy of `0xAB0` bytes (`li r5, 0xAB0`); returns `0x8000000A` on timeout |
| `NuiSkeletonTrackingEnable` | `ObReferenceObjectByHandle` on the event handle; `0x800704DF` when already enabled |
| `NuiImageStreamOpen` | large; `ObCreateObject` with the NUI object type; string `"NuiApi: No open stream matches"` nearby |
| `NuiImageStreamGetNextFrame` / `ReleaseFrame` | `ObReferenceObjectByHandle` on the stream handle, list manipulation with a `0x68` entry stride |
| `NuiCameraElevationSetAngle` | calls `XamNuiCameraElevationSetAngle` (retail) or issues a Detroit request (2010) |
| `NuiTransformSmooth` | reads the five float parameters, defaults `0.5, 0.5, 0.5, 0.05, 0.04` |

Tools:

- For builds with a linker map (Project Milo): `tools/nui/milo_sym.py --find NAME`
  gives the address; `tools/nui/milo_dis.py --func NAME 100` disassembles.
- For retail builds: run the title once with `--nui=true
  --nui_dump_module_image=<dir>`; the hook installer writes
  `<module>_<base>.bin` (the loaded image) and a `.json` sidecar (base
  address, code page ranges, static libraries). Load the `.bin` into Ghidra
  (PowerPC 32 big-endian, at the base address from the sidecar) and use the
  anchors above. A Ghidra FunctionID
  library built from the Milo image finds most nuiapi functions in later
  versions because the library is not LTCG-compiled.

## 3. Generate signatures

`tools/nui/gen_signatures.py` emits the first N instruction words of a function
with masks over relocation-sensitive fields (branch displacements, `lis`/`addi`
pairs), as a C++ initializer:

```
python tools/nui/gen_signatures.py --exe image.exe --map image.map --words 16 \
    NuiSkeletonGetNextFrame NuiImageStreamOpen ...
```

Add `--verify other.exe --map2 other.map` to prove each signature matches
exactly once in another build of the same library (the two Milo builds are a
ready-made pair). Signatures that match more than once need more words; ones
that match zero times in the other build are relocation-sensitive and need a
wider mask.

## 4. Add the entry

In `nui_signatures_builtin.cc` add a `NuiLibrarySignatures` with:

- `library = "NUI"`, `version = "<major.minor.build.qfe>"`, or `module_name`
  for a specific build;
- the quirks of that generation (timestamp unit, floor plane unit, HRESULT
  values seen in the function epilogues);
- one `NuiFunctionSignature` per function with `mode = "hle"` (emulated),
  `"tripwire"` (log and fail; for features not emulated yet such as speech) or
  `"native"` (left alone).

`fixed_address` may be set for a known build; the words are still checked
before hooking. Every `hle` function must resolve, otherwise the whole version
is treated as unsupported and the title sees no sensor (a half-replaced runtime
crashes in ways that are hard to debug).

## 5. Verify

Run with `--nui=true --nui_trace=true --log_level=2` and check:

- `NUI HLE: NUI <version> in <module>: N functions hooked`
- `NuiHLE: first skeleton frame delivered` / `first image frame delivered`
- no `undefined extern call` lines mentioning `Nui`, `PsCam`, `Detroit`, `Mca`.
