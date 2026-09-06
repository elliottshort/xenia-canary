# NUI library 2.0.11427.0 (XDK 11427 beta, Project Milo, May 2010)

Source of truth: `xbe/Project Milo (May 17 2010)/miloReleaseLIB.{map,pdb,exe}` and the
disassembly notes in the Milo session scratchpad (`nuipinit.txt`, `nuirt.txt`).
Static libraries in the XEX: `NUI 2.0.11427.0`, `ST 2.0.11427.0`, `NUISP 2.0.11427.0`;
Milo's own hand tracking is the `spock` library (not `NUIHNDL`).

## Public API present

`NuiInitialize` (16-byte thunk into `NuipInitialize`), `NuiShutdown`,
`NuiSkeletonTrackingEnable/Disable`, `NuiSkeletonGetNextFrame`, `NuiTransformSmooth`,
`NuiImageStreamOpen/GetNextFrame/ReleaseFrame`, `NuiImageGetColorPixelCoordinatesFromDepthPixel`,
`NuiCameraElevationSetAngle/GetAngle`, `NuiCameraGetNormalToGravity`; speech in `nuispeech`
(`NuiSpeechEnable`, `NuiSpeechLoadGrammarFromMemory`, `NuiSpeechStartRecognition`, ...).
Absent: `NuiSetFrameEndEvent`, `NuiSkeletonSetTrackedSkeletons`,
`NuiImageStreamSetImageFrameFlags`, `NuiTransformDepthImageToSkeleton` (the SkeletonToDepth
transform is an inline in `identityrecognition.obj`).

## Calling conventions and quirks

- `NuiInitialize(DWORD flags, DWORD thread_index)`: the thunk passes `r5 = 8` and
  `r6 = g_pszDatabaseName`. `NuipInitialize` rejects `flags & ~0x8000003B` and `flags == 0`
  with `E_INVALIDARG (0x80070057)`; `thread_index == -1` means 5, else it must be `< 6`
  (hardware thread for the "Natal" worker, `KeSetAffinityThread(1 << index)`). Skeletal
  tracking (ST) is set up when `flags & 0x80000009`. Milo passes `0x80000032`.
  Flow: `RtlEnterCriticalSection(NuipApiLock)` -> memset `NuipRuntimeState` (0x2010 bytes) ->
  resolve `PsCamDeviceRequestProxy` through the `KeDebugMonitorData+0x18` callback (id 0x9A),
  else the static import -> `NtCreateEvent` -> `NuipInitializeTransferBuffers` ->
  `ExCreateThread(NuipThreadRoutine, flags 0x81)` + affinity -> `D3DDevice_NuiInitialize(&state->d3d_device)`
  (the title's D3D lib hands over its device) -> `NuipAllocateSTTextures`, `ST_Initialize`,
  depth-segmentation streams for ST, `CDRP_Initialize`, LUT textures built with
  `XGSetTextureHeaderEx`/`XGOffsetResourceAddress`.
- `NuiSkeletonTrackingEnable(HANDLE event, DWORD flags)`: requires `NuipRuntimeState.flags & 0x8`
  else `0x83010005`; only flag bit 0 (`SUPPRESS_NO_FRAME_DATA`) is accepted, others give
  `E_INVALIDARG`; already enabled -> `0x800704DF`; the event handle is referenced with
  `ObReferenceObjectByHandle(ExEventObjectType)` and stored at `state+0x4F4`.
- `NuiSkeletonGetNextFrame(DWORD ms, NUI_SKELETON_FRAME*)`: null frame -> `E_INVALIDARG`;
  tracking not enabled (`state+0x4E0 == 0`) -> `0x83010002`; device not ready
  (`state+0x54 == 0`) -> `0x8007048F`; `ms` clamped to 8000 unless `-1`; waits on the
  `KEVENT` at `state+0x4E4`; timeout -> `0x8000000A (E_PENDING)`; on success copies 0xAB0 bytes
  from the ring slot (`state+0x510 + index*0xAD0`, slot header 0x20 bytes then the frame) and
  resets both events; no data -> `0x83010001`; a slot state of 2 -> `0x8301000B`.
  Timestamps are 100 ns ticks; `vFloorClipPlane.w` is in millimetres (the title scales by 0.001).
- `NuiImageStreamGetNextFrame(HANDLE, DWORD ms, const NUI_IMAGE_FRAME**)`: the handle is a
  title-defined object (`ObCreateObject(NuiObjectType)` / `ObReferenceObjectByHandle(NuiObjectType)`);
  the stream object starts with a `KEVENT`; `+0x14` flags (`0x10000` = suppress no-frame-data),
  `+0x18` entry count (= frame limit + 2), `+0x1C` entry array, `+0x24` free list, `+0x2C` held
  count, `+0x30` held list. Entries are 0x68 bytes: `LIST_ENTRY` at 0, the title-visible
  `NUI_IMAGE_FRAME` at +8 (timestamp +8, frame number +0x10, image type +0x14, resolution +0x18,
  `pFrameTexture` at +0x1C, i.e. frame+0x14), pixel pointer at +0x5C, state at +0x60
  (1 = held). Frame limit exhausted -> `0x83010004`; not ready -> `0x8007048F`; timeout ->
  `E_PENDING`.
- `NuiImageStreamReleaseFrame`: validates `(frame - 8 - entries) / 0x68 < count` and state == 1,
  else `E_INVALIDARG`.
- `NuiCameraElevationSetAngle(LONG)`: range -28..28 checked (`0x1C`/`-0x1C`), then a Detroit
  control request (`code 0x00033148`, angle in the int16 at +0x18); status query code
  `0x000F0410`.
- Titles read image frames with `D3DResource_BlockUntilNotBusy(pFrameTexture)` (reads `Fence`
  at header +8) and `D3DTexture_LockRect(pFrameTexture, 0, &rect, NULL, 0x10)`
  (`D3D::Lock2DSurface`, derives the pointer/pitch from the fetch constant).
- Kernel imports used by the runtime: `PsCamDeviceRequest(_NUICAM_REQUEST*)` (opcodes: 0 query
  state, 1 open, 2 close, 5 transfer, 6/7 control, 0x0B/0x0E registration params, 0x0D version,
  0x0F blanking, 0x12 start, 0x19 vblank info), `DetroitDeviceRequest(DETROIT_CONTROL_REQUEST*)`;
  XAM: `XamUserNuiBind/Unbind/GetUserIndex/GetEnrollmentIndex`. No `McaDeviceRequest`;
  speech audio goes through `XamVoice*`/`XAudio*`.

## Emulation notes

Signatures for all 13 public functions are in `nui_signatures_builtin.cc` (16 words each,
unique in both Milo builds). Quirks: `timestamp_100ns`, `floor_plane_millimetres`,
`init_flags_mask 0x8000003B`, `init_skeleton_flags 0x80000009`, `hr_wait_timeout 0x8000000A`,
`hr_stream_not_enabled 0x83010002`, `hr_frame_limit_exceeded 0x83010004`,
`hr_feature_not_initialized 0x83010005`.
Milo runs in seated "NUI Hands" mode: it opens a depth stream and derives hand positions from
the depth image in its `spock` library (`NuiHands1..4`), so the depth stream must contain a
plausible player silhouette.
