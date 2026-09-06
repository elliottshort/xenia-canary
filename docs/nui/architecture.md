# Kinect (NUI) emulation architecture

## Why the API layer

On Xbox 360 a Kinect title links the whole NUI runtime statically: `NUI`
(public API and camera plumbing), `ST` (skeletal tracking, partly on the GPU),
`NUISP` (speech recognition) and later `NUIHNDL` (hand cursor). The console only
provides raw sensor transfers (`PsCamDeviceRequest`), tilt/accelerometer
(`DetroitDeviceRequest`, `XamNuiCamera*`), mic-array audio
(`XamVoiceGetMicArrayAudioEx`), and a tracking database on the disc
(`game:\Database.xmplr` through `XamContentOpenFile`).

Feeding synthetic depth into the title's own tracker would require emulating the
camera protocol and running a GPU-assisted classifier trained on real depth. The
emulator instead replaces the runtime's public functions inside the title and
serves them from a host device model. The sensor-level path stays a possible
fallback for titles that bypass the public API.

## Layers

```
 title code  --bl-->  NuiSkeletonGetNextFrame (rewritten to sc 2; blr)
                          |  src/xenia/kernel/nui/nui_hle_handlers.cc
                          v
              xe::nui::NuiSystem  (src/xenia/nui/nui_system.cc)
                          ^  30 Hz pacer thread, frame numbers, tracking ids,
                          |  slots, user bindings, tilt model, Holt smoothing
                     NuiSource  (virtual | playback | webcam | none)
```

- `src/xenia/kernel/nui/nui_hle.cc`: at title load (after title patches and
  title hooks, before precompilation) reads the XEX static-library table,
  selects a signature set for the NUI version, resolves every function
  (fixed address or masked pattern search over code pages) and installs the
  hooks with `XexModule::InstallExternHook` (`sc 2; blr` at the entry, host
  handler via `GuestFunction::SetupExtern`). Functions another hook layer
  already owns are left alone, together with the rest of their ownership set
  (see below).
- `nui_hle_handlers.cc`: the handlers. Skeleton frames are converted to the
  big-endian `NUI_SKELETON_FRAME`; image streams get guest memory allocated
  once per (type, resolution, frame limit): frame descriptors, D3D texture
  headers (linear `k_16` / `k_8_8_8_8` fetch constants pointing at physical
  pixel buffers) and the pixel slots. Stream handles are real kernel objects
  (an `XEvent` per stream) so `NtClose` works. Guest events passed to
  `NuiSkeletonTrackingEnable`, `NuiImageStreamOpen` and `NuiSetFrameEndEvent`
  are retained and set from the pacer thread after every publish.
- `src/xenia/nui/nui_system.cc`: the device model. The pacer thread ticks at
  30 Hz in guest time, pulls the newest source frame, maps source "person
  keys" to stable tracking ids and skeleton slots, applies the tracked/
  position-only policy (title-selected or nearest-first), seated mode, the
  floor plane and gravity vector for the configured height and simulated
  tilt, and publishes the frame under a mutex/condition variable that guest
  threads wait on with a bounded timeout. Image frames are derived on request
  from the latest source frame (decimation/upsampling from 320x240 / 640x480).
- `src/xenia/nui/nui_source.h`: what a source produces per frame: up to six
  bodies in sensor space, a player mask and depth image at 320x240, and a
  colour image at 640x480. `sources/virtual_nui_source.cc` synthesizes a
  standing player from the gamepad and renders depth with
  `depth_synthesizer.cc` (capsules per bone). The webcam source (pose
  estimation on RGB) and recording playback plug into the same interface.

## Composing with title-specific hooks

A title-specific hook layer (`src/xenia/kernel/title_hooks/*`) may replace
some NUI functions of one build with its own implementation, for instance
Project Milo's `NuiImageStreamGetNextFrame` and `NuiSkeletonGetNextFrame`.
`KernelState::FinishLoadingUserModule` runs `ApplyTitleHooks` first and
`nui::AttachNuiHle` immediately after, so those entries are already rewritten
to the `sc 2` trampoline when the generic layer resolves its signatures.

Replacing only *some* functions of the runtime is not safe: the functions of
the NUI API share state that a foreign implementation cannot see. A title
that gets our stream handle from `NuiImageStreamOpen` and passes it to
somebody else's `NuiImageStreamGetNextFrame` gets neither model. Every
signature entry therefore names an ownership set (`xe::nui::NuiHookSet` in
`src/xenia/nui/nui_hook_sets.h`), and a set is hooked all-or-nothing:

| set | functions | shared state |
| --- | --- | --- |
| `kLifecycle` | `NuiInitialize`, `NuiShutdown` | the device model's initialized flag |
| `kSkeleton` | `NuiSkeletonTrackingEnable/Disable`, `NuiSkeletonGetNextFrame`, `NuiSkeletonSetTrackedSkeletons`, `NuiTransformSmooth` | tracking flags, frame cursor, smoother |
| `kImage` | `NuiImageStreamOpen`, `NuiImageStreamGetNextFrame`, `NuiImageStreamReleaseFrame`, `NuiImageStream{Set,Get}ImageFrameFlags`, `NuiSetFrameEndEvent` | stream handles and their guest frame pools |
| `kCamera` | `NuiCameraElevation{Set,Get}Angle`, `NuiCameraGetNormalToGravity` | the simulated tilt motor |
| `kTransform` | `NuiImageGetColorPixelCoordinatesFromDepthPixel` | none (pure), but must agree with the image set |
| `kNone` | tripwire stubs and other standalone entries | none; decided one by one |

`PlanNuiHooks` (pure, unit tested in `src/xenia/nui/testing/nui_hook_set_test.cc`)
decides per set:

- any function of the set is already hooked by somebody else (its first word
  is `sc 2`) → **cede** the whole set, hook none of it, log which function is
  already owned;
- otherwise a required function of the set is missing (or has no host
  handler) → **leave the set native**; only that set, the others still get
  hooked;
- otherwise **hook** every function of the set.

Only when nothing at all could be replaced and nothing was ceded is the
sensor reported absent (`SetDevicePresent(false)`); a title whose runtime is
partly served by title hooks keeps a present sensor.

Pattern-resolved signatures no longer match a function whose entry was
rewritten, so `nui_hle.cc` also searches for the trampoline followed by the
rest of the signature (`InstallExternHook` overwrites exactly 16 bytes) to
tell "already owned" from "not found".

Whoever owns a ceded set reads skeletons or images straight from
`NuiSystem` without going through our `NuiInitialize` or
`NuiImageStreamOpen`, so the installer calls
`NuiSystem::SetExternalConsumers(skeletons, images)` when it cedes the
skeleton or image set. `BuildDeviceState` then forces `skeleton_tracking`,
`want_depth`, `want_player_mask` and `want_color` on, so the source
synthesizes what a foreign handler may read. Nothing changes when no set is
ceded, and `ResetGuestState` clears the flags when the title is terminated.

## Conventions

Skeleton space: metres, right-handed, origin at the sensor, +X to the sensor's
left, +Y up, +Z away from the sensor. Depth/colour images are mirror views.
Depth pixels are `(millimetres << 3) | player_index`. The 2010 (XDK 11427)
runtime reports timestamps in 100 ns ticks and the floor plane distance in
millimetres; retail runtimes use milliseconds and metres (`NuiHleQuirks`).

## Threads and locks

The pacer thread touches only host state and host events; it never calls into
the kernel or writes guest memory. Guest threads run the handlers, copy frames
into guest memory and block on the publish condition variable. Lock order is
kernel global lock, then `NuiSystem` state, then the handler state.
