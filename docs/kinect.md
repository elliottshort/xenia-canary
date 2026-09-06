# Kinect (NUI) emulation

Xenia can emulate a Kinect for Xbox 360 sensor so that Kinect titles run without
the hardware. The data the title sees (skeletons, depth/player-index images,
colour images) comes from a selectable host-side *source*.

Status: work in progress. Skeleton and depth streams work with the virtual
(gamepad-driven) source for titles whose NUI runtime version is known to the
emulator; webcam-based tracking is being built on the same interface.

## Enabling

```
xenia_canary_netplay.exe game.xex --nui=true --nui_source=virtual
```

Everything is off unless `nui=true` (config section `[NUI]`). With it on, the
title is told a sensor is connected and ready, and the public NUI runtime
functions statically linked into the title are replaced by host code when the
title's NUI library version has a signature entry (see the log line starting
with `NUI HLE:`). Titles with an unknown NUI version see no sensor.

## Sources (`nui_source`)

- `virtual` (default): a standing player driven by the gamepad of user slot 0.
  The keyboard driver feeds the same gamepad, so keyboard bindings work too
  (`--keyboard_mode=1`).

  | Control | Effect |
  |---|---|
  | left / right stick | left / right hand (sideways, up-down) |
  | left / right trigger | push the hand towards the sensor |
  | d-pad | walk (left/right strafe, up towards the sensor, down away) |
  | A | "engage": right hand held out in front with a gentle sweep |
  | B | hands at the sides |
  | X | jump |
  | Y | T-pose |
  | LB / RB | lean left / right |
  | Back | crouch while held |
  | Start | toggle a second, idle player standing to the right |

  The source also renders a depth/player-index image from the skeleton, so
  silhouette effects and hand-cursor libraries that read depth get data.
- `none`: a working sensor that never sees anybody.
- `webcam`: a real player in front of a USB/laptop camera, tracked with
  host-side pose estimation. See [Webcam](#webcam) below.
- `playback`: replays a `.nuirec` recording made with `nui_record_path`. See
  [Recording and playback](#recording-and-playback) below.

## Other settings

| cvar | Meaning |
|---|---|
| `nui_camera_height` | Sensor height above the floor in metres (floor plane); 0 = unknown |
| `nui_tilt_mode` | `simulate` (default) rotates the virtual sensor when the title tilts it; `ignore` only reports the angle |
| `nui_max_players` | Fully tracked skeletons, 1 or 2 |
| `nui_smoothing_override` and `nui_smoothing`, `nui_correction`, `nui_prediction`, `nui_jitter_radius`, `nui_max_deviation_radius` | Replace the title's NuiTransformSmooth parameters |
| `nui_auto_bind_user0` | Bind the first tracked player to the signed-in user 0 |
| `nui_trace` | Log every call into the emulated NUI runtime |
| `nui_log_stats` | Log frame statistics every 5 seconds |
| `nui_hle` | Set to false to leave the title's whole NUI runtime untouched (debugging only) |

`nui_hle` is not the way to make room for a title-specific hook layer: the
generic layer already cedes a whole group of functions to whoever hooked one
of them first (see "Composing with title-specific hooks" in
`docs/nui/architecture.md`), so both can be on at the same time. Turning it
off removes every generic replacement — camera tilt, image streams, the
lifecycle — and is only useful when comparing against the title's own
runtime.

## Webcam

`--nui_source=webcam` turns a plain RGB camera into a Kinect: the image is
captured with Media Foundation, MediaPipe BlazePose (running on ONNX Runtime,
DirectML on the GPU or CPU) finds up to two people and their 33 landmarks per
frame, and the emulator synthesizes the 20-joint Kinect skeletons, the
player-index/depth image and the colour stream from that. Absolute distance
is not observable with one camera, so it is estimated from the size of the
body in the image (see calibration below).

### Setup

1. Build the emulator (Windows only for now; the capture and DirectML paths
   are Windows APIs).
2. Install the inference runtime and the models next to the emulator:

   ```
   python tools/nui/setup_nui.py --exe build-nui\bin\Windows\Release\xenia_canary_netplay.exe
   ```

   The script needs only the Python standard library. It downloads the
   ONNX Runtime DirectML build and DirectML from NuGet and the BlazePose ONNX
   models, verifies every file against a pinned SHA-256 and writes
   `models.json`. Files go into the `nui` folder of the emulator's storage
   root (the executable folder on Windows):

   ```
   <exe folder>\nui\runtime\onnxruntime.dll
   <exe folder>\nui\runtime\DirectML.dll
   <exe folder>\nui\models\pose_detection.onnx
   <exe folder>\nui\models\pose_landmark_{lite,full,heavy}.onnx
   <exe folder>\nui\models\models.json
   ```

   `--dest DIR` installs somewhere else (then point `nui_runtime_path` and
   `nui_model_path` at `DIR\runtime` and `DIR\models`); `--skip-download`
   only verifies an existing installation; `--runtime-only` / `--models-only`
   split the job.
3. Check the camera and the models with the demo tool (below), then run a
   title:

   ```
   xenia_canary_netplay.exe game.xex --nui=true --nui_source=webcam
   ```

Without the runtime or the models the source still starts in a camera-only
mode: the colour stream works, but no player is ever tracked. The log says
why (`NUI webcam: pose estimation unavailable: ...`).

The runtime is loaded from our own folder with an absolute path; the
`onnxruntime.dll` / `DirectML.dll` that Windows ships in `System32` (an old
WinML build) is never used.

### Cvars

| cvar | Default | Meaning |
|---|---|---|
| `nui_camera` | (first) | Camera to use: an index (`0`, `1`, ...) or part of the device name (`--nui_camera="USB Camera"`) |
| `nui_capture_width`, `nui_capture_height`, `nui_capture_fps` | 640, 480, 30 | Requested capture mode; the nearest mode the driver offers is used |
| `nui_camera_hfov` | 70 | Horizontal field of view of the camera in degrees. Needed for distances; typical laptop/USB webcams are 60-78, wide-angle ones 90+ |
| `nui_camera_mirrored` | false | Set when the driver already mirrors the image (raise your right hand: it must appear on the right of the preview) |
| `nui_camera_height` | 1.0 | Height of the camera above the floor in metres (floor plane); 0 = unknown |
| `nui_camera_pitch` | 0 | Physical pitch of the camera in degrees, positive looking up |
| `nui_user_scale` | 1.0 | Multiplies the estimated distance to the player |
| `nui_model_quality` | auto | `lite`, `full` or `heavy` landmark model; `auto` is `full` on DirectML and `lite` on CPU |
| `nui_execution_provider` | auto | `dml` (GPU), `cpu`, or `auto` (DirectML when available) |
| `nui_inference_threads` | 0 | CPU threads for inference (0 = automatic) |
| `nui_model_path`, `nui_runtime_path` | | Override the `nui\models` and `nui\runtime` folders |
| `nui_segmentation` | true | Use the model's person segmentation for the player silhouette instead of the capsule render of the skeleton |
| `nui_max_players` | 2 | People processed with the landmark model (1 or 2) |

### Calibration tips

- Camera placement: like a Kinect, at 0.6-1.2 m height, pointing at the
  play area, with the whole body (head to feet) visible at 1.5-3 m. Set
  `nui_camera_height` and `nui_camera_pitch` to what you measured; the
  floor plane and the "up" vector that titles use come from them.
- Field of view: the distance estimate scales with `tan(hfov/2)`, so a wrong
  `nui_camera_hfov` makes you appear nearer or farther than you are. Look the
  value up for your camera (or measure: stand at a known distance, the
  demo's `--source` mode prints the hip centre in metres) and fine-tune with
  `nui_user_scale` until 2 m reads as 2 m.
- Mirroring: Kinect images are mirror views. The pipeline flips the webcam
  image once before tracking and maps the pose model's visual left/right to
  the player's anatomical sides accordingly. If the driver already mirrors,
  set `nui_camera_mirrored=true`; with the wrong setting the whole skeleton
  and the images are mirrored (a raised right hand shows on the wrong side
  of the preview), which is the symptom to look for.
- Lighting and background: even, frontal light and a plain background help
  the detector; strong backlight (a window behind you) loses the legs and
  the segmentation.
- Performance: DirectML with the `full` model runs at camera rate on any
  recent GPU; on CPU use `lite` (`auto` does this). If the emulator's own
  rendering fights for the GPU, try `--nui_execution_provider=cpu` with
  `--nui_model_quality=lite`.

### Demo tool

`xenia-nui-webcam-demo.exe` (built with the emulator, Windows only) exercises
each stage without running a title. It accepts every `nui_*` cvar in the
`--name=value` form.

```
xenia-nui-webcam-demo --list
xenia-nui-webcam-demo --capture [N] [--out DIR]
xenia-nui-webcam-demo --pose [SECONDS] [--out DIR]
xenia-nui-webcam-demo --source [SECONDS] [--out DIR]
```

- `--list` prints the cameras Media Foundation sees, with their index and
  device path.
- `--capture` opens the camera with the `nui_camera*` settings, grabs `N`
  frames (default 30) as `frame_NNN.png` and prints the negotiated mode,
  the achieved frame rate and the mean brightness (a black image means the
  camera is blocked or in use elsewhere).
- `--pose` runs the pose estimator live for `SECONDS` (default 10) and
  prints, once per second, the frame rate, backend, model, person count and
  the first person's nose / wrists (normalized image coordinates) and hip
  depth; `pose_NNN.png` shows the landmarks and bones drawn over the
  (mirrored) camera image.
- `--source` drives `WebcamNuiSource` through the same interface the
  emulator uses, polled at 30 Hz, and prints the bodies with tracking ids,
  hip centre, head and hands in metres, joint states (`T` tracked, `I`
  inferred) and clipping flags every second. `depth_NNN.png` is the
  synthesized depth image (bright = near) with the player mask tinted per
  player; `color_NNN.png` is the 640x480 colour stream.

Exit codes: 0 success, 1 usage, 2 camera error, 3 model/runtime error (in
`--source` mode the camera-only fallback also exits with 3 after running).

### Troubleshooting

- "cannot open camera" / access denied: Windows privacy settings. Settings >
  Privacy & security > Camera: "Camera access" and "Let desktop apps access
  your camera" must be on.
- The camera opens but no frames arrive (timeouts, black `frame_000.png`):
  another application (browser tab, Teams, OBS, the vendor's camera app)
  holds the camera; many UVC cameras serve one client. Close it. Some
  cameras also need a second or two after opening before the first frame.
- "no cameras found": the device is disabled in Device Manager, or it is a
  virtual camera that does not register as a Media Foundation video capture
  source.
- "onnxruntime.dll not found" / "pose_detection.onnx not found": run
  `tools/nui/setup_nui.py` (see Setup) or check `nui_runtime_path` /
  `nui_model_path`. The message lists the folders that were searched.
- DirectML not used (the status says `CPU`): `DirectML.dll` is missing next
  to `onnxruntime.dll`, the runtime is not the DirectML build, or the GPU
  driver is too old. `--nui_execution_provider=dml` forces the attempt and
  logs the reason for the fallback. CPU works but use `nui_model_quality=lite`.
- Nobody is tracked although the preview shows you: stand back so the whole
  body is visible, improve the light, or lower the capture resolution
  (`--nui_capture_width=640 --nui_capture_height=480`); the detector wants
  the person to fill roughly a third of the frame.
- Hands are swapped or you move the wrong way: toggle `nui_camera_mirrored`.
- You appear too near or too far: fix `nui_camera_hfov`, then
  `nui_user_scale` (see Calibration tips).
- The colour image is cropped or has black bands top/bottom: expected. The
  Kinect colour camera has a 62 degree field of view and a 4:3 aspect; the
  webcam image is remapped into it, so a narrower or 16:9 camera cannot
  fill the whole frame.

## Recording and playback

Everything the emulated sensor publishes can be recorded to a `.nuirec` file
and played back later with `--nui_source=playback`, whatever source produced
it (virtual, webcam, ...). Recordings capture the source frame behind each
published 30 Hz tick: the bodies with their source-local person keys, the
floor plane and gravity vector, and - when the title had the corresponding
streams open so the source produced them - the 320x240 player mask, the
320x240 depth image and the 640x480 colour image.

### Recording

```
xenia_canary_netplay.exe game.xex --nui=true --nui_source=webcam --nui_record_path=C:\rec\session.nuirec
```

| cvar | Meaning |
|---|---|
| `nui_record_path` | File to write. Empty (default) = no recording |

The cvar is polled once per second, so it can be set or cleared while a
title runs (through the settings UI or the config file): setting it starts a
new file, changing it starts another, clearing it closes the current one. An
existing file at that path is overwritten. Frames go through a 64-frame ring
drained by a low-priority thread, so recording does not stall the sensor; if
the disk cannot keep up, frames are dropped and a warning is logged.

Which optional sections a file contains is decided by the first frame
written to it: a file started before the title opened its depth or colour
streams has skeletons only. Start (or restart) the recording after the
title has initialized the sensor to capture the images too. The header also
stores `nui_camera_height` and `nui_camera_hfov` at the time of recording.

Depth and colour planes are zstd-compressed and the player mask is
run-length coded; a skeleton-only recording is about 1 KB per frame, depth
adds roughly 20-60 KB per frame and colour a few hundred KB per frame.

### Playback

```
xenia_canary_netplay.exe game.xex --nui=true --nui_source=playback --nui_playback_path=C:\rec\session.nuirec
```

| cvar | Default | Meaning |
|---|---|---|
| `nui_playback_path` | | The `.nuirec` file to play. The source fails to start (the title sees no sensor) when it is missing or unreadable; the log says why |
| `nui_playback_loop` | true | Restart from the first frame when the end is reached. With `false` the last frame stays |
| `nui_playback_realtime` | true | Deliver frames by their recorded timestamps (frames are skipped if the emulator falls behind). With `false` every 30 Hz tick consumes exactly one recorded frame, which is deterministic and useful for reproducing a bug |

The recorded person keys become the tracking ids the title sees through the
same mapping the live sources use, so a person tracked across the whole
recording keeps one id, and the sensor tilt, seated mode and user bindings
work as with a live source. Frames are decoded ahead of time on a reader
thread; the playback clock starts at the first frame the sensor delivers.

### File format

`src/xenia/nui/nui_recorder.h` documents the layout: a 24-byte header
(`XNUI`, version 1, section flags, frame rate, camera height, horizontal
field of view) followed by frames until the end of the file. Each frame holds
a 64-bit timestamp in microseconds, up to six bodies serialized field by
field, the floor plane and gravity vector, and the optional sections declared
in the header. `NuirecWriter` / `NuirecReader` in the same header read and
write the format and are covered by `src/xenia/nui/testing/nuirec_test.cc`.

## How it works

See `docs/nui/architecture.md`. In short: the NUI runtime (skeletal tracking,
speech, hand cursor) is statically linked into every Kinect title and only
reaches the console through a handful of kernel imports for raw sensor
transfers. Instead of emulating the PrimeSense camera, the emulator replaces the
runtime's public API functions (`NuiInitialize`, `NuiSkeletonGetNextFrame`,
`NuiImageStreamGetNextFrame`, ...) inside the title with host implementations
served by `xe::nui::NuiSystem`, a 30 Hz device model fed by the selected source.
Functions are located per NUI library version with instruction signatures
(`src/xenia/kernel/nui/nui_signatures_builtin.cc`); see
`docs/nui/adding_signatures.md` for adding a version. Signatures are grouped
into ownership sets that are replaced all-or-nothing, so a title-specific
hook layer can own part of the runtime (for instance Project Milo's own
frame functions) while the emulator keeps the rest.
