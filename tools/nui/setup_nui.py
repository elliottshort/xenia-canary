#!/usr/bin/env python3
"""Installs the runtime files for xenia's Kinect (NUI) webcam source.

Downloads ONNX Runtime (DirectML build) and DirectML from NuGet, extracts
onnxruntime.dll and DirectML.dll into <dest>/runtime, downloads the BlazePose
ONNX models into <dest>/models, verifies every file against a pinned SHA-256
and writes <dest>/models/models.json.

<dest> is the "nui" folder inside the emulator's storage root, which is the
folder of the emulator executable on Windows:

    python tools/nui/setup_nui.py --exe path/to/xenia_canary_netplay.exe
    python tools/nui/setup_nui.py --dest path/to/xenia/nui

Without --exe/--dest the build folders next to this repository (build/ and
build-nui/) are tried; the script never falls back to a machine-specific path.

The script is idempotent: files whose hash already matches are left alone.
Use --skip-download to only verify what is installed. Exit codes: 0 all files
installed and verified, 1 a file is missing or corrupt (or no destination
found), 2 everything installed but a substitute was used (the older 2021
detector stands in for pose_detection.onnx, see --detector-url). Downloaded
NuGet packages are kept in --cache-dir so a re-run after a failed extraction
does not fetch 200 MB again.

Only the Python standard library is required.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

# Without --exe/--dest the emulator folder is looked for relative to the
# repository (the CMake build trees), never at a machine-specific path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_EXE_DIR_CANDIDATES = [
    os.path.join(REPO_ROOT, "build", "bin", "Windows", "Release"),
    os.path.join(REPO_ROOT, "build-nui", "bin", "Windows", "Release"),
]

ORT_VERSION = "1.24.4"
DIRECTML_VERSION = "1.15.4"
ORT_API_VERSION = 24

# NuGet packages: (id, version, sha256 of the .nupkg, size in bytes).
NUGET_FLATCONTAINER = "https://api.nuget.org/v3-flatcontainer/{id}/{ver}/{id}.{ver}.nupkg"
PACKAGES = {
    "onnxruntime": {
        "id": "microsoft.ml.onnxruntime.directml",
        "version": ORT_VERSION,
        "sha256": "57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681",
        "size": 12458649,
    },
    "directml": {
        "id": "microsoft.ai.directml",
        "version": DIRECTML_VERSION,
        "sha256": "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
        "size": 202292617,
    },
}

# Files extracted from the packages into <dest>/runtime. The DirectML license
# must accompany DirectML.dll (Microsoft Software License Terms, section 1a);
# the ONNX Runtime license is MIT.
RUNTIME_FILES = [
    {
        "file": "onnxruntime.dll",
        "package": "onnxruntime",
        "member": "runtimes/win-x64/native/onnxruntime.dll",
        "sha256": "e7eedec6a6f26dc39dc948276a75ef6d2bee3fff944d874ceed0bbd3b97bff40",
        "size": 17328152,
    },
    {
        "file": "LICENSE-onnxruntime.txt",
        "package": "onnxruntime",
        "member": "LICENSE",
        "sha256": None,
    },
    {
        "file": "DirectML.dll",
        "package": "directml",
        "member": "bin/x64-win/DirectML.dll",
        "sha256": "9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1",
        "size": 18527776,
    },
    {
        "file": "LICENSE-DirectML.txt",
        "package": "directml",
        "member": "LICENSE.txt",
        "sha256": None,
    },
]

HF_BLAZEPOSE = (
    "https://huggingface.co/unity/inference-engine-blaze-pose/resolve/"
    "d96e13766db93eaaac6ad398d7b01f1c94c7ff6e/models/"
)
MEDIAPIPE_ASSETS = "https://storage.googleapis.com/mediapipe-assets/"

LANDMARK_OUTPUTS = [
    {"name": "Identity", "shape": [1, 195],
     "meaning": "39 landmarks x [x, y, z, visibility_logit, presence_logit] in 256-crop pixels"},
    {"name": "Identity_1", "shape": [1, 1], "meaning": "pose presence score (sigmoid applied)"},
    {"name": "Identity_2", "shape": [1, 256, 256, 1], "meaning": "segmentation logits over the crop"},
    {"name": "Identity_3", "shape": [1, 64, 64, 39], "meaning": "landmark heatmap logits (HWC)"},
    {"name": "Identity_4", "shape": [1, 117], "meaning": "39 world landmarks x [x, y, z] metres, hip-centred"},
]
DETECTOR_OUTPUTS = [
    {"name": "Identity", "shape": [1, 2254, 12],
     "meaning": "per-anchor [x_center, y_center, w, h, 4 x (kp_x, kp_y)] in 224-pixel units relative to the anchor"},
    {"name": "Identity_1", "shape": [1, 2254, 1], "meaning": "class score logits (clip to [-100, 100], sigmoid)"},
]

# Models installed into <dest>/models. "url" is None when the file is not
# hosted anywhere (our own conversion); it can then come from --local-models
# or --detector-url, otherwise the "fallback" entry is installed under the
# same file name.
MODELS = [
    {
        "role": "pose_detection",
        "file": "pose_detection.onnx",
        "sha256": "74bb9484595bda1bba645ffc620152ede708e653c77d8a2f536b0690045a2c22",
        "url": None,
        "source": (
            MEDIAPIPE_ASSETS + "pose_detection.tflite (sha256 "
            "9ba9dd3d42efaaba86b4ff0122b06f29c4122e756b329d89dca1e297fd8f866c) "
            "densified and converted with tf2onnx 1.17 --opset 13"
        ),
        "license": "Apache-2.0",
        "input": {"name": "input_1", "shape": [1, 224, 224, 3], "layout": "NHWC RGB [-1,1]"},
        "outputs": DETECTOR_OUTPUTS,
        "fallback": {
            "sha256": "72081da8481170bc6d8fafa716455ee210b61a8cefed84c67fcbbf889a4c38cf",
            "url": HF_BLAZEPOSE + "pose_detection.onnx",
            "source": "MediaPipe v0.8.4 (2021) pose detector, Unity inference-engine-blaze-pose conversion",
        },
    },
    {
        "role": "pose_detection_fallback",
        "file": "pose_detection_unity2021.onnx",
        "sha256": "72081da8481170bc6d8fafa716455ee210b61a8cefed84c67fcbbf889a4c38cf",
        "url": HF_BLAZEPOSE + "pose_detection.onnx",
        "source": "MediaPipe v0.8.4 (2021) pose detector, Unity inference-engine-blaze-pose conversion",
        "license": "Apache-2.0",
        "input": {"name": "input_1", "shape": [1, 224, 224, 3], "layout": "NHWC RGB [-1,1]"},
        "outputs": DETECTOR_OUTPUTS,
    },
    {
        "role": "pose_landmark_lite",
        "file": "pose_landmark_lite.onnx",
        "sha256": "c9661bfb07f5baf0638cd67263d16b71d6512ae94908a069be0ade7f529b7248",
        "url": HF_BLAZEPOSE + "pose_landmarks_detector_lite.onnx",
        "source": MEDIAPIPE_ASSETS + "pose_landmark_lite.tflite converted with tf2onnx",
        "license": "Apache-2.0",
        "input": {"name": "input_1", "shape": [1, 256, 256, 3], "layout": "NHWC RGB [0,1]"},
        "outputs": LANDMARK_OUTPUTS,
    },
    {
        "role": "pose_landmark_full",
        "file": "pose_landmark_full.onnx",
        "sha256": "ae17ee8f076a5bbc28f65b939f46139c10f10c51ec4392a011e56d06d3f76c5d",
        "url": HF_BLAZEPOSE + "pose_landmarks_detector_full.onnx",
        "source": MEDIAPIPE_ASSETS + "pose_landmark_full.tflite converted with tf2onnx",
        "license": "Apache-2.0",
        "input": {"name": "input_1", "shape": [1, 256, 256, 3], "layout": "NHWC RGB [0,1]"},
        "outputs": LANDMARK_OUTPUTS,
    },
    {
        "role": "pose_landmark_heavy",
        "file": "pose_landmark_heavy.onnx",
        "sha256": "717c32b11d95e214612e2ff0edf825c38a8dd24c9b7c99f34f52f89431c68453",
        "url": HF_BLAZEPOSE + "pose_landmarks_detector_heavy.onnx",
        "source": MEDIAPIPE_ASSETS + "pose_landmark_heavy.tflite converted with tf2onnx",
        "license": "Apache-2.0",
        "input": {"name": "input_1", "shape": [1, 256, 256, 3], "layout": "NHWC RGB [0,1]"},
        "outputs": LANDMARK_OUTPUTS,
    },
]

USER_AGENT = "xenia-nui-setup/1.0 (+https://github.com/xenia-canary/xenia-canary)"


class SetupError(Exception):
    pass


def log(message):
    print(message, flush=True)


_SHA256_CACHE = {}


def sha256_of(path):
    """SHA-256 of a file, memoized on (path, size, mtime) for this run so the
    model folders are not re-hashed for every candidate of every model."""
    path = os.path.abspath(path)
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    cached = _SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    _SHA256_CACHE[key] = result
    return result


def file_ok(path, sha256):
    """True when |path| exists and (if a hash is pinned) matches it."""
    if not os.path.isfile(path):
        return False
    if sha256 is None:
        return os.path.getsize(path) > 0
    return sha256_of(path) == sha256


def download(url, dest_path, expected_sha256=None, expected_size=None, attempts=3):
    """Streams |url| into |dest_path| (via a temp file), verifying the hash."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    last_error = None
    for attempt in range(1, attempts + 1):
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(dest_path) + ".", suffix=".part",
            dir=os.path.dirname(os.path.abspath(dest_path)))
        os.close(tmp_fd)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            digest = hashlib.sha256()
            received = 0
            started = time.time()
            last_report = started
            with urllib.request.urlopen(request, timeout=60) as response, open(tmp_path, "wb") as out:
                total = response.headers.get("Content-Length")
                total = int(total) if total and total.isdigit() else expected_size
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    now = time.time()
                    if now - last_report >= 2.0:
                        last_report = now
                        if total:
                            log("    %5.1f%%  %6.1f MB / %.1f MB" % (
                                100.0 * received / total, received / 1e6, total / 1e6))
                        else:
                            log("    %6.1f MB" % (received / 1e6))
            elapsed = time.time() - started
            log("    %.1f MB in %.1f s" % (received / 1e6, elapsed))
            actual = digest.hexdigest()
            if expected_sha256 and actual != expected_sha256:
                raise SetupError("hash mismatch for %s\n      expected %s\n      got      %s" % (
                    url, expected_sha256, actual))
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.replace(tmp_path, dest_path)
            return actual
        except (urllib.error.URLError, OSError, SetupError) as e:
            last_error = e
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if isinstance(e, SetupError):
                break
            log("    attempt %d failed: %s" % (attempt, e))
            time.sleep(2.0 * attempt)
    raise SetupError("download failed: %s (%s)" % (url, last_error))


def ensure_package(name, cache_dir, skip_download, force):
    """Returns the path of the verified .nupkg for PACKAGES[name]."""
    pkg = PACKAGES[name]
    filename = "%s.%s.nupkg" % (pkg["id"], pkg["version"])
    path = os.path.join(cache_dir, filename)
    if not force and file_ok(path, pkg["sha256"]):
        log("  package %s: cached" % filename)
        return path
    if os.path.isfile(path):
        log("  package %s: cached copy has a wrong hash, re-downloading" % filename)
    if skip_download:
        raise SetupError("package %s is not cached and --skip-download is set" % filename)
    url = NUGET_FLATCONTAINER.format(id=pkg["id"], ver=pkg["version"])
    log("  downloading %s (%.0f MB)" % (url, pkg["size"] / 1e6))
    download(url, path, pkg["sha256"], pkg["size"])
    return path


def extract_member(package_path, member, dest_path, sha256):
    with zipfile.ZipFile(package_path) as zf:
        try:
            info = zf.getinfo(member)
        except KeyError:
            raise SetupError("%s has no member %s" % (package_path, member))
        tmp_path = dest_path + ".part"
        with zf.open(info) as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    if sha256 is not None:
        actual = sha256_of(tmp_path)
        if actual != sha256:
            os.remove(tmp_path)
            raise SetupError("hash mismatch for %s extracted from %s\n      expected %s\n      got      %s" % (
                member, os.path.basename(package_path), sha256, actual))
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.replace(tmp_path, dest_path)


def install_runtime(runtime_dir, cache_dir, skip_download, force):
    """Installs RUNTIME_FILES; returns the list of problems (empty = ok)."""
    problems = []
    os.makedirs(runtime_dir, exist_ok=True)
    packages = {}
    for entry in RUNTIME_FILES:
        dest = os.path.join(runtime_dir, entry["file"])
        if not force and file_ok(dest, entry["sha256"]):
            log("  %-24s ok (%d bytes)" % (entry["file"], os.path.getsize(dest)))
            continue
        if os.path.isfile(dest) and not force:
            log("  %-24s present but hash mismatch, replacing" % entry["file"])
        try:
            name = entry["package"]
            if name not in packages:
                packages[name] = ensure_package(name, cache_dir, skip_download, force)
            extract_member(packages[name], entry["member"], dest, entry["sha256"])
            log("  %-24s installed (%d bytes)" % (entry["file"], os.path.getsize(dest)))
        except SetupError as e:
            problems.append("%s: %s" % (entry["file"], e))
            log("  %-24s FAILED: %s" % (entry["file"], e))
    return problems


def find_local_copy(local_dirs, sha256):
    """Looks for any file with the given hash in |local_dirs| (non-recursive)."""
    for directory in local_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and name.lower().endswith(".onnx") and sha256_of(path) == sha256:
                return path
    return None


def install_models(models_dir, local_dirs, detector_url, allow_fallback, skip_download, force):
    """Installs MODELS; returns (installed entries for models.json, problems,
    notes). Notes are non-fatal but worth repeating in the final summary (a
    fallback model was installed)."""
    notes = []
    problems = []
    installed = []
    os.makedirs(models_dir, exist_ok=True)
    for model in MODELS:
        dest = os.path.join(models_dir, model["file"])
        candidates = []
        url = detector_url if (model["url"] is None and detector_url) else model["url"]
        candidates.append({"sha256": model["sha256"], "url": url, "source": model["source"]})
        if model.get("fallback"):
            fb = model["fallback"]
            candidates.append({"sha256": fb["sha256"], "url": fb["url"], "source": fb["source"],
                               "is_fallback": True})

        # Already installed with one of the accepted hashes?
        chosen = None
        if os.path.isfile(dest) and not force:
            actual = sha256_of(dest)
            for candidate in candidates:
                if candidate["sha256"] == actual:
                    chosen = candidate
                    break
            if chosen is None:
                log("  %-32s present but hash mismatch, replacing" % model["file"])
        if chosen is not None:
            log("  %-32s ok (%d bytes)%s" % (model["file"], os.path.getsize(dest),
                                             " [fallback build]" if chosen.get("is_fallback") else ""))
        else:
            error = None
            for candidate in candidates:
                if candidate.get("is_fallback") and not allow_fallback:
                    continue
                try:
                    local = find_local_copy(local_dirs, candidate["sha256"])
                    if local:
                        shutil.copyfile(local, dest + ".part")
                        os.replace(dest + ".part", dest)
                        log("  %-32s copied from %s" % (model["file"], local))
                        chosen = candidate
                        break
                    if candidate["url"] is None:
                        error = "no download URL (a local conversion; pass --local-models or --detector-url)"
                        continue
                    if skip_download:
                        error = "missing and --skip-download is set"
                        continue
                    log("  %-32s downloading %s" % (model["file"], candidate["url"]))
                    download(candidate["url"], dest, candidate["sha256"])
                    log("  %-32s installed (%d bytes)%s" % (
                        model["file"], os.path.getsize(dest),
                        " [fallback build]" if candidate.get("is_fallback") else ""))
                    chosen = candidate
                    break
                except SetupError as e:
                    error = str(e)
                    log("  %-32s FAILED: %s" % (model["file"], e))
            if chosen is None:
                problems.append("%s: %s" % (model["file"], error))
                continue
            if chosen.get("is_fallback"):
                note = ("%s: installed the older 2021 detector as a fallback; the current "
                        "detector is not hosted anywhere yet (pass --detector-url or "
                        "--local-models)" % model["file"])
                log("  NOTE: " + note)
                notes.append(note)

        installed.append({
            "role": model["role"],
            "file": model["file"],
            "sha256": chosen["sha256"],
            "url": chosen["url"],
            "source": chosen["source"],
            "license": model["license"],
            "input": model["input"],
            "outputs": model["outputs"],
        })
    return installed, problems, notes


def write_manifest(models_dir, installed):
    manifest = {
        "runtime_version": ORT_VERSION,
        "ort_api_version": ORT_API_VERSION,
        "directml_version": DIRECTML_VERSION,
        "generated_by": "tools/nui/setup_nui.py",
        "models": installed,
    }
    path = os.path.join(models_dir, "models.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return path


def resolve_dest(args):
    if args.dest:
        return os.path.abspath(args.dest)
    if args.exe:
        return os.path.join(os.path.dirname(os.path.abspath(args.exe)), "nui")
    existing = [d for d in DEFAULT_EXE_DIR_CANDIDATES if os.path.isdir(d)]
    # An install that is already there wins (idempotent re-runs update it),
    # then the first build folder that holds an emulator executable.
    for candidate in existing:
        if os.path.isdir(os.path.join(candidate, "nui", "runtime")) or \
                os.path.isdir(os.path.join(candidate, "nui", "models")):
            return os.path.join(candidate, "nui")
    for candidate in existing:
        if any(name.lower().startswith("xenia") and name.lower().endswith(".exe")
               for name in os.listdir(candidate)):
            return os.path.join(candidate, "nui")
    raise SetupError(
        "no build folder found (looked for %s); pass --exe <xenia executable> or --dest <folder>"
        % ", ".join(DEFAULT_EXE_DIR_CANDIDATES))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", help="the nui folder to install into (contains runtime/ and models/)")
    parser.add_argument("--exe", help="emulator executable; <dest> becomes <its folder>/nui")
    parser.add_argument("--skip-download", action="store_true",
                        help="never touch the network; only verify (and extract from cached packages)")
    parser.add_argument("--force", action="store_true", help="re-install even when hashes already match")
    parser.add_argument("--cache-dir", help="where downloaded NuGet packages are kept (default <dest>/cache)")
    parser.add_argument("--local-models", action="append", default=[], metavar="DIR",
                        help="folder with pre-converted .onnx files to copy from (matched by hash); repeatable")
    parser.add_argument("--detector-url", help="download URL for the current pose_detection.onnx conversion")
    parser.add_argument("--no-detector-fallback", action="store_true",
                        help="fail instead of installing the 2021 detector as pose_detection.onnx")
    parser.add_argument("--models-only", action="store_true", help="skip the runtime DLLs")
    parser.add_argument("--runtime-only", action="store_true", help="skip the models")
    args = parser.parse_args(argv)

    try:
        dest = resolve_dest(args)
    except SetupError as e:
        log("error: %s" % e)
        return 1
    runtime_dir = os.path.join(dest, "runtime")
    models_dir = os.path.join(dest, "models")
    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else os.path.join(dest, "cache")
    log("xenia NUI setup: ONNX Runtime %s (DirectML %s), BlazePose models" % (ORT_VERSION, DIRECTML_VERSION))
    log("  dest: %s" % dest)

    problems = []
    notes = []
    if not args.models_only:
        log("runtime -> %s" % runtime_dir)
        problems += install_runtime(runtime_dir, cache_dir, args.skip_download, args.force)
    if not args.runtime_only:
        log("models -> %s" % models_dir)
        # The models folder itself is searched too, so a file already
        # downloaded under another name (the fallback detector) is reused.
        installed, model_problems, model_notes = install_models(
            models_dir, list(args.local_models) + [models_dir], args.detector_url,
            not args.no_detector_fallback, args.skip_download, args.force)
        problems += model_problems
        notes += model_notes
        manifest = write_manifest(models_dir, installed)
        log("  wrote %s (%d models)" % (manifest, len(installed)))

    if problems:
        log("FAILED:")
        for problem in problems:
            log("  " + problem)
        return 1
    if notes:
        # Exit code 2: everything installed, but with a substitute model.
        log("done, with notes:")
        for note in notes:
            log("  " + note)
        return 2
    log("done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
