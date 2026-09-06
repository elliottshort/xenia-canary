# NUI (Kinect) reverse-engineering helpers

Scripts used to build and maintain the NUI runtime signature database
(`src/xenia/kernel/nui/nui_signatures_builtin.cc`). Requirements: Python 3,
`pefile`, `capstone`.

The `milo_*.py` scripts default to the Project Milo dev build in
`xbe/Project Milo (May 17 2010)/` (set `XENIA_MILO_DIR` or pass `--map`/`--exe`
to use another symbolized build):

- `milo_pdb.py [PDB] [EXE]`: extracts function symbols (including module-local
  ones the map does not list) from a PDB into `milo_symbols.tsv` next to the
  script.
- `milo_sym.py ADDR...` / `--find NAME`: symbolizes addresses using the linker
  map plus `milo_symbols.tsv`.
- `milo_dis.py ADDR [COUNT]` / `--func NAME [COUNT]` / `--bytes ADDR [LEN]`:
  PowerPC disassembly (capstone) from the raw PE image with symbolized branch
  targets.
- `milo_xref.py ADDR` / `--calls ADDR` / `--str SUBSTR`: data references
  (lis/addi pairs), call sites, and string search.
- `milo_imports.py [lib]`: XEX import thunks with ordinals and names.
- `gen_signatures.py NAME...`: emits masked instruction signatures for the
  signature database, optionally verifying uniqueness against a second build
  (`--verify other.exe --map2 other.map`). See `docs/nui/adding_signatures.md`.

For retail titles without symbols, dump the module image by running the title
with `--nui=true --nui_dump_module_image=<dir>` and analyse the `.bin` in
Ghidra at the base address given in the `.json` sidecar. `gen_signatures.py`
currently reads PE images (`--exe`); for raw dumps use `--addr`/`--name` after
adding a raw-image reader (the sidecar gives the code ranges).

## Webcam source runtime files (`setup_nui.py`)

The webcam source (`--nui_source=webcam`) needs ONNX Runtime with DirectML and
the BlazePose models at run time; nothing is linked at build time.
`setup_nui.py` (standard library only) installs them into the emulator's
storage root:

```
python tools/nui/setup_nui.py --exe path/to/xenia_canary_netplay.exe
python tools/nui/setup_nui.py --dest <storage>/nui --skip-download   # verify
```

It downloads `Microsoft.ML.OnnxRuntime.DirectML` 1.24.4 and
`Microsoft.AI.DirectML` 1.15.4 from NuGet (kept in `<dest>/cache`), extracts
`onnxruntime.dll`, `DirectML.dll` and their licenses into `<dest>/runtime`,
downloads `pose_landmark_{lite,full,heavy}.onnx` and the 2021
`pose_detection_unity2021.onnx` from Hugging Face (Apache-2.0) into
`<dest>/models`, verifies SHA-256 hashes and writes `models.json`. The current
MediaPipe pose detector (`pose_detection.onnx`) is a local tf2onnx conversion
that is not hosted anywhere yet: pass `--local-models DIR` (copied when the
hash matches) or `--detector-url URL`; otherwise the 2021 detector is
installed under that name and reported as a fallback.
