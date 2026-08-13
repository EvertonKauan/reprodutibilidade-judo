# detector_quedas — standalone executable

Packaging of the throw-detection module (same logic as `run_fall_batch_headless.py`,
detection unchanged) as an executable, to support reproducibility of the associated
thesis/paper.

> **Naming note:** the executable and its companion script (`detector_quedas`,
> `worker_deteccao.py`) keep their original names. They are a compiled artifact and a
> script it invokes internally by that exact name; renaming either without rebuilding
> and retesting the executable risks breaking it, so they were intentionally left
> untouched while the rest of this repository was translated to English.

## How it works

The executable (`detector_quedas`) does **not** bundle `torch`/`ultralytics` — these two
dependencies are heavy (hundreds of MB to a few GB with CUDA), and bundling them would
make the executable huge and fragile across different machines/OSes. Instead:

1. You install `ultralytics` (which brings `torch` as a dependency) in your own Python:
   ```
   pip install ultralytics
   ```
2. The executable does a quick check at startup. If the dependency is not installed, it
   shows a clear warning with the exact command to run, and exits cleanly (no crash, no
   confusing error).
3. If everything is fine, the executable delegates the heavy processing (YOLO pose + the
   detection logic) to the **system Python** via a subprocess — not to a Python
   interpreter bundled inside the executable. This avoids the incomplete-standard-library
   issues that happen when trying to import heavy libraries inside a "frozen" interpreter
   (PyInstaller).

## Requirements

- `pip install ultralytics` (brings `torch` along) in the Python that is on `PATH` as
  `python3` (or point to another one with `--python /path/to/python3`).
- `opencv-python` and `numpy` (also installed automatically as a dependency of the rest
  of the pipeline).
- **Internet connection on first run**: the `yolo11n-pose.pt` weights (Ultralytics'
  official pose model, not trained by this project) do not ship in this folder —
  `ultralytics` itself downloads and caches it automatically the first time it runs
  (official file, straight from the `github.com/ultralytics/assets` repository). This
  avoids redistributing a third-party artifact and keeps the license of what we ship
  simpler: only the project's own code and model (`tatame_guard.pt`).

## Contents of this folder (keep everything together)

```
detector_quedas            <- the executable itself
worker_deteccao.py        <- runs on the system Python (do not modify)
modulos/                  <- detection logic (fall-detector)
```

`yolo11n-pose.pt` (Ultralytics' official model, AGPL-3.0 license) is **not** in this
folder — it is downloaded automatically by `ultralytics` on first run and cached in the
directory where the command is executed.

## Note on `tatame_guard`

This version does **not** include `tatame_guard` (validation that the fall occurred
inside the tatami area). This simplifies the distributed package, but changes the
behavior: every fall detected by pose is reported, even if it occurred outside the
tatami (crowd, referee, etc.) — without that extra filtering, the false-positive rate is
higher. This is also documented in the paper: the throw-detection module "tends to
overdetect events, occasionally identifying defensive postures or other non-throw
movements as throws" — manual review of the automatically extracted clips is required
before use.

## Usage

```bash
./detector_quedas \
  --list candidates.json \
  --videos-base /path/to/videos \
  --output throws_report.json
```

`candidates.json`: a list in the format `[{"id": "...", "arquivo": "video1.mp4"}, ...]`,
where `arquivo` is relative to `--videos-base`.

Output (`throws_report.json`): a list with the timestamps (mm:ss) of every detected
throw per video — same format as `run_fall_batch_headless.py`.

## Tested on

- Linux (Ubuntu/Pop!_OS), Python 3.10, ultralytics installed via pip.
- Verified: (1) real detection produces results identical to the original script (same
  video, same 8 detected throws); (2) the missing-dependency warning appears correctly
  when tested against a venv without ultralytics/torch installed.
- **Not tested** on Windows/macOS — the mechanism for finding the "system Python" via
  `python3`/`python` on PATH should work, but has not been verified on those platforms.
