# ICIS 2026 Workshop

This folder contains simple Python examples for video tracking.

## Folder Contents

- `Examples/` contains the example Python scripts.
- `Results/` contains example CSV output files.
- `pixi.toml` and `pixi.lock` define the Python environment.

## Install Pixi

Install Pixi from:

https://pixi.sh/latest/

Then install the project environment:

```bash
pixi install
```

## Run Examples

Run a script with Pixi like this:

```bash
pixi run python Examples/HandTracking_example.py
```

```bash
pixi run python Examples/ColorTracking_Example.py
```

The videos are not included in this folder. Use your own video and update the video path in the script if needed.

## Models And Inputs

For hand tracking, download the MediaPipe hand model here:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Save it as:

```text
Models/hand_landmarker.task
```

The color tracking example does not need a model. It uses color ranges inside the script. You can adapt the HSV color ranges in `COLOR_BANDS` for your own video.

## What The Examples Do

- `HandTracking_example.py` tracks hands in a video and saves hand landmark data.
- `ColorTracking_Example.py` tracks colored objects in a video and saves color position data.
