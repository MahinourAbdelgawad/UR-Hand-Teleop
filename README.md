# ArmBot
![License](https://img.shields.io/badge/License-MIT-green?logo=github)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Landmarker-blue?logo=google)
![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics%20Simulation-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-blue?logo=opencv)
![ROS](https://img.shields.io/badge/Robot-UR5e-lightgrey?logo=ros)
![Focus](https://img.shields.io/badge/Focus-Human%20Robot%20Mirroring-red)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

> Control a simulated robotic arm using your own arm.
> Runs on MuJoCo using a UR5e robot arm

## Demo
Point your webcam at yourself, raise or bend your arm, and watch the UR5e follow in the simulator window alongside your camera feed.

## Requirements
- Linux
- Python 3.10+
- Webcam


## Installation

Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/MahinourAbdelgawad/ArmBot
cd armbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the MediaPipe pose model:

```bash
mkdir -p mp_models

wget -O mp_models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```
> The full model provides more stable and accurate tracking than the lite version, at the cost of slightly higher CPU usage.

Download the UR5e MuJoCo model:
```bash
mkdir -p arm_models

git clone --depth=1 https://github.com/google-deepmind/mujoco_menagerie.git

cp -r mujoco_menagerie/universal_robots_ur5e/ur5e arm_models/ur5e_model

rm -rf mujoco_menagerie
```

## Usage

```bash
python -m src.main
```

Two windows will open: your webcam feed with skeleton overlay, and the MuJoCo viewer with the UR5e. Move your right arm to control the robot. Press `Q` to quit.

## Project Structure

```
ArmBot/
├── src/
│   ├── main.py
│   └── modules/
│       ├── arm_tracker.py       
│       ├── angle_mapper.py     
│       └── mujoco_wrapper.py  
├── arm_models/
│   └── ur5e_model/    
├── mp_models/
│   └── pose_landmarker_full.task
├── requirements.txt
└── README.md
```

## How It Works
The shoulder, elbow, and wrist positions are extracted using MediaPipe Pose and used to compute joint angles via vector dot products. Those angles are normalized against the human range of motion, mapped to the UR5e joint limits, smoothed with an exponential moving average filter, and written to the MuJoCo actuators every frame.

The simulation runs a standard MuJoCo PD controller on the UR5e model, so the arm moves physically plausibly rather than snapping directly to target angles.

## Configuration

All tunable parameters live at the top of each module.

**Smoothing** — controls how much the EMA filter dampens jitter. Lower is smoother but laggier, higher is more responsive but may jitter.

```python
mapper = AngleMapper(smoothing_factor=0.8)
```

**Human angle ranges** — calibrate these to your own range of motion for best results.

```python
self.HUMAN_ELBOW_MIN  = 30.0   # degrees, fully curled
self.HUMAN_ELBOW_MAX  = 175.0  # degrees, fully extended
```

**MediaPipe model** — swap between lite and full depending on your hardware.

```python
tracker = ArmTracker(model_path="mp_models/pose_landmarker_full.task")
```

## Dependencies

- [MediaPipe](https://developers.google.com/mediapipe) — pose landmark detection
- [MuJoCo](https://mujoco.org) — physics simulation
- [OpenCV](https://opencv.org) — webcam capture and display
- [NumPy](https://numpy.org) — angle computation
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — UR5e robot model


## Roadmap
- Wrist joint control
- Hand to Gripper control
- Side-by-side display combining the webcam feed and simulation render into a single window
- Support for additional robot models

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
