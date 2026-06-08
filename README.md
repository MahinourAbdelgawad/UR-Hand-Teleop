# UR Hand Teleoperator
![License](https://img.shields.io/badge/License-MIT-green?logo=github)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-blue?logo=google)
![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics%20Simulation-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-blue?logo=opencv)
![ROS](https://img.shields.io/badge/Robot-UR5e-lightgrey?logo=ros)
![Focus](https://img.shields.io/badge/Focus-Human%20Robot%20Mirroring-red)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

> Control a simulated robotic arm using your own hand.
> Runs on MuJoCo using a UR5e robot arm

## Demo
Point your webcam at yourself, move your hand, and watch the UR5e follow in the simulator window alongside your camera feed.

## Requirements
- Linux
- Python 3.10+
- Webcam or Camera


## Installation

Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/MahinourAbdelgawad/UR-Hand-Teleop
cd UR-Hand-Teleop
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the MediaPipe hand landmarker model:

```bash
mkdir -p mp_models

wget -O mp_models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

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

Two windows will open: your webcam feed with hand keypoints overlay, and the MuJoCo viewer with the UR5e. Move your hand to control the robot arm. Press `Q` to quit.

## Project Structure

```
UR-Hand-Teleop/
├── src/
│   ├── main.py
│   └── modules/
│       ├── hand_tracker.py       
│       ├── depth_estimator.py     
│       ├── ik_solver.py     
│       └── mujoco_wrapper.py  
├── arm_models/
│   └── ur5e_model/    
├── mp_models/
│   └── hand_landmarker.task
├── requirements.txt
└── README.md
```

## How It Works
Hand keypoints are detected using MediaPipe Hand Landmarker, providing real-time hand position and orientation data. Depth estimation is used to determine the 3D position of the hand, which is then converted to UR5e joint angles using inverse kinematics (IK). Those computed angles are smoothed and applied to the MuJoCo simulation actuators every frame.

The simulation runs a standard MuJoCo PD controller on the UR5e model, so the arm moves physically plausibly while following the hand's commands.

## Dependencies

- [MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection
- [MuJoCo](https://mujoco.org) — physics simulation
- [OpenCV](https://opencv.org) — webcam capture and display
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — UR5e robot model


## Roadmap
- Better depth estimation
- More accurate movement
- Add a scene and objects for testing
- Side-by-side display combining the webcam feed and simulation render into a single window
- Support for additional robot models

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
