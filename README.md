# UR Hand Teleoperator
![License](https://img.shields.io/badge/License-MIT-green?logo=github)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-blue?logo=google)
![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics%20Simulation-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-blue?logo=opencv)
![ROS](https://img.shields.io/badge/Robot-UR5e-lightgrey?logo=ros)
![Focus](https://img.shields.io/badge/Focus-Human%20Robot%20Mirroring-red)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

> Control a simulated UR5e robotic arm using your bare hand.

## Demo
Point your webcam at yourself, give a thumbs up to start tracking, move your hand, and watch the UR5e follow in the simulator window alongside your camera feed.

## Requirements
- Linux
- Python 3.10+
- Webcam or camera
- An AprilTag (tag36h11 family) for depth estimation

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

Two windows open: your webcam feed with hand keypoints overlaid, and the MuJoCo viewer with the UR5e. Use gestures to control the arm. Press `Q` to quit.

| Gesture | Action |
|---|---|
| 👍 Thumbs up | Arm tracking — start following your hand |
| ✊ Fist | Close gripper |
| 🖐 Open hand | Open gripper |

## How It Works

The system runs two threads in parallel: a camera thread at 30 Hz and a control thread at 50 Hz.

**Camera thread** captures frames from the webcam and runs two things concurrently: MediaPipe Hand Landmarker detects 21 hand keypoints and computes the palm centroid position, and an AprilTag detector estimates the hand's depth (Z distance) via PnP pose solving. Both outputs are smoothed with an exponential moving average (EMA) before being passed to the control thread, which reduces the effect of frame-to-frame noise on the arm.

**Control thread** reads the latest hand state and runs a gesture state machine. When the user arms with a thumbs-up, the system captures a reference hand position and end-effector pose. From that point on, hand movement is tracked as a delta from the reference — how far the hand has moved from where it was when arming, not where it is in absolute terms. That delta is scaled and added to the reference end-effector position to produce a desired pose.

A PD controller smoothly drives the arm toward the desired pose each tick, outputting a small incremental Cartesian step rather than a full position command. That step is fed into a damped least-squares Jacobian IK solver, which converts it to joint angles. Those angles are written to the MuJoCo actuators and the simulation is stepped forward.


## Project Structure

```
UR-Hand-Teleop/
├── src/
│   ├── main.py                
│   └── modules/
│       ├── hand_tracker.py     
│       ├── depth_estimator.py 
│       ├── pd_controller.py    
│       ├── ik_solver.py        
│       └── mujoco_wrapper.py    
├── tests/
│   ├── test_ik.py    
│   └── test_keyboard.py  
├── arm_models/
│   └── ur5e_model/
├── mp_models/
│   └── hand_landmarker.task
├── requirements.txt
└── README.md
```

## Dependencies

- [MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection
- [MuJoCo](https://mujoco.org) — physics simulation
- [OpenCV](https://opencv.org) — webcam capture and display
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — UR5e robot model
- [dt-apriltags](https://github.com/duckietown/dt-apriltags) — AprilTag detection for depth estimation

## Roadmap

- Better depth estimation — the current AprilTag approach requires a physical marker in frame and is sensitive to calibration; proper monocular depth or an RGB-D camera would be significantly more robust
- More accurate open/close detection — finger state detection based on landmark Y positions breaks under hand rotation; a learned classifier or wrist-relative geometry would handle more poses
- Better input smoothing — reduce arm jitter under fast or noisy hand movement without adding lag
- Add a scene and objects for testing — tables, objects to grasp, and pick-and-place tasks to benchmark tracking quality
- Side-by-side display combining the webcam feed and simulation render into a single window
- Support for additional robot models

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.