# Incremental Sheet Metal Forming using ABB IRB 1410

This project demonstrates the implementation of **Single Point Incremental Forming (SPIF)** on sheet metal using an **ABB IRB 1410 robotic arm**, integrating CAD design, robot kinematics, force/impedance control, and physics-based simulation.

## 🛠️ Key Features

- 🎯 Trajectory planning for complex geometries using G-code parsing
- 🤖 Real-time force and impedance control using Python-based control loops
- 🏗️ Custom-designed tool holder and forming tool for stable interaction
- 🧪 Simulated environment for validation using **MuJoCo** and **Gazebo**
- 📈 Analysis of deformation, springback, and forming accuracy

## 📁 Project Structure

- `docs/` – Final report and documentation
- `cad_models/` – Design files of tool holder and workpiece
- `simulation/` – MuJoCo and Gazebo-based robotic simulations
- `control/` – Control scripts including force and impedance control
- `ros2_ws/` – ROS 2 package with robot interface and launch files
- `data/` – Raw experimental results and log data
- `analysis/` – Jupyter notebooks for post-processing and analysis

## 🧰 Technologies Used

- ROS 2 Humble
- Python 3.10
- ABB IRB 1410 with IRC5 Controller
- MuJoCo Physics Engine
- Gazebo Simulator
- SolidWorks for CAD
- NumPy, pandas, matplotlib

## 🧪 Future Work

- Closed-loop feedback using real-time vision
- Adaptive path correction based on online force feedback
- Multi-pass forming with springback compensation

## 📄 License

This project is licensed under the MIT License.

## 👥 Contributors

- Yogesh G. R.
- [Other teammates if any]
- IIT Madras Mechanical Engineering UGRP 2024–25
