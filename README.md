# Legged Gym on Apple Silicon (Genesis + MPS)

This repository provides a **Legged Gym** environment that runs on  
**Apple Silicon (M1 / M2 / M3)** using the **Genesis physics engine** and  
**PyTorch MPS (Metal Performance Shaders)** backend.

The primary goal of this project is to enable **legged robot reinforcement learning on macOS**
without relying on CUDA, by replacing Isaac Gym with Genesis while preserving
the original **legged_gym** training pipeline.

---

## Overview

- Apple Silicon–compatible Legged Gym
- Genesis physics engine as the simulator backend
- PPO training using `rsl_rl`
- GPU acceleration via PyTorch MPS
- Minimal changes to the original legged_gym code structure

This project is intended for research and experimentation on macOS systems
where CUDA-based simulators are unavailable.

---

## Based On

This repository is built by combining and adapting the following open-source projects:

- **Genesis (Apple Silicon installation and runtime support)**  
  https://github.com/adamcroft330/genesis

- **Genesis-compatible Legged Gym**  
  https://github.com/aCodeDog/genesis_legged_gym

- **Original Legged Gym**  
  https://github.com/leggedrobotics/legged_gym

Specifically:
- The Genesis installation and Apple Silicon compatibility are derived from (1)
- The Genesis–Legged Gym interface and environment structure are adapted from (2)

---

## System Requirements

- macOS running on Apple Silicon (M1 / M2 / M3)
- Python >= 3.9 (Python 3.11 recommended)
- Miniconda or Conda
- Approximately 2GB of free disk space
- Internet connection for dependency installation

---