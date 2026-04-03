<div align="center">
  <h1 align="center"> OpenTrack </h1>
  <h3 align="center"> GALBOT · Tsinghua </h3>
<!--   <p align="center">
    <a href="README.md"> English </a> | <a href="README_zh.md">中文</a>
  </p>     -->

:page_with_curl:[Paper](https://arxiv.org/abs/2509.13833) | :house:[Website](https://zzk273.github.io/Any2Track/)


This repository is the official implementation of OpenTrack, an open-source humanoid motion tracking codebase that uses MuJoCo JAX for simulation and supports multi-GPU parallel training.
</div>

# News 🚩

[November 30, 2025] LAFAN1 generalist v1 released. **Now you can track cartwheel, kungfu, fall and getup, and many other motions within a single policy.**

[September 19, 2025] Simple Domain Randomization released.

[September 19, 2025] Tracking codebase released.

# TODOs

- [x] Release motion tracking codebase
- [x] Release simple domain randomization
- [x] Release pretrained LAFAN1 generalist v1 checkpoints
- [ ] Release DAgger code
- [ ] Release AnyAdapter
- [ ] Release more pretrained checkpoints
- [ ] Release real-world deployment code

# Prepare

1. Clone the repository:
  ```shell
   git clone git@github.com:GalaxyGeneralRobotics/OpenTrack.git
  ```
2. Create a virtual environment and install dependencies:
  ```shell
   uv sync -i https://pypi.org/simple 
  ```
3. Create a `.env` file in the project directory with the following content:
  ```dotenv
   export GLI_PATH=<your_project_path>
   export WANDB_PROJECT=<your_project_name>
   export WANDB_ENTITY=<your_wandb_entity>
   export WANDB_API_KEY=<your_wandb_api_key>
  ```
4. Download the [mocap data](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main/Lafan1/mocap/UnitreeG1) and put them under `storage/data/mocap/`. Thanks for the retargeting motions of LAFAN1 dataset from [LocoMuJoCo](https://github.com/robfiras/loco-mujoco/)! The file structure should be like:
  ```
   storage                   
   ├── assets
   │   ├── mujoco_menagerie
   │   └── unitree_g1
   ├── data
   │   ├── hfield            
   │   │   └── terrain.npz
   │   └── mocap
   │       └── lafan1   
   │           └── UnitreeG1
   ├── logs
   │   ├── dagger      # dagger student checkpoints
   │   └── track       # tracking checkpoints  
   └── ...
  ```
5. Initialize assets
  ```shell
   # First, initialize the environment
   source .venv/bin/activate; source .env;
   # Then, initialize the assets
   python track_mj/app/mj_playground_init.py
  ```

# Usage

## Initialize environment

1. Initialize the MuJoCo environment:
  ```shell
   source .venv/bin/activate; source .env;
  ```

## Play pretrained checkpoints

1. Download pretrained checkpoints and configs from [checkpoints and configs](https://drive.google.com/drive/folders/1wDL4Chr6sGQiCx1tbvhf9DowN73cP_PF?usp=drive_link), and put them under `storage/logs/dagger`. Visualization results: [videos](https://drive.google.com/drive/folders/1yFAG2UIZq5-504MkKTwevquwRO1OsGOL?usp=sharing).
2. Run the evaluation script:
  ```shell
   python -m track_mj.eval.dagger.mj_onnx_video --task G1TrackingGeneral --exp_name general_tracker_lafan_v1 [--use_viewer] [--use_renderer] [--play_ref_motion]
  ```

As of **November 30, 2025**, we have open-sourced **a generalist model on LAFAN1**, daggered from four teachers. This checkpoint was trained with simple domain randomization (DR). You may try deploying it on a Unitree G1 robot using your own deployment code, since we have not yet open-sourced our real-robot deployment pipeline.

## Train from scratch

1. Train the specialist teachers:
  ```shell
   python -m track_mj.learning.train.train_ppo_track --task G1TrackingGeneralDR --exp_name <your_exp_name>
  ```

## Evaluate the model

1. Evaluate the specialist teachers:
  ```shell
   # First, convert the Brax model checkpoint to ONNX
   python -m track_mj.app.brax2onnx_tracking --task G1TrackingGeneral --exp_name <your_exp_name>
   # Next, run the evaluation script
   python -m track_mj.eval.tracking.mj_onnx_video --task G1TrackingGeneral --exp_name <your_exp_name> [--use_viewer] [--use_renderer] [--play_ref_motion]
  ```

# Acknowledgement

This repository is build upon `jax`, `brax`, `loco-mujoco`, and `mujoco_playground`.

If you find this repository helpful, please cite our work:

```bibtex
@misc{zhang2025trackmotionsdisturbances,
      title={Track Any Motions under Any Disturbances}, 
      author={Zhikai Zhang and Jun Guo and Chao Chen and Jilong Wang and Chenghuai Lin and Yunrui Lian and Han Xue and Zhenrong Wang and Maoqi Liu and Jiangran Lyu and Huaping Liu and He Wang and Li Yi},
      year={2025},
      eprint={2509.13833},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.13833}, 
}
```

