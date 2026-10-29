# BulletTime: Decoupled Control of Time and Camera Pose for Video Generation

<div align="center">

**CVPR 2026**

[Yiming Wang](https://19reborn.github.io/)<sup>1,2</sup>, [Qihang Zhang](https://zqh0253.github.io/)<sup>3,2\*</sup>, [Shengqu Cai](https://primecai.github.io/)<sup>2\*</sup>, [Tong Wu](https://wutong16.github.io/)<sup>2†</sup>, [Jan Ackermann](https://janackermann.info/)<sup>2†</sup>, [Zhengfei Kuang](https://zhengfeikuang.com/)<sup>2†</sup>, [Yang Zheng](https://y-zheng18.github.io/)<sup>2†</sup>, [Frano Rajič](https://m43.github.io/)<sup>1†</sup>, [Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html)<sup>1</sup>, [Gordon Wetzstein](https://stanford.edu/~gordonwz/)<sup>2</sup>

<sup>1</sup>ETH Zurich &nbsp; <sup>2</sup>Stanford University &nbsp; <sup>3</sup>CUHK  
<!-- \*Equal contribution &nbsp; †Equal contribution -->
\*,† Equal contribution

[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://19reborn.github.io/Bullet4D/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.05076-b31b1b.svg)](https://arxiv.org/abs/2512.05076)
[![YouTube](https://img.shields.io/badge/YouTube-Demo-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=HS5nUlmH4CE)

Given a single input video, **BulletTime** generates new videos with **decoupled control** over world time and camera pose — enabling bullet-time effects such as moving the camera while freezing or slowing scene dynamics.

<br>



https://github.com/user-attachments/assets/c644ea54-a689-4a5f-8f4e-1a72c4d6494e



<br>

</div>

---

## Setup

This project builds on [CogVideoX1.5-5B](https://github.com/zai-org/CogVideo). For environment setup and issues with upstream pretrained weights (e.g., VAE), please refer to the official CogVideoX repository.

We also provide a conda environment tested with **Python 3.10** and **PyTorch 2.5.1**:

```bash
conda env create -f environment.yaml
conda activate bullettime
```

---

## Inference

Given a **single input video**, **BulletTime** synthesizes new videos with user-specified control over **camera pose** and **world time**.

### 1. Prepare checkpoints

Download the pretrained BulletTime checkpoint from Hugging Face:

- **Pretrained checkpoints**: [`19reborn/bulletTime_ckpt`](https://huggingface.co/19reborn/bulletTime_ckpt)


### 2. Prepare input video

We use example videos from [ReCamMaster](https://github.com/KlingAIResearch/ReCamMaster/tree/main/example_test_data/videos) as reference inputs. Organize your data directory as follows:

```
your_data/
├── metadata.csv          # columns: file_name, text
└── videos/
      └── {video_name}.mp4
```

`metadata.csv` should list each video path (relative to the data root) and its text prompt. See the released dataset for a complete example. For videos without captions, you can generate prompts automatically from the input videos using video understanding models (e.g., [MiniCPM-V](https://github.com/openbmb/MiniCPM-V)).

### 3. Prepare camera and time controls

BulletTime requires separate control signals for **camera pose** and **world time**. Example files are provided in [`example_scripts/example_data/`](example_scripts/example_data/); the expected formats are:

| Control | Format | Example |
|---------|--------|---------|
| **Camera trajectory** | JSON with per-frame c2w matrices | [`example_scripts/example_data/camera.json`](example_scripts/example_data/camera.json) |
| **Camera intrinsics** | JSON with focal length and principal point | [`example_scripts/example_data/intrinsic.json`](example_scripts/example_data/intrinsic.json) |
| **World time trajectory** | `.npy` or `.txt`, per-frame world time of shape `[T, 1]` | [`example_scripts/example_data/slow_motion.npy`](example_scripts/example_data/slow_motion.npy) |

You can also design custom trajectories with the interactive tools below.

#### Interactive control tools

We also provide browser-based tools for designing control signals:

- [`example_scripts/camera_control.html`](example_scripts/camera_control.html) — interactive camera trajectory editor (export camera JSON)
- [`example_scripts/time_control.html`](example_scripts/time_control.html) — time-curve tuner (export timing `.npy`)

Open either file locally in a browser to visualize and export trajectories.

### 4. Run inference

Edit the user-editable paths at the top of [`example_scripts/eval.sh`](example_scripts/eval.sh):

- `DATA_ROOT`: directory containing `metadata.csv` and input videos
- `CHECKPOINT`: local path to the downloaded BulletTime checkpoint
- `OUTPUT_DIR`: directory to save generated videos
- `GPU_IDS`: GPU id(s) to use

The script also sets the control-signal paths below; update them if needed:

- `--validation_camera_dir`: camera JSON file or directory of `.json` files (default: `example_scripts/example_data/camera.json`)
- `--validation_timing_dir`: timing `.npy` / `.txt` file or directory (default: `example_scripts/example_data/slow_motion.npy`)

Then launch from the repo root:

```bash
bash example_scripts/eval.sh
```

Key flags (already configured in `example_scripts/eval.sh`):

- `--do_validation true` — run generation
- `--only_validation true` — skip training and run inference only
- `--data_flag "real_world"` / `--validation_data_flag "real_world"`
- `--num_inference_steps 50` — diffusion sampling steps
- `--train_resolution "81x384x640"` — output resolution as `frames x height x width`

The model generates one output video for each `(camera, timing)` pair. Results are saved to `{output_dir}/my_validate/`.


---

## Training

### 1. Training Dataset

Download the training dataset from Hugging Face:

- **Training dataset**: [`19reborn/bulletTime_dataset`](https://huggingface.co/datasets/19reborn/bulletTime_dataset)

### 2. Configure data and output paths

In [`example_scripts/train.sh`](example_scripts/train.sh), set:

- `DATA_ROOT`: path to training data (use `--data_flag "simulator"` for the released synthetic dataset)
- `VALIDATION_DIR`: path to validation data (can be the same as `DATA_ROOT`, or real-world videos prepared as in the Inference section above)
- `OUTPUT_DIR`: directory to save checkpoints and logs

> Make sure `DATA_ROOT` and `VALIDATION_DIR` point to valid local paths.

### 3. Launch training

Edit the distributed settings at the top of [`example_scripts/train.sh`](example_scripts/train.sh):

- `NUM_PROCESSES`: number of processes (typically equals the number of GPUs)
- `GPU_IDS`: comma-separated GPU ids (e.g., `"0,1"`)

For DeepSpeed settings, edit [`finetune/accelerate_config.yaml`](finetune/accelerate_config.yaml).

Then launch from the repo root:

```bash
bash example_scripts/train.sh
```

---

## Dataset Simulator

We also provide the source code to generate our synthetic dataset with disentangled 4D controls: [19reborn/BulletTime_dataSimulator](https://github.com/19reborn/BulletTime_dataSimulator).


## Note on Implementation

In this release, camera and time AdaLN are applied only before attention; the post-attention AdaLN blocks are omitted. We found this design achieves **comparable performance with roughly half added parameters**.

## Acknowledgements

We thank the authors of [CogVideoX](https://github.com/THUDM/CogVideo), [ReCamMaster](https://github.com/KwaiVGI/ReCamMaster) for their open-source contributions. 



## Citation

If you find this repository useful for your research, please cite:

```bibtex
@inproceedings{wang2026bullettime,
  title   = {Bullettime: Decoupled control of time and camera pose for video generation},
  author  = {Wang, Yiming and Zhang, Qihang and Cai, Shengqu and Wu, Tong and Ackermann, Jan and Kuang, Zhengfei and Zheng, Yang and Raji{\v{c}}, Frano and Tang, Siyu and Wetzstein, Gordon},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages   = {18319--18330},
  year    = {2026}
}
```

