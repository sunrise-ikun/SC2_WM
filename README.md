# SC²-WM

**SC²-WM: A Self-Correcting World Model with Closed-Loop Feedback for Vision-and-Language Navigation in Continuous Environments**

Xuan Yao, Yuze Zhu, Junyu Gao, Zongmeng Wang, Changsheng Xu

*Accepted at the 43rd International Conference on Machine Learning (ICML 2026)*

[[Paper]](TODO_PAPER_LINK) [[Project Page]](TODO_PROJECT_PAGE)

---

## Overview

SC²-WM is a framework for Vision-and-Language Navigation in Continuous
Environments (VLN-CE). The agent uses a **world model** to mentally imagine
future panoramic observations, and a **self-correcting spatial cognition**
mechanism to reconcile its imagined and observed views before committing to
a navigation decision. The framework is built on Habitat-Sim and is
evaluated on both R2R-CE and RxR-CE.

## Project Structure

```
SC2-WM/
├── README.md
├── LICENSE
├── DIRECTORY_LAYOUT.txt           Full layout + path index (read me first)
├── environment.yaml               Conda environment specification
├── run.py                         Unified entry: train / eval / inference
│
├── bert_config/                   BERT / XLM-R tokenizer & config (download)
│   ├── bert-base-uncased/
│   └── xlm-roberta-base/
│
├── data/                          Datasets, checkpoints, scenes, runtime logs
│   ├── ViT-B-16.pt                CLIP ViT-B/16 weights
│   ├── ddppo-models/              DDPPO depth encoder
│   ├── scene_datasets/            Matterport3D scenes
│   ├── datasets/                  R2R-CE / RxR-CE episodes
│   ├── checkpoints/               Released SC²-WM model weights
│   └── logs/                      Runtime: tensorboard, eval, ckpts, video
│
├── pretrained/                    R2R-CE backbone & auxiliary modules (download)
├── rxr_pretrained/                RxR-CE backbone & init weights (download)
│
├── habitat_extensions/            Habitat task / sensor / sim extensions
├── run_r2r/                       R2R experiment configs & launch script
│   ├── main.bash
│   ├── iter_train.yaml
│   └── r2r_vlnce.yaml
├── run_rxr/                       RxR experiment configs & launch script
│   ├── main.bash
│   ├── iter_train.yaml
│   └── rxr_vlnce.yaml
├── utils_p/                       Memory module, prompt, losses, metrics
└── vlnce_baselines/               Core model & trainer
    ├── config/                    Default configs
    ├── models/                    Policy, VLN-BERT, CLIP, NeRF, graph utils
    ├── common/                    Base trainer, environments, aux losses
    ├── waypoint_networks/         ResNetUNet (occupancy / segm / waypoint)
    ├── ss_trainer_ETP.py          Main trainer  (TRAINER_NAME = SS-ETP)
    └── dagger_trainer.py          Legacy DAgger trainer
```

A full path-to-file index is in **`DIRECTORY_LAYOUT.txt`**. Each downloadable
directory additionally contains a `PLACE_HERE.txt` describing its expected
contents.

## Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yaml
conda activate sc2-wm
```

> The environment pins Python 3.7, PyTorch 1.12.1 (CUDA 11.3), and
> habitat-sim 0.2.1.

### 2. Install Habitat-Lab

Follow the [Habitat](https://github.com/facebookresearch/habitat-lab) and
[VLN-CE](https://github.com/jacobkrantz/VLN-CE) installation instructions to
install `habitat-lab`. Make sure it is compatible with `habitat-sim==0.2.1`.

### 3. Install extension modules

```bash
# K-nearest feature search
git clone https://github.com/thomgrand/torch_kdtree
cd torch_kdtree
git submodule init && git submodule update
pip install .
cd ..

# Faster MLP inference
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```

## Data Preparation

All external assets are kept under three top-level directories: `data/`,
`pretrained/`, `rxr_pretrained/` (plus `bert_config/` for the language
encoders). None of the asset files are tracked in git — please download them
from the project's release page (Google Drive link below) and drop them into
the matching directory. The expected file names are documented inside each
`PLACE_HERE.txt`.

**Download:** [Google Drive](https://drive.google.com/drive/folders/1hP8oIbn6bcDA3ZNLJLV0hSAPqgFvurxl?usp=drive_link)

Target layout:

```
data/
├── ViT-B-16.pt                                       # CLIP ViT-B/16
├── ddppo-models/
│   └── gibson-2plus-resnet50.pth                     # depth encoder
├── scene_datasets/mp3d/                              # Matterport3D meshes
├── datasets/
│   ├── R2R_VLNCE_v1-2_preprocessed_BERTidx/
│   │   ├── train/train_bertidx.json.gz
│   │   ├── val_seen/val_seen_bertidx.json.gz
│   │   └── val_unseen/val_unseen_bertidx.json.gz
│   └── RxR_VLNCE_v0_enc_xlmr/
│       ├── train/train_guide.json.gz
│       ├── val_seen/val_seen_guide.json.gz
│       └── val_unseen/val_unseen_guide.json.gz
└── checkpoints/
    ├── ckpt.46600.pth                                # R2R fine-tuned
    └── ckpt.45600.pth                                # RxR fine-tuned

pretrained/
├── model_step_100000.pt                              # VLN-BERT (R2R)
├── model_step_82500.pt                               # VLN-BERT (default)
├── segm.pt                                           # image segmenter
├── cwp_predictor.pth                                 # CWP waypoint predictor
├── NeRF_p16_8x8.pth                                  # NeRF module
└── resnet18-f37072fd.pth                             # ResNet-18 backbone

rxr_pretrained/
├── ckpt.iter31100.pth
└── mlm.sap_rxr/ckpts/model_step_90000.pt

bert_config/
├── bert-base-uncased/      (HuggingFace: bert-base-uncased)
└── xlm-roberta-base/       (HuggingFace: xlm-roberta-base)
```

> Matterport3D access requires signing the
> [Terms of Use](https://niessner.github.io/Matterport/).

## Usage

All commands are launched from the project root.

### Training

```bash
bash run_r2r/main.bash train      # R2R-CE
bash run_rxr/main.bash train      # RxR-CE
```

### Evaluation

```bash
bash run_r2r/main.bash eval       # R2R-CE
bash run_rxr/main.bash eval       # RxR-CE
```

### Inference (test-set submission)

```bash
bash run_r2r/main.bash infer      # R2R-CE
bash run_rxr/main.bash infer      # RxR-CE
```

### Custom configuration

You can override any config field on the command line, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --exp_name my_experiment \
    --run-type train \
    --exp-config run_r2r/iter_train.yaml \
    SIMULATOR_GPU_IDS [0] \
    TORCH_GPU_IDS [0] \
    GPU_NUMBERS 1 \
    NUM_ENVIRONMENTS 1 \
    IL.iters 60000 \
    IL.lr 1e-5
```

Selected SC²-WM-specific arguments:

| Argument          | Description                                | Default |
|-------------------|--------------------------------------------|---------|
| `--memory_size`   | Memory bank capacity                        | 1000    |
| `--neighbor`      | Number of nearest neighbours retrieved      | 5       |
| `--prompt_alpha`  | Prompt weighting factor                     | 0.1     |
| `--warm_n`        | Memory warm-up steps                        | 5       |
| `--imagine_T`     | Imagination horizon for the world model     | 2       |

## Results

### R2R-CE (val_unseen)

| Method        | SR | SPL | nDTW |
|---------------|----|-----|------|
| SC²-WM (Ours) | —  | —   | —    |

### RxR-CE (val_unseen)

| Method        | SR | SPL | nDTW |
|---------------|----|-----|------|
| SC²-WM (Ours) | —  | —   | —    |

> See the paper for the complete set of results.

## Citation

```bibtex
@inproceedings{yao2026sc2wm,
    title     = {{$SC^2$-WM}: A Self-Correcting World Model with Closed-Loop Feedback for Vision-and-Language Navigation in Continuous Environments},
    author    = {Yao, Xuan and Zhu, Yuze and Gao, Junyu and Wang, Zongmeng and Xu, Changsheng},
    booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
    year      = {2026}
}
```

## Acknowledgements

Our code is built on top of several excellent open-source projects:

- [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- [CLIP](https://github.com/openai/CLIP)
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

We thank the authors of these projects for releasing their code.

## License

Released under the [MIT License](LICENSE).
