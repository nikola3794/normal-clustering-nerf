# Surface Normal Clustering for Implicit Representation of Manhattan Scenes
This repository contains the implementation of the method proposed in ICCV23 paper ["Nikola Popovic, Danda Pani Paudel, Luc Van Gool - Surface Normal Clustering for Implicit Representation of Manhattan Scenes"](https://arxiv.org/abs/2212.01331).

![teaser](https://github.com/nikola3794/normal-clustering-nerf/blob/main/teaser.jpg)

# Abstract
Novel view synthesis and 3D modeling using implicit neural field representation are shown to be very effective for calibrated multi-view cameras. Such representations are known to benefit from additional geometric and semantic supervision. Most existing methods that exploit additional supervision require dense pixel-wise labels or localized scene priors. These methods cannot benefit from high-level vague scene priors provided in terms of scenes' descriptions. In this work, we aim to leverage the geometric prior of Manhattan scenes to improve the implicit neural radiance field representations. More precisely, we assume that only the knowledge of the indoor scene (under investigation) being Manhattan is known -- with no additional information whatsoever -- with an unknown Manhattan coordinate frame. Such high-level prior is used to self-supervise the surface normals derived explicitly in the implicit neural fields. Our modeling allows us to cluster the derived normals and exploit their orthogonality constraints for self-supervision. Our exhaustive experiments on datasets of diverse indoor scenes demonstrate the significant benefit of the proposed method over the established baselines.

# Python environment
```
conda create --name ENV_NAME python=3.8.5
conda activate ENV_NAME
cd REPO_DIR
pip install --ignore-installed torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# Had to comment out some line for this to work: https://github.com/NVIDIA/apex/pull/323
pip install plotly==5.10.0
pip install imgviz
pip install h5py
pip install pandas
pip install wandb
# The line below needs to be called every time the cuda scripts are modified
pip install models/csrc/
```

# Data
Under preparation...
### Hypersim
### ScanNet
### Replica

# Train
Under preparation...

# Evaluate
Under preparation...

# Code structure
Under preparation...

# Contact
Under preparation...

# Citation
Under preparation...
