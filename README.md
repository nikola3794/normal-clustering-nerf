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
1. Download all Hypersim scenes by using the script at ```code_root/datasets/hypersim_src/_utils/``` . Provide necessary arguments when calling the script. Please note that the argument args.contains is already hardcoded in main(), to download only files useful for this repo (each scene has many other accompanying files that are not used, and therefore not downloaded). If you want to download only a subset of scenes (e.g. only 20 scenes from split A), you need to modify the script. One way is to comment out the defined variable URLS = [..] and create a new one with desired scenes.
2. After downloading, the dataset root should contain a folder for every scene ```d_root/scene``` (e.g. "d_root/ai_053_004"). Inside every scene's folder should be a folder ```d_root/scene/_detail``` and a folder ```d_root/scene/images```. Inside ```d_root/scene/images```, there should be ```d_root/scene/images/scene_cam_00_final_hdf5``` where images are stored as hdf5 files, as well as ```d_root/scene/images/scene_cam_00_geometry_hdf5``` where depth maps, normal maps and semantics are stored. Sometimes, a scene has multiple cameras, and there are cam_01, cam_02,... versions of these two folders, which will never be used because only the first camera trajectory is used.
3. The following files contained in ```code_root/datasets/hypersim_src/metadata/``` should be copied to the Hypersim root ```d_root/```: ```hypersim_A_scenes.json```; ```hypersim_B_scenes.json```; ```hypersim_C_scenes.json```; ```all_scenes_metadata.json```; ```most_scenes_list.json```; ```scene_boundaries.json```; ```scene_semantic_classes.json```; ```metadata_camera_parameters.csv```.

### ScanNet
1. Download prepared scenes available at [this repo](https://github.com/zju3dv/manhattan_sdf), and place them into the root of the ScanNet dataset directory.
   
### Replica
1. Download pre-rendered scenes available at [this repo](https://github.com/Harry-Zhi/semantic_nerf), and place them into the root of the Replica dataset directory.

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
