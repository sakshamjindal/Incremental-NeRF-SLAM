# Incremental NeRF SLAM

[Project Page](https://www.notion.so/8be942de989d4fc5aaff0ac043aa124c)

## Bridging explicit and implicit representations for SLAM

### **Broad Idea**

Recent techniques (like [iMAP](https://edgarsucar.github.io/iMAP/)) adopt a novel view of SLAM -- that of online learning. The goal of such systems is to build a *representation* (a *map*) of the environment that is suited for navigation and relocalization, fully online (i.e., as a robot explores a new environment).

The iMAP paper uses a *fully implicit* map, i.e., a NeRF is the sole map representation used for mapping and localization. This imposes operational constraints (2 Hz map update rate, 8 Hz localization update rate).

The idea is to build a SLAM *system* that, akin to ORB-SLAM, brings together components that work well in practice (but in the context of implicit representations for SLAM).

### Road Map

1. Implement major parts of the NeRF-SLAM pipeline
    1. ~~Data loader (load in videos)~~
    2. ~~RGB-D NeRF pipeline (i.e., replace RGB image rendering loss with losses proposed in iMAP)~~
    3. ~~Active image sampling~~
    4. Keyframe management logic - -- until this point, assume localization is GT
    5. ~~Localization pipeline using NeRF~~
2. Replace localization pipeline with traditional RGB-D odometry

### Related Papers

1. iMAP: implicit mapping and positioning in real-time [[Project](https://edgarsucar.github.io/iMAP/)] [[PDF](https://arxiv.org/pdf/2103.12352.pdf)]
2. iNeRF: inverting neural radiance fields for pose estimation [[Project](https://yenchenlin.me/inerf/)] [[PDF](https://arxiv.org/pdf/2012.05877.pdf)]
3. Nerf--: Neural radiance fields without known camera parameters [[Project](https://nerfmm.active.vision/)] [[PDF](https://arxiv.org/pdf/2102.07064.pdf)]
4. NeRF: Representing scenes as neural radiance fields for view synthesis [[Project](https://www.matthewtancik.com/nerf)] [[PDF](https://arxiv.org/pdf/2003.08934.pdf)]
5. Fourier features let networks learn high frequency functions in low dimensional domains [[Project](https://bmild.github.io/fourfeat/)] [[PDF](https://arxiv.org/pdf/2006.10739.pdf)]
6. SIREN: Implicit neural representations with periodic activation functions [[Project](https://vsitzmann.github.io/siren/)] [[PDF](https://arxiv.org/pdf/2006.09661.pdf)]
7. KiloNeRF: Speeding up neural radiant

### Useful External Links

- Blogs
    - Understanding TUM dataset ([https://www.programmersought.com/article/51194902722/](https://www.programmersought.com/article/51194902722/))
    - Toolkit for TUM dataset ([https://vision.in.tum.de/data/datasets/rgbd-dataset/tools](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools))
- Papers

## Code details

Unofficial reimplementation of Implicit Mapping and Positioning in Real-Time [(link)](https://edgarsucar.github.io/iMAP/). The project page can be found [here](https://sakshamjindal.notion.site/NeRF-SLAM-a9438af19f0849a1858d4cab7a2d388c) This repo uses [(inverse-NeRF)](https://arxiv.org/abs/2012.05877) for incremental camera pose estimation and simulateously building the map of the scene encoded and decoded by NeRF modules. 

| Update: The official implementation has been released and can be found inside the [(repo)](https://github.com/kxhit/vMAP)

## Code structure

  ```
  main-repository/
  │
  ├── train.py - incremental localization and mapping on TUM dataset
  ├── train_nerf.py - static NeRF training of a scene
  ├── train_inv.py - used inverse-nerf for camera pose estimation of a scene
  │
  ├── utils/ - helper functions for camera pose estimation 
  │   ├── pose_utils.py - borrowed from inverse-nerf implementation [(here)](https://github.com/salykovaa/inerf)
  │   ├── align_trajectory.py - helper function to align trajectories and finding scale using Umeya's method
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │
  ├── models/ - contains implementation of nerf encoder and decoder
  │   ├── lie_group_helpers.py - borrowed from inverse-nerf implementation [(here)](https://github.com/salykovaa/inerf)
  │   ├── nerf_origin.py - original nerf implementation
  │   ├── rendering.py - contains the decoder of the nerf
  │   └── base_trainer.py
  |
  └── datasets/ - helper functions for camera pose estimation 
      ├── TUM_inc.py - dataset/dataloader for incremental localization and mapping on TUM dataset
      ├── llff_TUM.py - dataset/dataloader for static NeRF training of a scene

  ```