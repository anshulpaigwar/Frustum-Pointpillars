# Frustum-PointPillars: A Multi-Stage Approach for 3D Object Detection using RGB Camera and LiDAR


**Authors: Anshul Paigwar, David Sierra-Gonzalez, Ozgur Erkent, Christian Laugier**

<img src="https://github.com/anshulpaigwar/Frustum-Pointpillars/blob/main/doc/teaser.png" alt="drawing" width="400"/><img src="https://github.com/anshulpaigwar/Frustum-Pointpillars/blob/main/doc/mask.png" alt="drawing" width="400"/>

## Introduction
This repository is code release for our GndNet paper published in IEEE International Conference of Computer Vision, ICCV'2021, Workshop on Autonomous Vehicle Vision. [Link](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Paigwar_Frustum-PointPillars_A_Multi-Stage_Approach_for_3D_Object_Detection_Using_RGB_ICCVW_2021_paper.pdf)

## Abstract
Accurate 3D object detection is a key part of the perception module for autonomous vehicles. A better understanding of the objects in 3D facilitates better decision-making and path planning. RGB Cameras and LiDAR are the most commonly used sensors in autonomous vehicles for environment perception. Many approaches have shown promising results for 2D detection with RGB Images, but efficiently localizing small objects like pedestrians in the 3D point cloud of large scenes has remained a challenging area of research. We propose a novel method, Frustum-PointPillars, for 3D object detection using LiDAR data. Instead of solely relying on point cloud features, we leverage the mature field of 2D object detection to reduce the search space in the 3D space. Then, we use the Pillar Feature Encoding network for object localization in the reduced point cloud. We also propose a novel approach for masking point clouds to further improve the localization of objects. We train our network on the KITTI dataset and perform experiments to show the effectiveness of our network. On the KITTI test set our method outperforms other multi-sensor SOTA approaches for 3D pedestrian localization (Bird’s Eye View) while achieving a significantly faster runtime of 14 Hz.  

<img src="https://github.com/anshulpaigwar/Frustum-Pointpillars/blob/main/doc/fpp-architecture.png" alt="drawing" width="900"/>

## Getting Started

We would like to thank authors of PointPillars and SECOND detector. This repository is forked from nutonomy [PointPillars](https://github.com/nutonomy/second.pytorch) and [SECOND](https://github.com/traveller59/second.pytorch) for KITTI object detection.

### Code Support

ONLY supports python 3.6+, pytorch 1.4 +. Code has only been tested on Ubuntu 18.04.

### Install

#### 1. Clone code

```bash
git clone https://github.com/anshulpaigwar/Frustum-Pointpillars.git
```

#### 2. Install Python packages

<!-- It is recommend to use the Anaconda package manager.

First, use Anaconda to configure as many packages as possible.
```bash
conda create -n pointpillars python=3.7 anaconda
source activate pointpillars
conda install shapely pybind11 protobuf scikit-image numba pillow
conda install pytorch torchvision -c pytorch
conda install google-sparsehash -c bioconda
``` -->

You can use pip or Anaconda package manager to install following packages.
```bash
pip install --upgrade pip
pip install fire tensorboardX shapely pybind11 protobuf scikit-image numba pillow sparsehash
```

Finally, install SparseConvNet. This is not required for PointPillars, but the general SECOND code base expects this
to be correctly configured. 
```bash
git clone git@github.com:facebookresearch/SparseConvNet.git
cd SparseConvNet/
bash build.sh
# NOTE: if bash build.sh fails, try bash develop.sh instead
```

Additionally, you may need to install Boost geometry:

```bash
sudo apt-get install libboost-all-dev
```


<!-- #### 3. Setup cuda for numba

You need to add following environment variables for numba to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
``` -->

#### 4. PYTHONPATH

Add Frustum-PointPillars/ to your PYTHONPATH.

### Prepare dataset

#### 1. Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

Note: PointPillar's protos use ```KITTI_DATASET_ROOT=/data/sets/kitti_second/```.

#### 2. Create kitti infos:

```bash
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

#### 3. Create reduced point cloud:

```bash
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```

#### 4. Create groundtruth-database infos:

```bash
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```

#### 5. Modify config file

The config file needs to be edited to point to the above datasets:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```


### Train

```bash
cd ~/second.pytorch/second
python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* If you want to train a new model, make sure "/path/to/model_dir" doesn't exist.
* If "/path/to/model_dir" does exist, training will be resumed from the last checkpoint.
* Training only supports a single GPU. 
* Training uses a batchsize=2 which should fit in memory on most standard GPUs.
* On a single 1080Ti, training xyres_16 requires approximately 20 hours for 160 epochs.


### Evaluate


```bash
cd ~/second.pytorch/second/
python pytorch/train.py evaluate --config_path= configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* Detection result will saved in model_dir/eval_results/step_xxx.
* By default, results are stored as a result.pkl file. To save as official KITTI label format use --pickle_result=False.
