# **Center Focusing Network for Real-Time LiDAR Panoptic Segmentation**
![teaser](./imgs/framework.PNG)

Official code for CFNet

> **Center Focusing Network for Real-Time LiDAR Panoptic Segmentation**,
> Xiaoyan Li, Gang Zhang, Boyue Wang, Yongli Hu, Baocai Yin. (https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Center_Focusing_Network_for_Real-Time_LiDAR_Panoptic_Segmentation_CVPR_2023_paper.pdf)
> *Accepted by CVPR2023*

## NEWS

- [2023-02-24] CFNet is accepted by CVPR 2023
- [2022-11-17] CFNet achieves the 63.4 PQ and 68.3 mIoU on the SemanticKITTI LiDAR Panoptic Segmentation Benchmark with the inference latency of 43.5 ms on a single NVIDIA RTX 3090 GPU.
![teaser](./imgs/acc_vs_speed.PNG)

#### 1 Dependency

```bash
CUDA>=10.1
Pytorch>=1.5.1
PyYAML@5.4.1
scipy@1.3.1
```

#### 2 Training Process

##### 2.1 Installation

```bash
cd pytorch_lib
python setup.py install
```

##### 2.2 Prepare Dataset

Please download the [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#overview) dataset to the folder `./data` and the structure of the folder should look like:

```
./data
    ├── SemanticKITTI
        ├── ...
        └── dataset/
            ├──sequences
                ├── 00/         
                │   ├── velodyne/
                |   |	├── 000000.bin
                |   |	├── 000001.bin
                |   |	└── ...
                │   └── labels/ 
                |       ├── 000000.label
                |       ├── 000001.label
                |       └── ...
                ├── 08/ # for validation
                ├── 11/ # 11-21 for testing
                └── 21/
                    └── ...
```

And download the [object bank](https://drive.google.com/file/d/1QdSpkMLixvKQL6QPircbDI_0-GlGwsdj/view?usp=sharing) on the SemanticKITTI to the folder `./data` and the structure of the folder should look like:

```
./data
    ├── object_bank
        ├── bicycle
        ├── bicyclist
        ├── car
        ├── motorcycle
        ├── motorcyclist
        ├── other-vehicle
        ├── person
        ├── truck
```

##### 2.3 Training Script

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --config config/config_mvfcev2ctx_sgd_wce_fp32_lossv2_single_newcpaug.py
```

#### 3 Evaluate Process

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 evaluate.py --config config/config_mvfcev2ctx_sgd_wce_fp32_lossv2_single_newcpaug.py --start_epoch 0 --end_epoch 47
```
