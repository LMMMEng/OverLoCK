# Applying OverLoCK to Object Detection and Instance Segmentation

## 1. Requirements

```
pip install mmcv-full==1.7.2 --no-cache-dir
pip install mmdet==2.28.2 --no-cache-dir
```
💡 To enable torch>=2.1.0 to support mmcv 1.7.2, you need to make the following changes:  
> 1️⃣ https://goo.su/XhU5vWr     
> 2️⃣ https://goo.su/ogm4yO


## 2. Data Preparation

Prepare COCO 2017 according to the [guidelines](https://github.com/open-mmlab/mmdetection/blob/2.x/docs/en/1_exist_data_model.md). 

## 3. Main Results on COCO using Mask R-CNN framework

|    Backbone   |   Pretrain  | Schedule | AP_b | AP_m | Config | Download |
|:-------------:|:-----------:|:--------:|--------|:-------:|:------:|:----------:|
| OverLoCK-T | [ImageNet-1K](https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_t_in1k_224.pth)|    1x    |  48.3  |43.3     |[config](configs/maskrcnn_overlock/mask_rcnn_overlock_t_in1k_fpn_1x_coco.py)        |[model](https://github.com/LMMMEng/OverLoCK/releases/download/v1/maskrcnn1x_overlock_tiny_coco.pth)          |
|               |             |    3x    |49.6        |43.9      |[config](configs/maskrcnn_overlock/mask_rcnn_overlock_t_in1k_fpn_3x_coco.py)        |[model](https://github.com/LMMMEng/OverLoCK/releases/download/v1/maskrcnn3x_overlock_tiny_coco.pth)          |
| OverLoCK-S | [ImageNet-1K](https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_s_in1k_224.pth)|    1x    |49.4        |44.0         |[config](configs/maskrcnn_overlock/mask_rcnn_overlock_s_in1k_fpn_1x_coco.py)        |[model](https://github.com/LMMMEng/OverLoCK/releases/download/v1/maskrcnn1x_overlock_small_coco.pth)           |
|               |             |    3x    |51.0        |45.0         |[config](configs/maskrcnn_overlock/mask_rcnn_overlock_s_in1k_fpn_3x_coco.py)        |[model](https://github.com/LMMMEng/OverLoCK/releases/download/v1/maskrcnn3x_overlock_small_coco.pth)          |
| OverLoCK-B | [ImageNet-1K](https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_b_in1k_224.pth) |    1x    |49.9       |44.4         |[config](configs/maskrcnn_overlock/mask_rcnn_overlock_b_in1k_fpn_1x_coco.py)        |[model](https://github.com/LMMMEng/OverLoCK/releases/download/v1/maskrcnn1x_overlock_base_coco.pth)           |
|               |             |    3x    |51.4       |45.3         |[config](configs/maskrcnn_overlock/mask_rcnn_overlock_b_in1k_fpn_3x_coco.py)        |[model](https://github.com/LMMMEng/OverLoCK/releases/download/v1/maskrcnn3x_overlock_base_coco.pth)          |

## 4. Train
To train ``OverLoCK-T + Mask R-CNN 1x`` model on COCO dataset with 8 gpus (single node), run:
```
NUM_GPUS=8
CONFIG=configs/maskrcnn_overlock/mask_rcnn_overlock_t_in1k_fpn_1x_coco.py
bash scripts/dist_train.sh $CONFIG $NUM_GPUS
```

## 5. Validation
To evaluate ``OverLoCK-T + Mask R-CNN 1x`` model on COCO dataset, run:
```
NUM_GPUS=8
CKPT=path-to-checkpoint.pth
CONFIG=configs/maskrcnn_overlock/mask_rcnn_overlock_t_in1k_fpn_1x_coco.py
bash scripts/dist_test.sh $CONFIG $CKPT $NUM_GPUS --eval bbox segm
```

## Citation
If you find this project useful for your research, please consider citing:
```
@inproceedings{lou2025overlock,
  title={OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels},
  author={Lou, Meng and Yu, Yizhou},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  pages={128--138},
  year={2025}
}
```
