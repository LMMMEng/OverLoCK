# Applying OverLoCK to Semantic Segmentation   

## 1. Requirements

```
pip install mmcv-full==1.7.2 --no-cache-dir
pip install mmsegmentation==0.30.0 --no-cache-dir
```
💡 To enable torch>=2.1.0 to support mmcv 1.7.2, you need to make the following changes:  
> 1️⃣ https://goo.su/XhU5vWr     
> 2️⃣ https://goo.su/ogm4yO



## 2. Data Preparation

Prepare ADE20K dataset according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md).  

## 3. Main Results on ADE20K using UperNet framework

|    Backbone   |   Pretrain  | Schedule | mIoU |                         Config                          | Download |
|:-------------:|:-----------:|:--------:|--------|:-------------------------------------------------------:|:----------:|
| OverLoCK-T | [ImageNet-1K](https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_t_in1k_224.pth)|    160K    |  50.3     |    [config](configs/overlock/upernet_overlock_tiny_ade20k_8xb2.py)    |[model](https://github.com/LMMMEng/OverLoCK/releases/download/v1/upernet_overlock_tiny_ade20k.pth)          |
| OverLoCK-S | [ImageNet-1K](https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_s_in1k_224.pth)|    160K    |51.3       |    [config](configs/overlock/upernet_overlock_small_ade20k_8xb2.py)    |[model](https://github.com/LMMMEng/OverLoCK/releases/download/v1/upernet_overlock_small_ade20k.pth)           |
| OverLoCK-B | [ImageNet-1K](https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_b_in1k_224.pth) |    160K    |51.7        |    [config](configs/overlock/upernet_overlock_base_ade20k_8xb2.py)    |[model](https://github.com/LMMMEng/OverLoCK/releases/download/v1/upernet_overlock_base_ade20k.pth)           |

## 4. Train
To train ``OverLoCK-T + UperNet`` model on ADE20K dataset with 8 gpus (single node), run:
```
bash scripts/dist_train.sh configs/overlock/upernet_overlock_tiny_ade20k_8xb2.py 8
```

## 5. Validation
To evaluate ``OverLoCK-T + UperNet`` model on ADE20K dataset, run:
```
bash scripts/dist_test.sh configs/overlock/upernet_overlock_tiny_ade20k_8xb2.py path-to-checkpoint 8 --eval mIoU
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
