# Transformer-CV-models
Recent Transformer-based CV and related works on PaddlePaddle 2.0

## 1. 复现教程

- [模型复现方法总结](https://github.com/PaddleEdu/Transformer-CV-models/blob/main/docs/model_reproduction_skills.md)
- [常用功能模块](https://github.com/PaddleEdu/Transformer-CV-models/blob/main/docs/utils.md)

欢迎大家贡献复现过程中的技巧。

## 2. 复现候选Paper

复现Paper List如下，分两部分：Vision Transformer系列、其他CV模型

### 2.1 Vision Transformer系列模型

| 编号 | 论文 | 参考实现 | 数据集 | 验收要求 |
| ---| --- | --- | --- | --- |
| 01 | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | [DETR](https://github.com/facebookresearch/detr) | [COCO2017](https://cocodataset.org/#download) | 在指定数据集上，DETR-R50: mAP >= 42.0; DETR-DC5-R50: mAP>=43.3 |
| 02 | [DEFORMABLE DETR: DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION](https://arxiv.org/abs/2010.04159) | [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) | [COCO2017](https://cocodataset.org/#download) | 在指定数据集上，论文中不带两个加强trick的版本 >= 43.8mAP |
| 03 | [Generative Adversarial Transformers](https://arxiv.org/pdf/2103.01209.pdf) | [GANsformer](https://github.com/dorarad/gansformer) | CLEVR,LSUN-Bedroom ,FFHQ,Cityscapes任选其一 | CLEVR:9.24  LSUN-Bedroom: 6.15  FFHQ:7.42  Cityscapes:5.23 |
| 04 | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | [COCO2017](https://cocodataset.org/#download) | cascade mask rcnn + swin B, bbox mAP=51.9 mask mAP=45.0 |
| 05 | [Co-Scale Conv-Attentional Image Transformers](https://arxiv.org/abs/2104.06399) | [CoaT](https://github.com/mlpc-ucsd/CoaT) | [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/2012-downloads)  | CoaT-Lite Small Acc@1 81.9 |
| 06 | [Token Labeling: Training a 85.4% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet](https://arxiv.org/abs/2104.10858) | [TokenLabeling](https://github.com/zihangJiang/TokenLabeling) | [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/2012-downloads)  | LV-ViT-M(448) Acc@1 85.5 |
| 07 | [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370) | [big_transfer](https://github.com/google-research/big_transfer) | [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/2012-downloads) | Bit-L Acc@top1 87.54 |
| 08 | [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239) | [Cait](https://github.com/facebookresearch/deit) | [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/2012-downloads) | CAIT-XS-24 Acc@top1 84.1 |
| 09 | [TransReID: Transformer-based Object Re-Identification](https://arxiv.org/pdf/2102.04378.pdf) | [TransReID](https://github.com/heshuting555/TransReID) | MSMT17 | mAP 69.4% / R1 86.2% |

### 2.2 其他CV模型(持续更新中)

| 编号 | 论文 | 参考实现 | 数据集 | 验收要求 |
| ---| --- | --- | --- | --- |
| 01 | [SOGNet: Scene Overlap Graph Network for Panoptic Segmentation](https://arxiv.org/pdf/1911.07527.pdf) | [SOGNet](https://github.com/LaoYang1994/SOGNet) | [COCO2017](https://cocodataset.org/#download) | 在指定数据集上,PQ>=43.7 |
| 02 | [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf) | [DetectoRS](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors) | [COCO2017](https://cocodataset.org/#download) | 在指定数据集上，DetectoRS Cascade R50 mAP=47.4 |
| 03 | [Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning](https://arxiv.org/abs/1904.01786) | [softRas](https://github.com/ShichenLiu/softRas) | shapenet-13 categories (airplane, bench, dresser, car, chair, display, lamb, speaker, rifle, sofa, table, phone, vessel) | 在指定数据集上，mean IoU >= 64.64% |
| 04 | [ABCNet：Real-time Scene Text Spotting with Adaptive Bezier-Curve Network](https://arxiv.org/abs/2002.10200) | [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) | [Total-Text](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset) | 在指定数据集上，ABCNet-MS：（none）hmean=69.5，（full strong）hmean=78.4，fps=6.9;ABCNet-F：（none）hmean=61.9，（full strong）hmean= 74.1，fps=22.8 |
| 05 | [RepPoints V2: Verification Meets Regression for Object Detection](https://arxiv.org/abs/2007.08508) | [RepPointsV2](https://github.com/Scalsol/RepPointsV2) | [COCO2017](https://cocodataset.org/#download) | 在指定数据集上，R50_FPN COCO val mAP=41.1 |
| 06 | [Recurrent Residual Network for Video Super-resolution](https://arxiv.org/pdf/2008.05765.pdf) | [RRN](https://github.com/junpan19/RRN) | 取消 | 取消 | 
| 07 | [DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/pdf/1908.03826.pdf) | [DeblurGAN-v2](https://github.com/VITA-Group/DeblurGANv2) |  [GoPro dataset+DVD dataset+NFS dataset](https://)  | 参考对应Github | 
| 08 | [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://arxiv.org/pdf/2103.17249.pdf) | [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) |  [CelebA-HQ](https://)  | 人眼主观评价 | 
| 09 | [Multi-Stage Progressive Image Restoration](https://arxiv.org/pdf/2102.02808.pdf) | [Multi-Stage]() | 参考论文 | 达到论文PSNR指标 | 