# Transformer-CV-models
Recent Transformer-based CV and related works on PaddlePaddle 2.0

## 复现候选Paper

| 编号 | 论文名称 |
| ---| --- |
| 01 | End-to-End Object Detection with Transformers |
| 02 | DEFORMABLE DETR: DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION |
| 03 | Generative Adversarial Transformers |
| 04 | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows |
| 05 | Co-Scale Conv-Attentional Image Transformers |
| 06 | Token Labeling: Training a 85.4% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet |
| 07 | Big Transfer (BiT): General Visual Representation Learning|
| 08 | Going deeper with Image Transformers |

## 常用功能模块
* 在 utils 包中包含一些常用的功能模块
* 下面为简单介绍，具体用法可以参考 Vision Transformer/vit.py
### 网络层
* Identity：占位层
* DropPath：一种 Dropout 层
### 常用函数
* to_2tuple：int(x) -> (int(x), int(x))
* add_parameter：创建可学习参数
```python
# Paddle
from utils import add_parameter

class VisionTransformer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.pos_embed = add_parameter(self, paddle.zeros((1, num_patches + 1, embed_dim)))


# Pytorch
class VisionTransformer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```
### 常用初始化
* ones_：全一初始化
* zeros_：全零初始化
* trunc_normal_：trunc_normal 初始化（std=0.2）
```python
from utils import zeros_
zeros_(layer.bias)
```