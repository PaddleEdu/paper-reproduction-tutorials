# 常用功能模块
## 介绍
* 在 utils 包中包含一些常用的功能模块
* 下面为简单介绍，具体用法可以参考 Vision Transformer/vit.py
## 网络层
* Identity：占位层
* DropPath：一种 Dropout 层

    ```python
    from utils import Identity
    model = Identity()
    ```

## 常用函数
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

## 常用初始化
* ones_：全一初始化
* zeros_：全零初始化
* trunc_normal_：trunc_normal 初始化（std=0.2）

    ```python
    from utils import zeros_
    zeros_(layer.bias)
    ```
