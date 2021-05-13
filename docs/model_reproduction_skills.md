# 基于飞桨复现图像分类算法

内容出自[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)团队，感谢支持论文复现活动！喜欢的同学可以登录[PaddleClas Github](https://github.com/PaddlePaddle/PaddleClas)点击Star支持！


## 一、数据集：

ImageNet-1k：目前在图像分类领域，最具代表性的数据集是ImageNet-1k，其预训练权重用在很多下游任务上。所以本次复现是基于该数据集来训练。

## 二、复现方法：

复现方法大致分为前向对齐和训练对齐，前向对齐是训练对齐的前提，训练对齐是最终的目的。在paddlepaddle2.0.0及以上的版本，paddlepaddle的高层API与pytorch的API非常相近，用户可以很快将pytorch的模型转到paddlepaddle。

转换方法有两种，第一种是X2paddle，感兴趣的用户可以移步[X2paddle](https://github.com/PaddlePaddle/X2Paddle)，此处主要介绍第二种方法，即如何将pytorch的代码和权重转换成paddlepaddle的代码和权重。此处提供一个基于SqueezeNet1_0的复现方法：

准备相应的环境：python3.7；pytorch >=1.7；paddle >=2.0.0。安装教程移步[pytorch安装教程](https://pytorch.org/get-started/locally/)、[paddlepaddle安装教程](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

### 2.1.前向对齐：

#### 2.1.1 网络结构代码转换

（1）网络结构代码基于[torchvision中squeezenet](https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py)略加改动（去除暂使用的部分），代码如下：

```python
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from typing import Any

__all__ = ['SqueezeNet', 'squeezenet1_0']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth',
}


class Fire(nn.Module):

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(
        self,
        version: str = '1_0',
        num_classes: int = 1000
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 expected".format(version=version))

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version: str, pretrained: bool, progress: bool, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        global state_dict
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    return _squeezenet('1_0', pretrained, progress, **kwargs)

```

（2）转换后的基于paddlepaddle的代码：

```python
import paddle
import paddle.nn as nn
from typing import Any

__all__ = ['SqueezeNet', 'squeezenet1_0']

class Fire(nn.Layer):

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2D(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2D(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2D(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return paddle.concat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet_torch2paddle(nn.Layer):

    def __init__(
        self,
        version: str = '1_0',
        num_classes: int = 1000
    ) -> None:
        super(SqueezeNet_torch2paddle, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2D(3, 96, kernel_size=7, stride=2),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 expected".format(version=version))

        final_conv = nn.Conv2D(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(),
            nn.AdaptiveAvgPool2D((1, 1))
        )


    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return paddle.flatten(x, 1)


def _squeezenet_torch2paddle(version: str, 
                             pretrained: bool, 
                             progress: bool, **kwargs: Any) -> SqueezeNet_torch2paddle:
    model = SqueezeNet_torch2paddle(version, **kwargs)
    return model


def squeezenet1_0_torch2paddle(pretrained: bool = False, 
                               progress: bool = True, 
                               **kwargs: Any) -> SqueezeNet_torch2paddle:
    return _squeezenet_torch2paddle('1_0', pretrained, progress, **kwargs)

```

**注意事项：**如果有Dropout层，在这里注释掉，否则影响下一步权重转换。

（3）转换原则：

替换pytorch的高层API到Paddle的高层API，部分API可能稍有差异，下面列举了本次转换中有差异的API，仅供参考：

```python
torch.nn.conv2d()->paddle.nn.conv2D()

torch.nn.MaxPool2d()->paddle.nn.MaxPool2D()

torch.nn.AdaptiveAvgPool2d()->paddle.nn.AdaptiveAvgPool2D()

torch.nn.ReLU(inplace=True)->paddle.nn.ReLU()

torch.cat()->paddle.concat()

torch.flatten()->paddle.flatten()
```

其他的差异，可以参考[paddlepaddle官方API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)。



#### 2.1.2权重转换

网络结构的代码转换成功后，下一步就是转换权重。paddlepaddle动态图的权重和pytorch的权重命名方式几乎相同，所以pytorch的权重可以在不用更改名字的情况下直接转换为paddlepaddle的权重。转换代码如下：

```python
from collections import OrderedDict

def load_pytorch_pretrain_model(paddle_model, pytorch_state_dict):

    paddle_weight=paddle_model.state_dict()
    print("paddle num_params:",len(paddle_weight))
    print("torch num_params:", len(pytorch_state_dict))
    new_weight_dict=OrderedDict()

    torch_key_list=[]
    for key in pytorch_state_dict.keys():
        if "num_batches_tracked" in key:
            continue
        torch_key_list.append(key)

    for torch_key, paddle_key in zip(torch_key_list, paddle_weight.keys()):
        print(torch_key, paddle_key, pytorch_state_dict[torch_key].shape,paddle_weight[paddle_key].shape)
        if len(pytorch_state_dict[torch_key].shape) == 0:
            continue
        ##handle all FC weight cases
        if ("fc" in torch_key and "weight" in torch_key) or (len(pytorch_state_dict[torch_key].shape)==2 and pytorch_state_dict[torch_key].shape[0] == pytorch_state_dict[torch_key].shape[1]):
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].cpu().detach().numpy().T.astype("float32")
        elif int(paddle_weight[paddle_key].shape[-1])==int(pytorch_state_dict[torch_key].shape[-1])  :
            new_weight_dict[paddle_key]=pytorch_state_dict[torch_key].cpu().detach().numpy().astype("float32")
        else:
            new_weight_dict[paddle_key] = pytorch_state_dict[torch_key].cpu().detach().numpy().T.astype("float32")
    paddle_model.set_dict(new_weight_dict)
    return paddle_model.state_dict()
```

**注意事项：**

1.FC层的权重需要转置；

2.如果此处torch_key, paddle_key的名字没有对应上，需要写程序一一对应；

#### 2.1.3 手动生成tensor验证模型正确性

在pytorch中，生成全1的tensor，得到输出：

```python
import numpy as np

model = squeezenet1_0(pretrained=True)
model.eval()

img = np.ones([1,3,224,224]).astype("float32")
img =torch.from_numpy(img)
outputs = model(img)
print (outputs)
```

在paddlepaddle中，生成全1的tensor，得到输出：

```python
import numpy as np

paddle_state_dict = load_pytorch_pretrain_model(squeezenet1_0_paddle(), state_dict)
model_paddle = squeezenet1_0_torch2paddle()
model_paddle.set_state_dict(paddle_state_dict)
img = np.ones([1,3,224,224]).astype('float32')
img = paddle.to_tensor(img)
outputs = model_paddle(img)
print (outputs)
```

比较二者输出的差异，若差异很小（万分之一误差内），则视为单张图测试通过，继续下一步；若差异较大，需要逐层打印输出并对比差异，定位差异点，并分析问题所在。

#### 2.1.4保存权重，验证其在ImageNet-1k上的验证准确率

（1）保存权重；

```python
paddle.save(model_paddle.state_dict(),  "squeezenet1_0.pdparams")
```

（2）在PaddleClas中的ppcls/modeling/architectures中添加该网络结构，与此同时在ppcls/modeling/architectures/\_\_init_\_\.py中引用该模型；

（3）在PaddleClas中的dataset中放入ImageNet-1k数据集；

（4）修改PaddleClas中的eval配置文件configs/eval.yaml，如：

```yaml
mode: 'valid'
ARCHITECTURE:
    name: "squeezenet1_0_torch2paddle"

pretrained_model: "./squeezenet1_0"
classes_num: 1000
total_images: 1281167
topk: 5
image_shape: [3, 224, 224]

VALID:
    batch_size: 16
    num_workers: 4
    file_list: "./dataset/ILSVRC2012/val_list.txt"
    data_dir: "./dataset/ILSVRC2012/"
    shuffle_seed: 0
    transforms:
        - DecodeImage:
            to_rgb: True
            to_np: False
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
```

(5)修改tools/eval.sh并执行；

```bash
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/eval.py \
        -c ./configs/eval.yaml \
        -o use_gpu=True
```

```bash
sh tools/eval.sh
```

若最终模型的top-1 acc与pytorch模型的top-1 acc的差异在0.2%之内，视为前向对齐。否则，定位差异较大的输出所对应的图片，逐层定位问题。

### 2.2训练对齐：

一般来说，前向对齐后，需要训练对齐，一来保证论文是可复现的，二来保证模型是可以迁移到下游任务中的。

#### 2.2.1修改配置文件

此处可以根据实际情况修改训练的配置文件，配置文件在PaddleClas中的configs中，比如此处修改configs/SqueezeNet/SqueezeNet1_0.yaml：

```yaml
ARCHITECTURE:
    name: "squeezenet1_0_paddle"
```

#### 2.2.2 训练

修改tools/run.sh,并执行

```python
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/SqueezeNet/SqueezeNet1_0.yaml \
        -o print_interval=10
```

```bash
sh tools/run.sh
```

#### 2.2.3 观察log

在训练初期，观察loss是否持续下降，每一轮的训练top-1 acc是否稳定提升，如果出现异常，可以查看学习率、batch_size 、L2_decay、网络初始化是否与论文保持一致。最终在验证集的top-1 acc至少比目标top-1 acc高0.2%视为训练对齐。

## 三、训练策略

完成前向对齐和训练对齐后，如果想让模型达到更高的精度，可以参考PaddleClas中的[提升模型精度的方法](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/faq_series/faq_2020_s1.md#%E7%AC%AC5%E6%9C%9F)。