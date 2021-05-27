# 复现技巧

## 入门
**paddle-pytorch API对应表**，`dim→axis, Conv2d→Conv2D, BatchNorm2d→BatchNorm2D`

![image](https://user-images.githubusercontent.com/49911294/119772738-3bd35200-bef2-11eb-969f-6606a1e056da.png)



**`torch.nn.ConstantPad2d`->`paddle.nn.Pad2D`**

## 初级

权重初始化
**Pytorch**
```python
for m in self.modules():
if isinstance(m,nn.Conv2d):
    nn.init.kaiming_uniform_(m.weight, a=1)
    nn.init.constant_(m.bias, 0)   
```

**Paddle**
```python
for m in self.sublayers():
if isinstance(m,nn.Conv2D):
    m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.KaimingUniform())
    m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0)) 
```
**Pytorch**
```python
self.body = nn.Sequential(OrderedDict([
    ('block1', nn.Sequential(OrderedDict(
        [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
        [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
    ))),
    ('block2', nn.Sequential(OrderedDict(
        [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
        [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
    ))),
    ('block3', nn.Sequential(OrderedDict(
        [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
        [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
    ))),
    ('block4', nn.Sequential(OrderedDict(
        [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
        [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
    ))),
]))
```

**Paddle**
```python
self.body = nn.Sequential(
    ('block1', nn.Sequential(
        *[('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
         [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
    )),
    ('block2', nn.Sequential(
        *[('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
         [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
    )),
    ('block3', nn.Sequential(
        *[('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
         [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
    )),
    ('block4', nn.Sequential(
        *[('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
         [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
    )),
)
```

**学习率衰减，自定义学习率**
```python
import paddle
from paddle.optimizer.lr import LRScheduler

class StepDecay(LRScheduler):
    def __init__(self,
                learning_rate,
                step_size,
                gamma=0.1,
                last_epoch=-1,
                verbose=False):
        if not isinstance(step_size, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s." %
                type(step_size))
        if gamma >= 1.0:
            raise ValueError('gamma should be < 1.0.')

        self.step_size = step_size
        self.gamma = gamma
        super(StepDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        i = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma**i)

```

**利用`dir`命令查看方法**
```
import paddle

A  = paddle.nn.Linear(10, 10)
B = paddle.to_tensor([1, 2, 3, 4.0])
dir(A), dir(B)
C = paddle.nn.Conv2D(3, 3, 3)
dir(C)
C._in_channels
```

## 中级

**各种`Tensor`切片操作，特别是bool操作**
[bool文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_cn.html#luojiyunsuanfu)

**NumPy**

```python
import numpy as np

np.random.seed(1234)
A_numpy = np.random.randn(10, 1)
A_bool = np.random.randn(10) > 0

A_numpy[A_bool]
```

**Paddle**
```python
A_paddle = paddle.to_tensor(A_numpy)
A_bool = paddle.to_tensor(A_bool)
paddle.masked_select(A_paddle, A_bool.reshape([-1, 1])).reshape([-1, 1])
```
```
array([[ 0.47143516],
       [-1.19097569],
       [ 1.43270697],
       [ 0.88716294],
       [ 0.85958841],
       [-0.6365235 ],
       [ 0.01569637]])
```
**NumPy**
```python
import numpy as np
import torch

np.random.seed(1234)
A_numpy = np.random.randn(10, 4)
A_bool = np.random.randn(10) > 0

A_numpy[A_bool]
```

**Paddle**
```python
A_paddle = paddle.to_tensor(A_numpy)
A_bool = paddle.to_tensor(A_bool)
paddle.masked_select(A_paddle, A_bool.reshape([-1, 1]).astype('float32').multiply(paddle.ones(A_paddle.shape)).astype(bool)).reshape([-1, A_paddle.shape[1]])
```

**利用paddle.gather，paddle.nn.functional.one_hot，单位矩阵来完成复杂切片**

## 高级

**手写函数与方法**
[PaddleIssues](https://github.com/PaddlePaddle/Paddle/issues/32811)，[PSENet](https://aistudio.baidu.com/aistudio/projectdetail/1899550)，**PSENet/models/utils/fuse_conv_bn.py**
```python
class Identity(Layer):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = paddle.randn(128, 20)
        >>> output = m(input)
        >>> print(output.shape)
        [128, 20]

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input
```

**写字典类型数据读取函数**
```python
import random
import paddle


def DataLoader(dataset, batch_size, shuffle=True, drop_last=False):
    # get the dict names
    new_data = dict()
    all_keys = dataset[0].keys()
    for key in all_keys:
        new_data[key] = []
    def reader():
        # get the index list for shuffle
        index_list = list(range(len(dataset)))
        if shuffle:
            random.shuffle(index_list)
        for i in index_list:
            for key in all_keys:
                new_data[key].append(dataset[i][key].unsqueeze(0))
            # a batch
            if len(new_data[key]) == batch_size:
                for key in all_keys:
                    new_data[key] = paddle.concat(new_data[key])
                yield new_data
                for key in all_keys:
                    new_data[key] = []
        # a mini batch
        if len(new_data[key]) > 0:
            if drop_last:
                pass
            else:
                for key in all_keys:
                    new_data[key] = paddle.concat(new_data[key])
                yield new_data
    return reader
```

**本项目的`nms`函数**
```python
"""
Based on numpy NMS
"""
import paddle
import numpy as np


def paddle_nms(bbox, score, thresh):
    """
    nms
    :param dets: ndarray [x1,y1,x2,y2,score]
    :param thresh: int
    :return: list[index]
    """
    dets = np.c_[bbox.numpy(), score.numpy()]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    order = dets[:, 4].argsort()[::-1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        over = (w * h) / (area[i] + area[order[1:]] - w * h)
        index = np.where(over <= thresh)[0]
        order = order[index + 1] # Not include the zero
    return paddle.to_tensor(keep)

 
if __name__ == '__main__':
    boxes=np.array([[100,100,210,210],
                    [250,250,420,420],
                    [220,220,320.0,330],
                    [100,100,210,210],
                    [230,240,325,330],
                    [220,230,315,340]]).astype('float32')
    bbox = paddle.to_tensor(boxes)
    score = paddle.to_tensor([0.6, 0.3, 0.2, 0.1, 0.5, 0.7])
    keep = paddle_nms(bbox, score, 0.5)
    print(keep)
```

## 终极

**自定义算子**（待完善）

