import paddle
import paddle.nn as nn

from paddle.nn.initializer import TruncatedNormal, KaimingNormal, Constant, Assign


# Common initializations
ones_ = Constant(value=1.)
zeros_ = Constant(value=0.)
kaiming_normal_ = KaimingNormal()
trunc_normal_ = TruncatedNormal(std=.02)


# Common Functions
def to_2tuple(x):
    return tuple([x] * 2)


def add_parameter(layer, datas, name=None):
    parameter = layer.create_parameter(
        shape=(datas.shape),
        default_initializer=Assign(datas)
    )
    if name:
        layer.add_parameter(name, parameter)
    return parameter


# Common Layers
def drop_path(x, drop_prob=0., training=False):
    """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
