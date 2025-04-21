from torch import nn, Tensor
from .param_mng import parameter_manager
from .quant import QuantConv, replace_submodules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


def convmixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(dim)
        ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


class ConvMixer(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = convmixer(
            dim=dim,
            depth=6,
            kernel_size=5,
            patch_size=2,
            n_classes=10
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class QuantConvMixer(ConvMixer):
    def __init__(self):
        super().__init__()
        replace_submodules(
            self.net,
            predicate=lambda x: isinstance(x, nn.Conv2d),
            func=lambda x: QuantConv(
                in_channels=x.in_channels,
                out_channels=x.out_channels,
                kernel_size=x.kernel_size,
                padding=x.padding,
                dilation=x.dilation,
                groups=x.groups,
                bias=x.bias is not None,
                padding_mode=x.padding_mode,
            )
        )

    def to(self, *args, **kwargs):
        parameter_manager.to(*args, **kwargs)
        return super().to(*args, **kwargs)

