from typing import List
import torch.nn as nn


class VGGDecoder(nn.Module):
    def __init__(self, encoder):
        super(VGGDecoder, self).__init__()
        self.features = self.make_layers(encoder)

    def make_layers(self, encoder):
        features: List[nn.Module] = []
        list_ = encoder.features[7:]
        list_ = list_[::-1]
        for module in list_:
            if isinstance(module, nn.MaxPool2d):
                features.append(nn.MaxUnpool2d(kernel_size=module.kernel_size, stride=module.stride))
            elif isinstance(module, nn.Conv2d):
                conv2d = nn.Conv2d(in_channels=module.out_channels, out_channels=module.in_channels,
                                   kernel_size=module.kernel_size, stride=module.stride, padding=module.padding
                                   )
                features.append(conv2d)
                features.append(nn.BatchNorm2d(conv2d.out_channels))
                features.append(nn.ReLU())

        return nn.Sequential(*features)