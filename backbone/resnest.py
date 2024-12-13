##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""
import os
import sys
import torch
from .resnet import ResNet, Bottleneck
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

from .build import RESNEST_MODELS_REGISTRY

_url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
]}


def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]


resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
                      name in _model_sha256.keys()
                      }


@RESNEST_MODELS_REGISTRY.register()
def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(resnest_model_urls['resnest50']), strict=False)
    return model
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest50'], progress=True, check_hash=True))
    # return model


@RESNEST_MODELS_REGISTRY.register()
def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model


@RESNEST_MODELS_REGISTRY.register()
def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model


@RESNEST_MODELS_REGISTRY.register()
def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model

def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)