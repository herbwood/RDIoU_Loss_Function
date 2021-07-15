import warnings
from torch import nn
from feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

import misc as misc_nn_ops
from _utils import IntermediateLayerGetter
import resnet

# Adds a FPN on top of a model.
class BackboneWithFPN(nn.Module):

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

# Constructs a specified ResNet backbone with FPN on top. 
# Freezes the specified number of layers in the backbone.
def resnet_fpn_backbone(
    backbone_name,
    pretrained,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None):

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)

    # select layers that wont be frozen
    # if trainable_layer == 3 only train layer4, layer3, layer2 
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    if trainable_layers == 5:
        layers_to_train.append('bn1')
    
    # freeze layers 
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    # input channel list 
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


def _validate_trainable_layers(pretrained, trainable_backbone_layers, max_value, default_value):
    # dont freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable".format(max_value))
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers

if __name__ == "__main__":
    # usage example 
    import torch

    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)

    # get some dummy image
    x = torch.rand(1,3,64,64)

    # compute the output
    output = backbone(x)

    # output example 
    # returns
    # [('0', torch.Size([1, 256, 16, 16])),
    # ('1', torch.Size([1, 256, 8, 8])),
    # ('2', torch.Size([1, 256, 4, 4])),
    # ('3', torch.Size([1, 256, 2, 2])),
    # ('pool', torch.Size([1, 256, 1, 1]))]
    print([(k, v.shape) for k, v in output.items()])