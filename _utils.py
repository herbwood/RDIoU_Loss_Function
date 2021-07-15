from collections import OrderedDict

from torch import nn
from typing import Dict


class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        
        # raise error if layer name not in model 
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)

            # if module name in return layers
            # save module output in out dictionary 
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

if __name__ == "__main__":
    # usage example 
    import torch
    import torchvision

    # download pretrained resnet18
    m = torchvision.models.resnet18(pretrained=True)

    # extract layer1 and layer3, giving as names `feat1` and feat2`
    new_m = IntermediateLayerGetter(m,{'layer1': 'feat1', 'layer3': 'feat2'})
    out = new_m(torch.rand(1, 3, 224, 224))

    # output 
    # [('feat1', torch.Size([1, 64, 56, 56])),
    # ('feat2', torch.Size([1, 256, 14, 14]))]
    print([(k, v.shape) for k, v in out.items()])