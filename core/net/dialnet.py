from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls

from core.net.deconv_head import DeconvHead
from core.net.resnet import resnet_backbone_18


class DialNet(nn.Module):
    def __init__(self):
        super(DialNet, self).__init__()

        self.resnet_backbone = resnet_backbone_18()
        self.heat_map_head = DeconvHead(output_channels=5)
        self.tag_map_head = DeconvHead(output_channels=4)

    def forward(self, x):
        x = self.resnet_backbone(x)
        heat_map = self.heat_map_head(x)
        tag_map = self.tag_map_head(x)

        return heat_map, tag_map

    def init_resnet_backbone(self):
        origin_resnet_dict = model_zoo.load_url(model_urls["resnet18"])
        origin_resnet_dict.pop('fc.weight', None)
        origin_resnet_dict.pop('fc.bias', None)
        self.resnet_backbone.load_state_dict(origin_resnet_dict)
