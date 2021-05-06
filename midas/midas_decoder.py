import torch
import torch.nn as nn
from midas.blocks import FeatureFusionBlock, Interpolate, _make_encoder

class MidasDecoder(nn.Module):
    def __init__(self, path=None, features=256, non_negative=True):

        super(MidasDecoder, self).__init__()

        _, self.scratch= _make_encoder(backbone="resnext101_wsl", features=256, use_pretrained=True)
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        #if path:
            #self.load(path)
            #self.load_state_dict(torch.load(path),strict=False)


    def forward(self, layer1, layer2, layer3, layer4):
        ## Midas_branch
        midas_layer_1_rn = self.scratch.layer1_rn(layer1)
        midas_layer_2_rn = self.scratch.layer2_rn(layer2)
        midas_layer_3_rn = self.scratch.layer3_rn(layer3)
        midas_layer_4_rn = self.scratch.layer4_rn(layer4)

        midas_path_4 = self.scratch.refinenet4(midas_layer_4_rn)
        midas_path_3 = self.scratch.refinenet3(midas_path_4, midas_layer_3_rn)
        midas_path_2 = self.scratch.refinenet2(midas_path_3, midas_layer_2_rn)
        midas_path_1 = self.scratch.refinenet1(midas_path_2, midas_layer_1_rn)

        midas_out = self.scratch.output_conv(midas_path_1)
        return torch.squeeze(midas_out, dim=1)
