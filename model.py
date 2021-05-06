import torch
import torch.nn as nn

import yolo_decoder as yolo_model
from midas.midas_decoder import MidasDecoder
from planercnn.planercnn_decoder import MaskRCNN

def _make_resnet_encoder(use_pretrained):
	pretrained = _make_pretrained_resnext101_wsl(use_pretrained)

	return pretrained

def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
	resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
	return _make_resnet_backbone(resnet)

class Model_Head(nn.Module):
    def __init__(self, cfg , planercfg ,non_negative=True):

        super(Model_Head, self).__init__()

        self.encoder = _make_resnet_encoder(True)
        self.midas_decoder = MidasDecoder()
        self.yolo_conv1 = nn.Conv2d( 2048, 1024, 1, padding= 0)
        self.yolo_conv2 = nn.Conv2d( 1024, 512, 1, padding= 0)
        self.yolo_conv3 = nn.Conv2d( 512, 256, 1, padding= 0)
      
        self.yolo_decoder=yolo_model.Darknet(cfg)

        self.planer_decoder= MaskRCNN(planercfg, self.encoder )


    # def midas_encoder(self, x):
    #     ##  encoder head
    #     layer_1 = self.pretrained.layer1(x)
    #     layer_2 = self.pretrained.layer2(layer_1)
    #     layer_3 = self.pretrained.layer3(layer_2)
    #     layer_4 = self.pretrained.layer4(layer_3)
    #     return layer_1, layer_2, layer_3, layer_4


    # def midas_decoder(self, layer_1, layer_2, layer_3, layer_4):

    #     midas_out = self.scratch.output_conv(midas_path_1)
    #     return torch.squeeze(midas_out, dim=1)

    # def yolo_decoder(self,Yolo_36, Yolo_61, Yolo_75):
    #     ## Yolo_branch
    #     # y_1024 = self.yolo_conv2(layer_4)
    #     # y_512 = self.yolo_conv3(layer_3)
    #     # y_256 = self.yolo_conv4(layer_2)
        
    #     yolo_out= self.yolo_decoder_old.forward( Yolo_75, Yolo_61,Yolo_36 )
    #     return yolo_out

    # def planercnn_decoder(self,x):
        
    #     layer_1, layer_2, layer_3, layer_4= self.midas_encoder(x)
    #     layers= [layer_1, layer_2, layer_3, layer_4]

    #     planer_out= maskrcnn(x, layers)
    #     return planer_out

    def forward(self, yolo_inp, plane_inp):
        
        x = yolo_inp
        plane_inp['input'][0] = yolo_inp

        layer_1 = self.encoder.layer1(x)
        layer_2 = self.encoder.layer2(layer_1)
        layer_3 = self.encoder.layer3(layer_2)
        layer_4 = self.encoder.layer4(layer_3)

        Yolo_75 = self.yolo_conv1(layer_4)
        Yolo_61 = self.yolo_conv2(layer_3)
        Yolo_36 = self.yolo_conv3(layer_2)

        if not self.training:
          inf_out, train_out = self.yolo_decoder(Yolo_75,Yolo_61,Yolo_36)
          yolo_out=[inf_out, train_out]
        else:
          yolo_out = self.yolo_decoder(Yolo_75,Yolo_61,Yolo_36)

        midas_out = self.midas_decoder(layer_1, layer_2, layer_3, layer_4)

        
        # yolo_out = self.yolo_decoder(Yolo_75, Yolo_61,Yolo_36)

        
        planer_out= self.planer_decoder.forward(plane_inp, layer_1, layer_2, layer_3, layer_4)
        return  planer_out, yolo_out, midas_out