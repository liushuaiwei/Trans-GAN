import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet32


# Target Model definition
class CIFAR10_target_net(nn.Module):
    def __init__(self):
        super(CIFAR10_target_net, self).__init__()
        self.model = resnet32()

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # CIFAR10: 3*32*32
        model = [
            nn.Conv2d(image_nc, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            # 32*16*16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 64*8*8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 128*4*4
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # CIFAR10:3*32*32
            nn.Conv2d(gen_input_nc, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*32*32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64*16*16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 128*8*8
        ]

        bottle_neck_lis = [ResnetBlock(128),
                           ResnetBlock(128),
                           ResnetBlock(128),
                           ResnetBlock(128), ]

        decoder_lis = [
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # state size. 64 x 16 x 16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # state size. 32 x 32 x 32
            nn.ConvTranspose2d(32, image_nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # state size. image_nc x 32 x 32
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
