# EdgeConnect edge model (stage 1) as a baseline.
#
# The classes BaseNetwork, EdgeGenerator, Discriminator, ResnetBlock and
# spectral_norm are copied unchanged from EdgeConnect
# (Nazeri et al., ICCVW 2019, https://github.com/knazeri/edge-connect, src/networks.py),
# licensed under CC BY-NC 4.0. Their module names are kept, so official
# EdgeModel_gen.pth / EdgeModel_dis.pth weights load into `.net`.
#
# Wrappers and adaptation to this framework: Copyright (c) 2026 Hassan Saad, CC BY-NC 4.0.
"""EdgeConnect edge generator / discriminator with this repo's interfaces.

    G(img, edge, mask) -> edge logits          (img in [-1, 1], edge in [0, 1], mask 1 = hole)
    D(edge, gray, return_features=False)       (gray image in [0, 1], as in EdgeConnect)

The edge map is first binarised at 0.5 (EdgeConnect was designed for binary Canny edges;
the dataset's edge maps are soft JPEG maps). Input construction then follows
EdgeModel.forward of the original code:
    images_masked = gray * (1 - mask) + mask
    edges_masked  = edge * (1 - mask)
    G input       = cat(images_masked, edges_masked, mask)
The original applies a sigmoid at the end of EdgeGenerator; the wrapper returns
the logits before it (the loss applies the sigmoid), which is numerically the same model.
"""

import torch
import torch.nn as nn

from networks.baselines import binarize_edge


# =============================================================================
# Original EdgeConnect code (src/networks.py), unchanged
# =============================================================================

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


# =============================================================================
# Wrappers for this framework
# =============================================================================

def _load_pth(path):
    try:   # official files also hold the iteration counter -> torch>=2.6 needs weights_only=False
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:   # torch < 1.13
        return torch.load(path, map_location='cpu')


def to_gray01(img):
    """(B, 1|3, H, W) image in [-1, 1] -> (B, 1, H, W) gray in [0, 1]."""
    if img.shape[1] == 3:
        img = img[:, 0:1] * 0.299 + img[:, 1:2] * 0.587 + img[:, 2:3] * 0.114
    return (img.to(torch.float32) + 1.0) / 2.0


class EdgeConnectEdgeGenerator(nn.Module):
    """EdgeConnect EdgeGenerator; returns logits. Resolution independent (fully convolutional).

    binarize_edges: the input edge map is binarised at this threshold (EdgeConnect uses binary
    Canny edges); None keeps it soft.
    """

    def __init__(self, img_resolution=512, residual_blocks=8, use_spectral_norm=True, binarize_edges=0.5):
        super().__init__()
        self.img_resolution = img_resolution
        self.binarize_edges = binarize_edges
        self.net = EdgeGenerator(residual_blocks=residual_blocks, use_spectral_norm=use_spectral_norm)

    def forward(self, img, edge, mask):
        gray = to_gray01(img)
        edge = binarize_edge(edge, self.binarize_edges)
        mask = mask.to(torch.float32)
        images_masked = gray * (1 - mask) + mask
        edges_masked = edge * (1 - mask)
        x = torch.cat((images_masked, edges_masked, mask), dim=1)
        x = self.net.encoder(x)
        x = self.net.middle(x)
        return self.net.decoder(x)          # logits; EdgeGenerator.forward = sigmoid(this)

    def load_official(self, gen_pth):
        """Load an official EdgeModel_gen.pth (e.g. the Places2 release)."""
        data = _load_pth(gen_pth)
        self.net.load_state_dict(data['generator'])


class EdgeConnectDiscriminator(nn.Module):
    """EdgeConnect edge discriminator (70x70 PatchGAN, spectral norm) on cat(gray, edge).

    Returns logits (use_sigmoid=False); the loss applies the non-saturating GAN loss
    with softplus, which equals EdgeConnect's 'nsgan' BCE on sigmoid outputs.
    """

    def __init__(self, img_resolution=512, use_spectral_norm=True):
        super().__init__()
        self.img_resolution = img_resolution
        self.net = Discriminator(in_channels=2, use_sigmoid=False, use_spectral_norm=use_spectral_norm)

    def forward(self, edge, gray, return_features=False):
        logits, features = self.net(torch.cat((gray, edge), dim=1))
        if return_features:
            return logits, features
        return logits

    def load_official(self, dis_pth):
        data = _load_pth(dis_pth)
        self.net.load_state_dict(data['discriminator'])
