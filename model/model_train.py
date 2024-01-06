"""Adapted from https://github.com/facebookresearch/GDT"""


import numpy as np
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from src.vmz import r2plus1d_18, r2plus1d_34
SEED=2022

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class TransposeSqueeze(nn.Module):
    def __init__(self, fdim, tdim):
        super(TransposeSqueeze, self).__init__()
        self.fdim = fdim
        self.tdim = tdim

    def forward(self, x):
        return x.view(-1, self.fdim, self.tdim).transpose(-1,-2)


class Flatten(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Unsqueeze(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(-1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.manual_seed(SEED)
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out') #, nonlinearity='relu')
            if m.bias is not None: 
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.3):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                Unsqueeze(),
                nn.BatchNorm1d(n_hidden),
                Flatten(),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        return self.block_forward(x)


def get_video_feature_extractor(
    vid_base_arch='r2plus1d_18', 
    pretrained=False, 
    duration=1, 
    pre_pool=False, 
):
    if vid_base_arch == 'r2plus1d_18':
        model = r2plus1d_18(pretrained=pretrained, larger_last=False)
        if not pretrained:
            print("Initializing models")
            model.apply(weight_init)
        if pre_pool:
            model.avgpool = nn.Identity()
        else:
            model.avgpool = nn.AdaptiveAvgPool3d((duration, 1, 1))
    elif vid_base_arch == 'r2plus1d_34':
        model = r2plus1d_34(pretrained=pretrained)
        if not pretrained:
            print("Randomy initializing models")
            random_weight_init(model)
        if pre_pool:
            model.avgpool = nn.Identity()
        else:
            model.avgpool = nn.AdaptiveAvgPool3d((duration, 1, 1))
    model.fc = Identity()
    return model



class VideoBaseNetwork(nn.Module):
    def __init__(
        self, 
        vid_base_arch='r2plus1d_18', 
        pretrained=False, 
        norm_feat=False, 
        duration=1, 
        pre_pool=False,
    ):
        super(VideoBaseNetwork, self).__init__()
        self.base = get_video_feature_extractor(
            vid_base_arch, 
            pretrained=pretrained,
            duration=duration,
            pre_pool=pre_pool,
        )
        self.norm_feat = norm_feat

    def forward(self, x):
        x = self.base(x).squeeze()
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=1)
        return x


class GDT(nn.Module):
    def __init__(
        self,
        vid_base_arch='r2plus1d_18', 
        vid2_base_arch='r2plus1d_18',
        pretrained=False, 
        norm_feat=True, 
        use_mlp=False,
        num_classes=256, 
    ):
        super(GDT, self).__init__()
        print('Using GDT model')

        encoder_dim = 512
        encoder_dim_a = 512
        n_hidden = 512

        # Save proprties
        self.use_mlp = use_mlp
        self.norm_feat = norm_feat
        self.encoder_dim = encoder_dim

        self.video_network = VideoBaseNetwork(
            vid_base_arch, 
            pretrained=pretrained
        )
        self.video_network1 = VideoBaseNetwork(
            vid2_base_arch, 
            pretrained=pretrained
        )

        if use_mlp:
            print("Using MLP projection layer")
            self.mlp_v = MLP(
                encoder_dim, num_classes, n_hidden=n_hidden)
            self.mlp_a = MLP(encoder_dim_a, num_classes)
        else:
            print("Using Linear Layer")
            self.mlp_v = nn.Linear(encoder_dim, num_classes)
            self.mlp_a = nn.Linear(encoder_dim_a, num_classes)


    def forward(self, vid, spec, whichhead=0):
        vid_features = self.video_network(vid).squeeze()
        vid2_features = self.video_network1(spec).squeeze()

        if len(vid2_features.shape) == 1:
            vid2_features = vid2_features.unsqueeze(0)
        if len(vid_features.shape) == 1:
            vid_features = vid_features.unsqueeze(0)

        nce_vid_features = self.mlp_v(vid_features)
        nce_vid2_features = self.mlp_a(vid2_features)
        if self.norm_feat:
            nce_vid_features = F.normalize(
                nce_vid_features, p=2, dim=1)
            nce_vid2_features = F.normalize(
                nce_vid2_features, p=2, dim=1)
        return nce_vid_features, nce_vid2_features








def load_model(
    model_type='stica',
    vid_base_arch='r2plus1d_18', 
    vid2_base_arch='r2plus1d_18',
    pretrained=False, 
    norm_feat=True, 
    use_mlp=False,
    num_classes=256, 
    args=None,
):  
    # Cross-modal GDT
    model = GDT(
            vid_base_arch=vid_base_arch, 
            vid2_base_arch=vid_base_arch,
            pretrained=pretrained, 
            norm_feat=norm_feat, 
            use_mlp=use_mlp,
            num_classes=num_classes, 
        )
    return model
