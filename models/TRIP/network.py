import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba.mamba_ssm import SRMamba

custom_config = {'base'      : {'modal': 'path',
                                'loss': 'CLSLoss',
                                'optimizer': 'Adam',
                                'lr': 2e-4,
                                'wsi_norm': False,
                               },
                 'customized': {'act': {'type': str, 'default': 'relu'},
                                'dropout': {'type': float, 'default': 0.25},
                               },
                }
                
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        layer = 2
        self.rate = 10
        feat_dim = 256

        self._fc1 = nn.Sequential(*[nn.Linear(1024, feat_dim), nn.GELU(), nn.Dropout(0.1)])
        self.norm = nn.LayerNorm(feat_dim)
        self.affine_matrix_mean = nn.Parameter(torch.zeros(1, 1, feat_dim, device='cuda'), requires_grad=True)
        self.affine_matrix_std = nn.Parameter(torch.ones(1, 1, feat_dim, device='cuda'), requires_grad=True)
        
        self.layers = nn.ModuleList()
        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(feat_dim),
                    SRMamba(
                        d_model=feat_dim,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,
                    ),
                    )
            )
            
        self.layers_rev = nn.ModuleList()
        for _ in range(layer):
            self.layers_rev.append(
                nn.Sequential(
                    nn.LayerNorm(feat_dim),
                    SRMamba(
                        d_model=feat_dim,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,
                    ),
                    )
            )

        self.affine_tta_mean = []
        self.affine_tta_std = []
        for i in range(4):
            self.affine_tta_mean.append(nn.Parameter(torch.zeros(1, 1, feat_dim, device='cuda'), requires_grad=True))
            self.affine_tta_std.append(nn.Parameter(torch.ones(1, 1, feat_dim, device='cuda'), requires_grad=True))
        
        self.n_classes = args.n_classes
        self.classifier = nn.Linear(feat_dim, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.apply(initialize_weights)

    def forward(self, x_path, phase, label=None):
        out_dict = {}
        h = self._fc1(x_path)  # [B, n, 256]

        for idx, layer in enumerate(self.layers):
            h_ = h
            h_ = layer[0](h_)
            h_ = (h_ - self.affine_tta_mean[idx]) / (self.affine_tta_std[idx] + 1e-10) 
            h_ = layer[1](h_, rate=self.rate)
            h0 = h + h_

        for idx, layer in enumerate(self.layers_rev):
            hr_ = torch.flip(h, dims=[1])
            hr_ = layer[0](hr_)
            hr_ = (hr_ - self.affine_tta_mean[idx+2]) / (self.affine_tta_std[idx+2] + 1e-10) 
            hr_ = layer[1](hr_, rate=self.rate)
            h1 = h + hr_
            
        h = (torch.flip(h1, dims=[1]) + h0) / 2

        h = self.norm(h)
        h = (h - self.affine_matrix_mean) / (self.affine_matrix_std + 1e-10)
        Att = self.attention(h) # [B, n, K]
        A = F.softmax(Att, dim=1) 
        A = torch.transpose(A, 1, 2) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)

        logits = self.classifier(h)  # [B, n_classes]
        out_dict['att'] = Att
        out_dict['feature'] = h
        out_dict['pred'] = logits
        return out_dict
    