from collections import OrderedDict
# from curses import A_ALTCHARSET
# from tkinter import OUTSIDE
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch import nn
from timm.models.layers import drop, drop_path, trunc_normal_
# from mmseg.models.builder import BACKBONES

# from mmseg.models.backbones import ResNet
# from mmseg.models.backbones import VisionTransformer as MMVisionTransformer

from timm.models.resnet import ResNet as TimmResNet
from timm.models.resnet import Bottleneck as TimmBottleneck

from functools import reduce
from operator import mul
import pdb
import math
from .ImageEncoderUtils import *
import torch.nn.init as init
# from mmcv.cnn import build_norm_layer
class VPTCLIPVisionTransformerWithEdge(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512, 
                 drop_path_rate=0.0, out_indices=[3, 5, 7, 11], pretrained=None, get_embeddings=False, 
                 num_tokens=20, prompt_dim=512, total_d_layer=11, norm_cfg=None,align_corners = False,
                 use_multi_scale_norm=True,mid_layers = [3, 5, 7, 11],**kwargs):
        super().__init__()
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        ## Setting of visual prompt tuning
        self.num_tokens = num_tokens 
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)

        self.get_embeddings = get_embeddings
        self.num_layers = layers

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        # pdb.set_trace()
        self.embed_dim = width

        ## Add the prompt parameters # exclude_key=prompt:
        self._init_prompt(patch_size, self.num_tokens, self.prompt_dim, self.total_d_layer)
        self.mid_layers = mid_layers
        for i in range(self.num_layers):
            if use_multi_scale_norm:
                norm = nn.LayerNorm(output_dim)
            else:
                norm = nn.Identity()
            setattr(self, f"multi_scale_norm{i}", norm)
        self.edgeocr_cls_head = nn.Conv2d(
            self.output_dim, 1, kernel_size=1, stride=1, padding=0,
            bias=True)
        _, self.edge_bn = nn.BatchNorm2d(1)
    
    def _init_edge(self):
        # init conv
       # Xavier/Glorot 初始化
        init.xavier_uniform_(self.edgeocr_cls_head.weight)
        # 对偏置使用常数初始化
        if self.edgeocr_cls_head.bias is not None:
            init.constant_(self.edgeocr_cls_head.bias, 0)
        # init bn
        self.edge_bn.weight.data.fill_(1)
        if self.edge_bn.bias is not None:
            self.edge_bn.bias.data.fill_(0)

    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        if total_d_layer >= 0:
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if total_d_layer > 0:  # noqa
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

        else: # total_d_layer < 0
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(abs(total_d_layer), num_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)
            

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        self._init_edge()
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    # (1025, 768)                      (197, 768)  
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    
                    spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear',align_corners=self.align_corners)
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size*self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear',align_corners=self.align_corners)
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)

        if self.total_d_layer >=0:
            # concat prompt
            x = torch.cat((
                x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)

        x = x.permute(1, 0, 2)

        features = []
        outs = []
        if self.total_d_layer == 0: #shallow
            for i, blk in enumerate(self.transformer.resblocks):
                x = blk(x)
                if len(self.out_indices) > 1:
                    if i in self.out_indices:
                        xp = x.permute(1, 0, 2)[:, 1+self.num_tokens:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                        features.append(xp.contiguous())
        elif self.total_d_layer > 0: # deep
            x, features,multi_scale_features = self.forward_deep_prompt(x, features, H, W)
        elif self.total_d_layer < 0:
            x, features = self.forward_reverse_deep_prompt(x, features, H, W)
        else:
            AttributeError('Input correct total_d_layer')

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, -(H*W):].reshape(B, H, W, -1).permute(0, 3, 1, 2)
            
            if len(self.out_indices) == 1:
                visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True)
                features.append(visual_embedding)

            outs.append(tuple(features))
            global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True)
            outs.append(global_embedding)
            # import pdb
            # pdb.set_trace()
            edge_input = visual_embedding.contiguous()
            edge_feature = self.edge_bn(self.edgeocr_cls_head(edge_input))  # b,1,32,32
            outs.append(edge_feature)
            outs.append(multi_scale_features)
        return outs


    def forward_deep_prompt(self, embedding_output, features, H, W, out_last=False):
        multi_scale_features = []
        B = embedding_output.shape[1]

        for i in range(self.num_layers):
            multi_scale_norm = getattr(self, f"multi_scale_norm{i}")
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
                # feat = hidden_states[-(H * W):, :, :].permute(1,0,2).contiguous() # b,1024,768
                # feat = hidden_states.permute(1,0,2) @ self.proj
                # feat = feat[:, -(H * W):, :].contiguous() # b,1024,512
                # feat = multi_scale_norm(feat).permute(1,0,2).contiguous()
                # multi_scale_features.append(feat) # 1024,b,768
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0)

                hidden_states = self.transformer.resblocks[i](hidden_states)
                # pdb.set_trace()
                
            else:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[-(H*W):, :, :]
                ), dim=0)
                hidden_states = self.transformer.resblocks[i](hidden_states)
            if i in self.mid_layers:
                # pdb.set_trace()
                feat = hidden_states.permute(1,0,2) @ self.proj
                feat = feat[:, -(H * W):, :].contiguous() # b,1024,512
                feat = multi_scale_norm(feat).permute(1,0,2).contiguous()
                multi_scale_features.append(feat)  
            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = hidden_states.permute(1, 0, 2)[:, -(H*W):, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())
            
            if i == (self.num_layers-2): #10
                before_last_feats = self.prompt_norm(hidden_states)

        multi_scale_features= torch.stack(multi_scale_features,dim=0) # l,1024,b,512
        # pdb.set_trace()
        encoded = self.prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features,multi_scale_features

    def forward_reverse_deep_prompt(self, embedding_output, features, H, W, out_last=False):
        B = embedding_output.shape[1]
        deep_num_no = (12 - self.deep_prompt_embeddings.shape[0])-1

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output) 
            elif 0<i<=deep_num_no:
                hidden_states = self.transformer.resblocks[i](hidden_states) 
            else: ## with deep prompts
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-deep_num_no-1]).expand(B, -1, -1)).permute(1, 0, 2)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[-(H*W):, :, :]
                ), dim=0)

                hidden_states = self.transformer.resblocks[i](hidden_states)
            
            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = hidden_states.permute(1, 0, 2)[:, -(H*W):, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())
            
            if i == (self.num_layers-2):
                before_last_feats = self.prompt_norm(hidden_states)

        encoded = self.prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features



if __name__ =="__main__":
    model = VPTCLIPVisionTransformerWithEdge(patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=512,
        out_indices=[11],
        #setting of vpt
        num_tokens=10,
        prompt_dim=768,
        total_d_layer=11,
        mid_layers = [3,5,7,11])
    
    image = torch.randn(1, 3, 512, 512)
    model.DAPM_replace(DPAM_layer = 20)