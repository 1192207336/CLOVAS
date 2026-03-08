from collections import OrderedDict
from typing import Tuple, Union,Callable, Optional, Sequence
import numpy as np
import torch
from torch import nn
from CLOVAS_lib.model import * # 
import torch.nn.functional as F

import math
from functools import reduce
from operator import mul
from torch.nn import Dropout

def init_prompts(self, patch, num_tokens, prompt_dim, total_d_layer):
        assert total_d_layer>=0, "total_d_layer should be greater than or equal to 0"
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

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

def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
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

            u, _ = self.load_state_dict(state_dict, False)
            print(f'pretrained image encoder weight loaded.')
            if len(u)>0:
                print(f'{u} are misaligned params in image encoder')

class CLIP_surgery_VisionTransformer(nn.Module):
    def __init__(self, 
                 input_resolution: int, 
                 patch_size: int, 
                 width: int, 
                 layers: int, 
                 heads: int, 
                 output_dim: int,
                 pretrained=None,
                 drop_path_rate=0.,
                 vpt_settings = None,
                 out_indices = [23]
                 ):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.layers = layers

        self.transformer = Transformer(width, layers, heads, need_weights=True)


        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        

        self.vpt_mode = vpt_settings!=None
        self.out_indices = out_indices
        if self.vpt_mode:
            self.num_tokens = vpt_settings['num_tokens']
            self.prompt_dim = vpt_settings['prompt_dim'] # 768
            self.total_d_layer = vpt_settings['total_d_layer']
            self.init_prompts(patch_size, self.num_tokens, self.prompt_dim,self.total_d_layer)
        
        # clip surgery settings
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        
    def init_prompts(self, patch, num_tokens, prompt_dim, total_d_layer):
        assert total_d_layer>=0, "total_d_layer should be greater than or equal to 0"
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

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

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if not pretrained:
            return
        
        if isinstance(pretrained, dict):
            checkpoint = pretrained
        elif isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
        else:
            print("Incorrect pretrained weights format.")
            return
        self.logit_scale = checkpoint['logit_scale']
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
        
        u, _ = self.load_state_dict(state_dict, False)
        print(f'pretrained image encoder weight loaded.')
        if len(u)>0:
            print(f'{u} are misaligned params in image encoder')
    
    @torch.no_grad()
    def apply_CLIP_attention_surgery(self):
        # reform the architecture during first inference
        if self.attn == None:
            # apply architecture surgery on the last 6 blocks
            for i in range(1, 7): # surgery 7, maskclip 2
                self.attn = VV_Attention(out_dim=self.embed_dim, dim=self.embed_dim, num_heads=self.num_heads, qkv_bias=True)
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn
            print("CLIP attention surgery applied.")
    
    def forward(self, x: torch.Tensor):
        # if self.attn_surgery:
        #     self.apply_CLIP_attention_surgery()
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        pos = self.positional_embedding.to(x.dtype)
        # cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        # spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        # spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        # pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x) # b,1370,1024

        

        patch_tokens = []
        if self.vpt_mode:
            assert self.total_d_layer > 0, "total_d_layer should be greater than or equal to 0"
            # concat prompt
            x = torch.cat((
                x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)

            x = x.permute(1, 0, 2)  # NLD -> LND(1045,b,768)
            [x, x_ori],patch_tokens =  self.forward_deep_prompt(x, H, W)
        else:
            x = x.permute(1, 0, 2)  # NLD -> LND
            # [x, x_ori],patch_tokens = self.transformer(x,self.out_indices)
            x_ori, patch_tokens = self.transformer(x, self.out_indices)
        x_ori = x_ori.permute(1, 0, 2)
        #x_ori = self.ln_post(x_ori)
        x_ori = x_ori@ self.proj
        global_embedding = x_ori[:, 0,:]
        visual_embedding = x_ori[:, -(H*W):,:].reshape(B, H, W, -1).permute(0, 3, 1, 2) # x[:, -(H*W):,:].reshape(B, H, W, -1).permute(0, 3, 1, 2)
        global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True)
        visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True)
        patch_token_list = []
        for patch_token in patch_tokens:
            patch_token = self.ln_post(patch_token.permute(1, 0, 2)) @ self.proj  #.contiguous() LND -> NLD
            # patch_token = patch_token[:,-(H*W):,:].contiguous()
            # patch_token = patch_token/patch_token.norm(dim=-1, keepdim=True)
            patch_token_list.append(patch_token)
        patch_tokens = patch_token_list
        
        return global_embedding,visual_embedding,patch_tokens
    
    def forward_deep_prompt(self, input_feat, H, W):
        out_tokens = []
        B = input_feat.shape[1]
        j = 0
        for i in range(self.layers):
            
            if i == 0:
                hidden_states = self.transformer.resblocks[i](input_feat)
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)
                if isinstance(hidden_states, list):
                    x,x_ori = hidden_states
                    x = torch.cat((
                        x[:1, :, :],
                        deep_prompt_emb,
                        x[(1+self.num_tokens):, :, :]
                    ), dim=0)
                    x_ori = torch.cat((
                        x_ori[:1, :, :],
                        deep_prompt_emb,
                        x_ori[(1+self.num_tokens):, :, :]
                    ), dim=0)
                    hidden_states = [x,x_ori]
                else:
                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_tokens):, :, :]
                    ), dim=0)

                hidden_states = self.transformer.resblocks[i](hidden_states)
                
            else:
                if isinstance(hidden_states, list):
                    x,x_ori = hidden_states
                    print(H*W)
                    x = torch.cat((
                        x[:1, :, :],
                        x[-(H*W):, :, :]
                    ), dim=0)
                    x_ori = torch.cat((
                        x_ori[:1, :, :],
                        x_ori[-(H*W):, :, :]
                    ), dim=0)
                    hidden_states = [x,x_ori]
                else:
                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        hidden_states[-(H*W):, :, :]
                    ), dim=0)
                hidden_states = self.transformer.resblocks[i](hidden_states)
            if i+1 in self.out_indices: # len(self.out_indices) > 1 and 
                if isinstance(hidden_states, list):
                    x,x_ori = hidden_states
                    feat = x
                else:
                    feat = hidden_states
                feat = self.prompt_norm(feat)
                feat = torch.cat((
                        feat[:1, :, :],
                        feat[-(H*W):, :, :]
                    ), dim=0) # 1024,b,768
                out_tokens.append(feat)
                j+=1
        return hidden_states,out_tokens

# @BACKBONES.register_module()
class CLIP_plain_VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 pretrained=None,align_corners=False):
        super().__init__()
        self.pretrained=pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.spatial_size = input_resolution // patch_size
        self.transformer = Transformer(width, layers, heads, need_weights=True)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        self.align_corners = align_corners
        self.init_weights = init_weights
        self.init_prompts = init_prompts

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        #x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x) # return both cls token and image tokens

        if self.proj is not None:
            x = x @ self.proj

        return x

# @BACKBONES.register_module()
class CLIP_Surgery_ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = VV_AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)
    def init_weights(self,pretrained):
        pretrained = pretrained or self.pretrained
        if not pretrained:
            if self.attnpool is not None:
                std = self.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.attnpool.c_proj.weight, std=std)

                for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
                    for name, param in resnet_block.named_parameters():
                        if name.endswith("bn3.weight"):
                            nn.init.zeros_(param)
    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        # shape BNC
        return x
    
# @BACKBONES.register_module()
class CLIP_plain_ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)
    def init_weights(self,pretrained):
        pretrained = pretrained or self.pretrained
        if not pretrained:
            if self.attnpool is not None:
                std = self.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.attnpool.c_proj.weight, std=std)

                for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
                    for name, param in resnet_block.named_parameters():
                        if name.endswith("bn3.weight"):
                            nn.init.zeros_(param)
    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_encoder = CLIP_surgery_VisionTransformer(
        input_resolution=512,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        attn_surgery=True,
        vpt_mode=True,

    ).to(device)

    pretrained = r"/mnt/ssd/home/jcheng/EfficientNet/YOLOXX/checkpoints/ViT-B-16.pt"
    image_encoder.init_weights(pretrained)
    image = torch.randn(2, 3, 512, 512).to(device)
    global_embedding, visual_embedding, patch_tokens = image_encoder(image)
    print(global_embedding.shape)# b,512
    print(visual_embedding.shape)# b,512,32,32
    print(patch_tokens[0].shape) # b,1024,512