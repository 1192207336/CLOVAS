from audioop import tomono
import copy
import math
from os import TMP_MAX
import os.path as osp
from collections import OrderedDict
import torch
import torch.nn as nn
# from timm.models.layers import drop, drop_path, trunc_normal_
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import sys
# sys.path.append("..")
# sys.path.append(".")
# from .utils import *
# from clip import clip
from .utils import tokenize,_tokenizer
# from clip import tokenize
from torch.nn import Dropout
from copy import deepcopy
templates = [
    'a photo of a {}.',
    'a photo of a small {}.',
    'a photo of a medium {}.',
    'a photo of a large {}.',
    'This is a photo of a {}.',
    'This is a photo of a small {}.',
    'This is a photo of a medium {}.',
    'This is a photo of a large {}.',
    'a {} in the scene.',
    'a photo of a {} in the scene.',
    'There is a {} in the scene.',
    'There is the {} in the scene.',
    'This is a {} in the scene.',
    'This is the {} in the scene.',
    'This is one {} in the scene.',
    ]

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])
class MultiGranularityPromptLearner(nn.Module):
    def __init__(self,
                 clip_model,
                 design_details
                 ):
        super().__init__()
        
        input_dim = clip_model.text_projection.shape[-1] # 512
        prompt_embedding_dim = clip_model.text_projection.shape[-1]  # 512
        content_dim = design_details["content_dim"] # 3
        patch_size = design_details["patch_size"] # 32
        n_layers = design_details["n_layers"]  # 4
        classnames = ['object']
        self.classnames = classnames

        self.n_cls = len(self.classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()
        # self.pretrained = pretrained
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.state_normal_list = [
            "{}",
        ]

        self.state_anomaly_list = [
            "damaged {}",
        ]
        # self.init_context(clip_model)
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num
        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0) # n_cls,n_ctx_pos,512
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0) # n_cls,n_ctx_neg,512
        else:
            # Random Initialization
            if True:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(self.n_cls*self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_cls*self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]


        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
     
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        #生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(-1, l, d)
            embedding_neg = embedding_neg.reshape(-1, l, d)
            # embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            # embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx_neg:, :])

        # n, d = tokenized_prompts_pos.shape
        # tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        # n, d = tokenized_prompts_neg.shape
        # tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos) # n_cls*normal_num,77
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg) # n_cls*anomaly_num,77
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)

        for i in range(n_ctx_pos):
            style_projector_pos = nn.Sequential(
                                    nn.Linear(input_dim*2, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, prompt_embedding_dim)
                                )
            setattr(self, f"style_projector_pos{i}", style_projector_pos)
        for i in range(n_ctx_neg):
            style_projector_neg = nn.Sequential(
                                    nn.Linear(input_dim*2, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, prompt_embedding_dim)
                                )
            setattr(self, f"style_projector_neg{i}", style_projector_neg)

        self.bottleneck = nn.Sequential(
                nn.Conv2d(input_dim, content_dim, 1),
                nn.Flatten()
        )
        self.content_proj =nn.Sequential(
            nn.Linear(n_layers*content_dim*patch_size*patch_size, 512),
                nn.ReLU(),
                nn.Linear(512, prompt_embedding_dim))
        self.fusion_proj = nn.Sequential(
            nn.Linear(prompt_embedding_dim, 512),
                nn.ReLU(),
                nn.Linear(512, prompt_embedding_dim))  #nn.Parameter(scale * torch.rand(n_cls,n_layers))
        self.fusion_attention = nn.MultiheadAttention(prompt_embedding_dim, 8)
        self.dropout = Dropout(0.1)
        self.apply(self._init_weights)
        
    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location='cpu')['state_dict']#.float().state_dict()
            state_dict = {}
            for k in checkpoint.keys():
                if k.startswith('prompt_learner.'):
                    new_k = k.replace('prompt_learner.', '')
                    state_dict[new_k] = checkpoint[k]
            u, _ = self.load_state_dict(state_dict, False)
            print(f'pretrained prompt learner weight loaded.')
            if len(u)>0:
                print(f'{u} are misaligned params in image encoder')
        elif pretrained==None:
            self.apply(self._init_weights)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            nn.init.trunc_normal_(m.weight,std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def fusion_projector(self,content_feat,text_feat):
        c,_ = text_feat.shape # c,dim
        b,dim = content_feat.shape # b,dim
        content_feat = content_feat.unsqueeze(0).expand(c,-1,-1).permute(1,0,2).contiguous() # b,c,dim
        text_feat = text_feat.unsqueeze(1).expand(-1,b,-1).permute(1,0,2).contiguous() #b,c,dim
        attention_output, _ = self.fusion_attention(content_feat, text_feat, text_feat)

        output_features = self.fusion_proj(attention_output.mean(dim=0))  # [n_cls, dim]
        return output_features
    
    def content_projector(self,im_feature):
        # im_feature : l*c*h*w   
        # error runtime: im_feature:h,w:24,24
        x = self.bottleneck(im_feature)
        x = x.reshape(1,-1)
        x = self.content_proj(x)
        return x
    
    def prepare_style_img_feature(self,im_feature):
        L, C, n = im_feature.shape
        H = W = int(math.sqrt(n))
        assert H*W == n, "n should be a square number"
        im_feature = im_feature.reshape(L, C, H, W)
        if L > self.n_ctx:
            group_size = L // self.n_ctx
            grouped_features = []
            for start in range(0, L, group_size):
                end = min(start + group_size, L)
                grouped_features.append(im_feature[start:end].mean(dim=0))  # 分组平均
            result = torch.stack(grouped_features, dim=0).to(im_feature.device)  # [N_ctx, C,H,W]

        elif L < self.n_ctx:
            replication = im_feature[-1].unsqueeze(0).repeat(self.n_ctx - L, 1, 1, 1)
            result = torch.cat((im_feature, replication), dim=0)  # [N_ctx, C, H,W]

        else:
            result = im_feature

        return result
    def compute_style_features(self, feature_maps,target_cls="pos"):
        L, C, H,W = feature_maps.shape
       
        spatial_dims = (1, 2)
        style_features_list = []

        for l in range(L):
            current_map = feature_maps[l]
            mean = current_map.mean(dim=spatial_dims)  # [C]
            std = current_map.std(dim=spatial_dims)    # [C]

            current_style = torch.cat((mean, std), dim=0)
            style_projector = getattr(self,f"style_projector_{target_cls}{l}")
            current_style = style_projector(current_style)
            style_features_list.append(current_style)

        
        style_features = torch.stack(style_features_list, dim=0)  # [2 * C]

        return style_features
    def gen_ctx(self, image_features,text_encoder):
        # image_features: im_features:(b,h*w,d)*L
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_feature_avg = [] #(1,h*w,d)*L
                for feat in image_features:
                    new_feat = feat.clone()
                    new_feat = new_feat[0].unsqueeze(0)
                    image_feature_avg.append(new_feat)
                ctx = self.forward_single(image_feature_avg, text_encoder,ctx_only=True)

        return ctx.detach().clone()
    def forward(self, im_features,clip_model):
        return self.forward_single(im_features,clip_model)
        

    def forward_single(self, im_features,clip_model):
        """
        :param im_features:(b,h*w,d)*L [batch_size, num_patches, hidden_dim]
        :return:
        """
        im_features = torch.stack(im_features,dim=0) # L,b,1025,512
        # im_features = im_features / im_features.norm(dim=-1, keepdim=True)
        im_features = im_features.permute(1,0,3,2).contiguous() # b,L,512,1025
        im_features = im_features[:,:,:,1:]

        bs,l,c,n = im_features.shape
        w = h=int(math.sqrt(n))
        assert w*h == n, "n should be a square number"

        prefix_pos = self.token_prefix_pos  # c,1,512
        suffix_pos = self.token_suffix_pos  # c,64,512
        prefix_neg = self.token_prefix_neg  # c,1,512
        suffix_neg = self.token_suffix_neg  # c,64,512

        style_features_pos = []
        style_features_neg = []
        content_features = []
        for i,img_feat in enumerate(im_features):
            # img_feat: l,c,h*w
            style_img_feat = self.prepare_style_img_feature(img_feat) 
            style_feat_pos = self.compute_style_features(style_img_feat,"pos") # n_ctx,dim
            style_features_pos.append(style_feat_pos)

            style_feat_neg = self.compute_style_features(style_img_feat,"neg")
            style_features_neg.append(style_feat_neg)

            content_token = self.dropout(self.content_projector(img_feat.reshape(l,c,h,w)).to(im_features.device)) # 1,dim
            content_features.append(content_token)

        style_features_pos = torch.stack(style_features_pos,dim=0) # b,n_ctx,dim
        style_features_neg = torch.stack(style_features_neg,dim=0) # b,n_ctx,dim
        ctx_pos = self.ctx_pos.to(im_features.device) # 1,n_ctx,dim
        ctx_shifted_pos = ctx_pos+style_features_pos # b,n_ctx,dim
        ctx_neg = self.ctx_neg.to(im_features.device) # 1,n_ctx,dim
        ctx_shifted_neg = ctx_neg+style_features_neg # b,n_ctx,dim


        content_features = torch.stack(content_features,dim=0) # b,1,dim
        prompts_pos = []
        for ctx_shifted_i in ctx_shifted_pos:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls*self.normal_num, -1, -1)
            pts_i  = torch.cat([
                prefix_pos,
                ctx_i,
                suffix_pos
            ],dim=1)
            prompts_pos.append(pts_i)
        prompts_pos = torch.stack(prompts_pos) # b,n_cls,77,768
        prompts_neg = []
        for ctx_shifted_i in ctx_shifted_neg:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls*self.anormaly_num, -1, -1)
            pts_i  = torch.cat([
                prefix_neg,
                ctx_i,
                suffix_neg
            ],dim=1)
            prompts_neg.append(pts_i)
        prompts_neg = torch.stack(prompts_neg) # b,n_cls,77,768
        

        text_tokens_pos = []
        for pts_i, ctt_i in zip(prompts_pos, content_features):
            # pts_i :(c,77,768)
            # ctt_i :(1,768)
            text_features = clip_model.encode_text_learn(pts_i, self.tokenized_prompts_pos,self.compound_prompts_text)            
            text_features = self.fusion_projector(ctt_i, text_features)
            text_tokens_pos.append(text_features)
        text_embedding_pos = torch.stack(text_tokens_pos, dim=0).to(im_features.device)  # b,c,768

        text_tokens_neg = []
        for pts_i, ctt_i in zip(prompts_neg, content_features):
            # pts_i :(c,77,768)
            # ctt_i :(1,768)
            text_features = clip_model.encode_text_learn(pts_i, self.tokenized_prompts_neg,self.compound_prompts_text)            
            text_features = self.fusion_projector(ctt_i, text_features)
            text_tokens_neg.append(text_features)
        text_embedding_neg = torch.stack(text_tokens_neg, dim=0).to(im_features.device)
        text_embedding = torch.cat([text_embedding_pos,text_embedding_neg],dim=1) # b,2*c,768
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding
   
    