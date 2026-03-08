# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import torch
import pdb
from typing import Tuple, Union
from .clip_image_encoder import CLIP_surgery_VisionTransformer
from .clip_text_encoder import DynamicPromptCLIPTextEncoder
from .atm_single_seg_head import ATMSingleHeadSeg
from .cosinSim_head import ConsineSimHead
from prompt_learners.AnomalyCLIP_prompt_learner import AnomalyCLIP_PromptLearner_without_tpt
from prompt_learners.LLMPromptsGenerator import LLMPromptsGenerator
from dataset import global_defect_classes
class CLOVAS(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 dtype=torch.float32,
                 exclude_key:List=None,
                 training = False,
                 out_indices = [24],
                 vpt_settings =None,
                 tpt_settings = None,
                 image_size = 518,
                 dataset = 'visa',
                 decoder = 'atm',
                 prompt_generator = None,
                 remove_background = False,
                 cocoop_mode = False,
                 use_lepe=True,  # 新增
                 use_tcs=True,  # 新增
                 use_hfca = True,
                 loss_config = {
                    "losses":["binary_anomaly_masks","images"],
                    "binary_focal_weight" : 1.0,
                    "binary_dice_weight" : 1.0,
                    "image_weight" :1.0,
                    "loss_weight":1.0
                }
                 ):
        super(CLOVAS, self).__init__()
        self.training = training
        
        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = CLIP_surgery_VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                vpt_settings = vpt_settings,
                out_indices = out_indices
            )
        self.text_encoder = DynamicPromptCLIPTextEncoder(
                context_length=context_length,
                vocab_size=vocab_size,
                transformer_width=transformer_width,
                transformer_heads=transformer_heads,
                transformer_layers=transformer_layers,
                embed_dim=embed_dim,
                dtype=dtype,
                tpt_settings = tpt_settings
            )   
        # assert self.with_decode_head
        self.exclude_key = exclude_key
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.use_lepe = use_lepe
        self.use_tcs = use_tcs
        self.use_hfca = use_hfca
        
        self.oa_prompt_learner = AnomalyCLIP_PromptLearner_without_tpt(self.text_encoder.to("cpu"), n_ctx = 12,cocoop_mode=cocoop_mode)
        if prompt_generator == 'LLM' and self.use_lepe:
            class_names = [key for key in global_defect_classes[dataset].keys()]
            self.prompt_generator = LLMPromptsGenerator(class_names,dataset,remove_background=remove_background)
            num_classes=self.prompt_generator.num_classes
        else:
            self.prompt_generator = None
            num_classes = 1

        if decoder == 'atm':
            num_classes = 1 if (
                        not hasattr(self, 'prompt_generator') or self.prompt_generator is None or getattr(self,
                                                                                                          'use_lepe',
                                                                                                          False) is False) else self.prompt_generator.num_classes
            self.decode_head = ATMSingleHeadSeg(img_size=image_size,
                                                in_channels=embed_dim, 
                                                embed_dims = embed_dim,
                                                remove_background=remove_background,
                                                loss_config=loss_config,
                                                use_tcs = self.use_tcs,
                                                use_hfca =self.use_hfca,
                                                num_classes=num_classes)
        elif decoder == 'cosineSim':
            self.decode_head = ConsineSimHead(img_size=image_size)
        else:
            raise NotImplementedError
        if training:
            self._freeze_stages(self.text_encoder, exclude_key=exclude_key)
            self._freeze_stages(self.visual, exclude_key=exclude_key) # 
        else:
            self.text_encoder.eval()
            self.visual.eval()
            self.decode_head.eval()
            self.oa_prompt_learner.eval()

    def init_weights(self,pretrained):
        self.visual.init_weights(pretrained)
        self.text_encoder.init_weights(pretrained)
        self.logit_scale = self.visual.logit_scale
        self.decode_head.logit_scale = self.logit_scale

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
        
    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count>0:
                        print('Finetune layer in encoder:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.visual(img)
        return x
    
    def text_embedding(self, texts, img):
        text_embeddings = self.text_encoder(texts.to(img.device))
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    def forward(self, img,prompts=None,topk=0.2):
        if self.train:
            return self.forward_train(img)
        else:
            return self.forward_test(img,prompts=prompts,topk=topk)
    def encode_cocoop_text(self, prompts,tokenized_prompts):
        text_embeddings = []
        for pts_i in prompts:
            text_embedding = self.text_encoder.forward_dynamic_prompts(pts_i, tokenized_prompts)
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 0)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings
    def forward_train(self,img,gt_masks):
        # feat = self.forward(img)
        feat= {}
        
        bs = img.shape[0]
        global_embedding,visual_embedding,patch_features= self.extract_feat(img)
        if self.oa_prompt_learner.cocoop_mode:
            oa_prompts, oa_tokenized_prompts = self.oa_prompt_learner.forward(global_embedding)
            oa_text_features = self.encode_cocoop_text(oa_prompts,oa_tokenized_prompts)
        else:
            oa_prompts, oa_tokenized_prompts = self.oa_prompt_learner.forward()
            oa_text_features = self.text_encoder.forward_dynamic_prompts(oa_prompts, oa_tokenized_prompts)
            # oa_text_features = torch.stack(torch.chunk(oa_text_features, dim = 0, chunks = 2), dim = 1)
            oa_text_features = oa_text_features/oa_text_features.norm(dim=-1, keepdim=True)

        if self.prompt_generator:
            text_embeddings = self.prompt_generator(self.text_encoder,oa_prompts)
            text_embeddings = text_embeddings/text_embeddings.norm(dim=-1, keepdim=True) # c,768
            feat.update({"text_embedding":text_embeddings})

        if "text_embedding" not in feat or feat["text_embedding"] is None:
            fallback = oa_text_features.mean(1, keepdim=True)  # [B,1,D]
            feat["text_embedding"] = fallback
            print("=== Using GAPL fallback prompt (1 class) ===")

        feat.update({"patch_features":patch_features})
        feat.update({"global_embedding":global_embedding})
        feat.update({"visual_embedding":visual_embedding})
        feat.update({"oa_text_embedding":oa_text_features})
        loss =  self.decode_head.forward_train(feat,gt_masks)
        return loss
    def forward_test(self,img,prompts=None,specify_ids=None,topk=0.2,fuse_thresh=1.5):
        feat= {}
        
        bs = img.shape[0]
        global_embedding,visual_embedding,patch_features= self.extract_feat(img)
        if self.oa_prompt_learner.cocoop_mode:
            oa_prompts, oa_tokenized_prompts = self.oa_prompt_learner.forward(global_embedding)
            oa_text_features = self.encode_cocoop_text(oa_prompts,oa_tokenized_prompts)
        else:
            oa_prompts, oa_tokenized_prompts = self.oa_prompt_learner.forward()
            oa_text_features = self.text_encoder.forward_dynamic_prompts(oa_prompts, oa_tokenized_prompts)
            # oa_text_features = torch.stack(torch.chunk(oa_text_features, dim = 0, chunks = 2), dim = 1)
            oa_text_features = oa_text_features/oa_text_features.norm(dim=-1, keepdim=True)
        if self.prompt_generator:
            text_embeddings,num_good_cls = self.prompt_generator.forward_test(self.text_encoder,oa_prompts,prompts)
            text_embeddings = text_embeddings/text_embeddings.norm(dim=-1, keepdim=True) # c,768
            feat.update({"text_embedding":text_embeddings})
            feat.update({"num_good_cls":num_good_cls})
        feat.update({"patch_features":patch_features})
        feat.update({"global_embedding":global_embedding})
        feat.update({"visual_embedding":visual_embedding})
        feat.update({"oa_text_embedding":oa_text_features})
        out =  self.decode_head.forward_test(feat,topk=topk,specify_ids=specify_ids,fuse_thresh=fuse_thresh)
        return out
    def clip_feature_surgery(self,global_embedding,visual_embedding, text_features, redundant_feats=None, t=2):
        image_features = visual_embedding.flatten(2).transpose(1, 2)
        image_features = torch.cat([global_embedding.unsqueeze(1), image_features], dim=1)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = text_features.to(global_embedding.dtype)
        if redundant_feats != None:
            return text_features - redundant_feats
    
        else:
            # weights to restrain influence of obvious classes on others
            prob = image_features[:,:1,:] @ text_features.t()
            prob = (prob * 2).softmax(-1)
            w = prob / prob.mean(-1, keepdim=True)
    
            # element-wise multiplied features
            b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
            feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
            feats *= w.reshape(b, 1, n_t, 1)
            redundant_feats = feats.mean(2, keepdim=True) # along cls dim
            feats = feats - redundant_feats
            
            # sum the element-wise multiplied features as cosine similarity
            # similarity = feats.sum(-1)
            return feats
    

    