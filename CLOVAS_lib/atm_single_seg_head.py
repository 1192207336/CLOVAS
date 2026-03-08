from ast import Gt
import numpy as np


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast

from timm.models.layers import trunc_normal_
import cv2
from loss_seg import SegLossPlus
from loss import FocalLoss, BinaryDiceLoss
from scipy.ndimage import gaussian_filter
def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
    
class ATMSingleHeadSeg(nn.Module):
    def __init__(
            self,
            img_size,
            in_channels,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=1,
            use_proj=True,
            align_corners = False,
            remove_background = False,
            loss_config = {
                "losses":["masks","binary_anomaly_masks","images"],
                "focal_weight":100.0,
                "dice_weight":1.0,
                "binary_focal_weight" : 1.0,
                "binary_dice_weight" : 1.0,
                "image_weight" :1.0,
                "loss_weight":1.0
            },
            use_tcs = True,
            use_hfca = True,
            num_classes = 0
    ):
        super(ATMSingleHeadSeg, self).__init__()
        self.align_corners = align_corners
        self.image_size = img_size
        self.use_stages = use_stages
        self.in_channels = in_channels
        self.use_tcs = use_tcs
        self.use_hfca = use_hfca

        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []


        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            decoder = TPN_Decoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders

        self.q_proj = nn.Linear(dim * 2, dim)
        self.semantic_seg_loss = SegLossPlus(
                            num_classes=num_classes,
                            losses=loss_config["losses"],
                            remove_background = remove_background,
                            loss_weight=loss_config["loss_weight"],
                            )

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
    
    def forward(self, feats):
        patch_features = feats['patch_features']
        global_embedding =feats['global_embedding'] # b,dim
        visual_embedding = feats['visual_embedding'] # b,dim,37,37
        
        inputs = tuple([visual_embedding]) # visual feature (b,512,32,32)
        cls_token = global_embedding #[1] # cls token (4,512)
        text_embedding = feats['text_embedding'] # c',512
        oa_text_embedding = feats['oa_text_embedding'] # c,512 or b,c,512
        out = {}
        # for binary anomaly segmentation
        logit_scale = self.logit_scale.exp()
        if len(oa_text_embedding.shape)==2:
            # text_probs = logit_scale * global_embedding @ oa_text_embedding.t() # b,2
            # oa_text_embedding=oa_text_embedding.unsqueeze(0) # 1,2,512
            oa_text_embedding = torch.stack(torch.chunk(oa_text_embedding, dim = 0, chunks = 2), dim = 1) # 1,2,768

            text_probs = global_embedding.unsqueeze(1) @ oa_text_embedding.permute(0, 2, 1) # 2,1,2
        else:
            logits = []
            for t_feat,i_feat in zip(oa_text_embedding,global_embedding):
                # logits.append(logit_scale * i_feat @ t_feat.t())
                # t_feat = t_feat.unsqueeze(0)
                t_feat = torch.stack(torch.chunk(t_feat, dim = 0, chunks = 2), dim = 1) # 1,2,768
                i_feat = i_feat.unsqueeze(0).unsqueeze(1)
                logit = i_feat @ t_feat.permute(0, 2, 1)
                logits.append(logit)
            text_probs = torch.cat(logits,dim=0) # 2,1,2
        # text_probs = text_probs/0.07
        similarity_map_list = []
        # similarity_map_list.append(similarity_map)
        for idx, patch_feature in enumerate(patch_features):
            patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
            similarity, _ = self.compute_similarity(patch_feature, oa_text_embedding[0])
            similarity_map = self.get_similarity_map(similarity[:, 1:, :], self.image_size).permute(0, 3, 1, 2)
            similarity_map_list.append(similarity_map)
        out.update({"text_probs":text_probs,
                    "similarity_map_list":similarity_map_list
                    })

        # for semantic segmentation
        
        x = []
        for stage_ in inputs[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()
        bs = x[0].size()[0]

        laterals = []
        for idx, (x_, proj_, norm_) in enumerate(zip(x, self.input_proj, self.proj_norm)):
            lateral = norm_(proj_(x_))
            if idx == 0:
                laterals.append(lateral)
            else:
                if laterals[idx - 1].size()[1] == lateral.size()[1]:
                    laterals.append(lateral + laterals[idx - 1])
                else:
                    # nearest interpolate
                    l_ = self.d3_to_d4(laterals[idx - 1])
                    l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
                    l_ = self.d4_to_d3(l_)
                    laterals.append(l_ + lateral)

        lateral = laterals[-1]
        # with torch.no_grad():
        # pred_masks_small = self.forward_mask(lateral,cls_token,text_embedding)

        if self.use_hfca:

            pred_masks_small = self.forward_mask(lateral, cls_token, text_embedding)
        else:

            print("=== HFCA disabled, using pure CLIP shallow similarity (ZegCLIP baseline) ===")
            patch_feat = patch_features[-1]  # [B, 1370, D]  ← 1 + 37×37
            patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True)


            patch_tokens = patch_feat[:, 1:, :]  # [B, 1369, D]


            if text_embedding.dim() == 2:  # [C, D]
                sim = patch_tokens @ text_embedding.t()  # [B, 1369, C]
            else:  # [B, C, D]
                sim = torch.bmm(patch_tokens, text_embedding.permute(0, 2, 1))  # [B, 1369, C]

            sim = sim.softmax(dim=-1)


            anomaly_score = sim[:, :, 1:].sum(dim=-1) - sim[:, :, 0]  # [B, 1369]


            B, L = anomaly_score.shape
            side = int(L ** 0.5)  # 一定是 37
            anomaly_map = anomaly_score.view(B, side, side).unsqueeze(1)  # [B, 1, 37, 37]


            pred_masks_small = F.interpolate(anomaly_map, size=(32, 32), mode='bilinear', align_corners=False)

            C = text_embedding.shape[0] if text_embedding.dim() == 2 else text_embedding.shape[1]
            pred_masks_small = pred_masks_small.repeat(1, C, 1, 1)

        pred_masks = F.interpolate(pred_masks_small,
                                size=(self.image_size, self.image_size),
                                mode='bilinear', align_corners=self.align_corners)
        out.update({"pred_masks":pred_masks})

        if len(text_embedding.shape)==2:
            # text_probs = global_embedding.unsqueeze(1) @ oa_text_embedding.permute(0, 2, 1) # b,1,2
            # llm_text_probs = logit_scale * global_embedding @ text_embedding.t() # b,2
            text_embedding = text_embedding.unsqueeze(0) # 1,2,768
            llm_text_probs = global_embedding.unsqueeze(1) @ text_embedding.permute(0, 2, 1) # 2,1,2
        else:
            logits = []
            for t_feat,i_feat in zip(text_embedding,global_embedding):
                t_feat = t_feat.unsqueeze(0) # 1,2,768
                i_feat = i_feat.unsqueeze(0).unsqueeze(1)
                # logits.append(logit_scale * i_feat @ t_feat.t())
                logit = i_feat @ t_feat.permute(0, 2, 1)
                logits.append(logit)
            llm_text_probs = torch.stack(logits,dim=0)
        out.update({"llm_text_probs":llm_text_probs})
        return out
    
    def forward_train(self,feat,gt):
        out = self.forward(feat)
        # print(out)
        num_classes = out['pred_masks'].shape[1]

        gt_masks = gt['masks']
        if gt_masks.max() >= num_classes:
            print(f"[Ablation Defense] WARNING: gt max={gt_masks.max().item()}, "
                  f"but pred has only {num_classes} classes. Clamping...")
            gt_masks = torch.clamp(gt_masks, 0, num_classes - 1)
        gt['masks'] = gt_masks
        print(f"[Ablation Safe] gt_masks clamped to 0~{num_classes - 1}, max was {gt_masks.max().item()}")
        out['text_probs']=out['text_probs'][:, 0, ...]/0.07
        losses=self.semantic_seg_loss(out,gt)
        return losses
    
    def forward_test(self, inputs,topk=0.2,sigma = 4,specify_ids=None,fuse_thresh=1.5):
        if not self.use_tcs:
            topk = 1.0
            fuse_thresh = -1
        # specify_abnormal_idx = 6
        if len(inputs['text_embedding'].shape)==3:
                inputs['text_embedding']=inputs['text_embedding'].squeeze(0)
        if specify_ids!=None:
            text_embedding = inputs['text_embedding'][0:1,:]
            valid_abnormal_idx = []
            for id in specify_ids:
                text_embedding = torch.cat([text_embedding,inputs['text_embedding'][id:id+1,:]],dim=0)
                valid_abnormal_idx.append(id-1)
            inputs['text_embedding'] = text_embedding
            # inputs['text_embedding'] = torch.cat([inputs['text_embedding'][0:1,:],inputs['text_embedding'][specify_abnormal_idx:specify_abnormal_idx+1,:]],dim=0)
            valid_normal_idx = torch.tensor([0]).to(inputs['text_embedding'].device) 
            valid_abnormal_idx = torch.tensor(valid_abnormal_idx).to(inputs['text_embedding'].device) #torch.tensor(range(1,inputs['text_embedding'].size(0))).to(inputs['text_embedding'].device)
            scores_normal=torch.tensor([1.0]).to(inputs['text_embedding'].device) 
            scores_abnormal=torch.ones_like(valid_abnormal_idx).to(inputs['text_embedding'].device) 
        else:
            text_embedding,valid_normal_idx,valid_abnormal_idx,scores_normal,scores_abnormal =self.context_scoringv2(inputs=inputs,topk=topk)
        # text_embedding,valid_normal_idx,valid_abnormal_idx,scores_normal,scores_abnormal = self.context_scoring(inputs,gamma = 2.0,topk=topk)
            inputs['text_embedding'] = text_embedding

        out  = self.forward(inputs)
        if not self.use_hfca:
            print("=== HFCA disabled, returning ZegCLIP-style anomaly map + dummy fields ===")

            raw_anomaly = out["pred_masks"][:, 0:1]  # [B, 1, 32, 32]
            anomaly_map = F.interpolate(raw_anomaly, size=(self.image_size, self.image_size),
                                        mode='bilinear', align_corners=self.align_corners)
            anomaly_map = anomaly_map.sigmoid()  # [B, 1, H, W]

            anomaly_np = anomaly_map.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
            anomaly_np = gaussian_filter(anomaly_np, sigma=sigma)
            ft_anomaly_map = torch.from_numpy(anomaly_np).unsqueeze(0).unsqueeze(0).to(anomaly_map.device)


            binary_mask = (anomaly_map > 0.5).long()  # [B, 1, H, W]
            pred_masks = binary_mask.squeeze(0)  # → [1, H, W]


            out["ft_anomaly_map"] = ft_anomaly_map
            out["anomaly_map"] = ft_anomaly_map
            out["pred_masks"] = pred_masks

            out["valid_normal_idx"] = torch.tensor([0], device=pred_masks.device)
            out["valid_abnormal_idx"] = torch.tensor([0], device=pred_masks.device)
            out["scores_normal"] = torch.tensor([1.0])
            out["scores_abnormal"] = torch.tensor([1.0])

            return out

        out["text_probs"] = (out["text_probs"]/0.07).softmax(-1)[:,0,0]
        pred_masks = out["pred_masks"]
        similarity_map_list = out["similarity_map_list"]
        anomaly_map_list = []
        for similarity_map in similarity_map_list:
            similarity_map = similarity_map.permute(0,2,3,1)
            anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
            anomaly_map_list.append(anomaly_map)
        anomaly_map = torch.stack(anomaly_map_list)
    
        anomaly_map = anomaly_map.sum(dim = 0) # 1,h,w 
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = sigma)) for i in anomaly_map.detach().cpu()], dim = 0)
        ft_anomaly_map = anomaly_map.clone().to(pred_masks.device)
        
        pred_masks = pred_masks.sigmoid()
        pred_masks = self.fuse_anomaly_map(ft_anomaly_map,pred_masks,suppress_thresh=fuse_thresh).sigmoid() #self.fuse_anomaly_map_v2(prob_map,pred_masks,suppress_factor=suppress_factor).sigmoid()#
        pred_masks = F.softmax(pred_masks,dim=1)
        
        pred_masks = pred_masks.argmax(dim=1)
        # convert to fine-tuned binary masks
        anomaly_prob = torch.sum(pred_masks[:,1:],dim=1,keepdim=True)+1-pred_masks[:,0:1]
        ft_anomaly_map = anomaly_prob/(anomaly_prob.sum(dim=1,keepdim=True)+1e-6)
        ft_anomaly_map = ft_anomaly_map.squeeze(1)
        # ft_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = sigma)) for i in ft_anomaly_map.detach().cpu()], dim = 0)

        for i,idx in enumerate(valid_abnormal_idx):
            pred_masks[pred_masks==i+1] = idx+1
        out["pred_masks"] = pred_masks
        out.update({"anomaly_map":anomaly_map})
        out.update({"ft_anomaly_map":ft_anomaly_map})
        out.update({"valid_normal_idx":valid_normal_idx})
        out.update({"valid_abnormal_idx":valid_abnormal_idx})
        out.update({"scores_normal":scores_normal})
        out.update({"scores_abnormal":scores_abnormal})
        return out
    
    def fuse_anomaly_map(self, anomaly_map, semantic_masks, base_weight=0.2, suppress_factor=0.2, suppress_thresh=None):
        """
        anomaly_map: Tensor of shape (1, H, W), values in [0,1]
        semantic_masks: Tensor of shape (1+C, H, W)
        base_weight: Baseline fusion weight for anomaly_map
        suppress_factor: Factor to suppress low-anomaly areas
        suppress_thresh: Threshold for anomaly suppression (default: mean of anomaly_map)
        """
        # Normalize anomaly_map
        A_min, A_max = anomaly_map.min(), anomaly_map.max()
        A_norm = (anomaly_map - A_min) / (A_max - A_min + 1e-6)  # Normalize to [0,1]
        A_norm = anomaly_map 
        # Extract foreground mask (excluding background)
        M_f = semantic_masks[0,1:]  # Shape: (C, H, W)

        # Apply weighted fusion
        M_fused = M_f * (A_norm + base_weight)

        # Suppress low anomaly responses
        if suppress_thresh is None:
            suppress_thresh = A_norm.mean()
        M_fused[:, A_norm.squeeze(0) < suppress_thresh] *= suppress_factor

        # Reconstruct final semantic mask (keeping background unchanged)
        M_final = torch.cat([semantic_masks[0,0:1], M_fused], dim=0)
        
        return M_final.unsqueeze(0)
    def fuse_anomaly_map_v2(self, anomaly_map, semantic_masks, base_weight=0.2, suppress_factor=0.2):


        A_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
        

        C = semantic_masks.shape[1] - 1
        M_normal = semantic_masks[0,0:1]  #  (1, H, W)
        M_abnormal = semantic_masks[0,1:]  #  (C, H, W)


        M_normal = M_normal * (1 - A_norm)
        

        M_fused = M_abnormal * (A_norm * (1 + base_weight))


        for c in range(C):
            thresh = M_fused[c].mean()
            M_fused[c][M_fused[c] < thresh] *= suppress_factor
        

        M_final = torch.cat([M_normal, M_fused], dim=0)
        return M_final.unsqueeze(0)
    def fuse_anomaly_map_v3(self, anomaly_map, semantic_masks, alpha=0.7,gamma=2.0,suppress_factor=0.2):

        # A_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
        A_norm = anomaly_map

        C = semantic_masks.shape[1] - 1
        normal_probs = semantic_masks[0,0:1]  # (1, H, W)
        abnormal_probs = semantic_masks[0,1:]  # (C, H, W)
        
        semantic_conf,_ = abnormal_probs.max(dim=0)
        gate_mask = (semantic_conf>alpha).float()
        residual = gamma*A_norm*gate_mask.unsqueeze(0)
        abnormal_logits = abnormal_probs+residual
        fused_logits = torch.cat([normal_probs,abnormal_logits],dim=0)
        M_final = torch.softmax(fused_logits,dim=0) 
        return M_final.unsqueeze(0)
    def context_scoring(self,inputs,gamma = 2.0,topk=0.2):
        num_good_cls= inputs['num_good_cls']
        num_bad_cls = inputs['text_embedding'].size(0)-num_good_cls
        global_embedding =inputs['global_embedding']
        text_embedding = inputs['text_embedding'] # c,512
        bs = global_embedding.size(0)
      
        scores = torch.mm(global_embedding,text_embedding.T) # [bs,c]
        normal_scores = scores[:,:num_good_cls] # bs,m
        abnormal_scores = scores[:,num_good_cls:]# bs,c-m
        min_normal = normal_scores.min(dim=1,keepdim=True)[0] # bs,1
        max_normal = normal_scores.max(dim=1,keepdim=True)[0] # bs,1
        min_abnormal = abnormal_scores.min(dim=1,keepdim=True)[0] # bs,1
        max_abnormal = abnormal_scores.max(dim=1,keepdim=True)[0] # bs,1

        delta_in = torch.zeros_like(scores)
        delta_out = torch.zeros_like(scores)
        delta_in[:,:num_good_cls] = torch.max(
            torch.cat([
                (min_normal-normal_scores).unsqueeze(0),
                (normal_scores-max_normal).unsqueeze(0)
            ],dim=0),
            dim=0
        )[0] # bs,m
        delta_out[:,:num_good_cls] = torch.max(
            torch.cat([
                (min_abnormal.expand_as(normal_scores)-normal_scores).unsqueeze(0),
                (normal_scores-max_abnormal.expand_as(normal_scores)).unsqueeze(0)
            ],dim=0),
            dim=0
        )[0] # bs,m

        delta_in[:,num_good_cls:] = torch.max(
            torch.cat([
                (min_abnormal-abnormal_scores).unsqueeze(0),
                (abnormal_scores-max_abnormal).unsqueeze(0)
            ],dim=0),
            dim=0
        )[0] # bs,c-m
        delta_out[:,num_good_cls:] = torch.max(
            torch.cat([
                (min_normal.expand_as(abnormal_scores)-abnormal_scores).unsqueeze(0),
                (abnormal_scores-max_normal.expand_as(abnormal_scores)).unsqueeze(0)
            ],dim=0),
            dim=0
        )[0] # bs,c-m
        delta_diff = delta_in-delta_out # [bs,c]
        context_scores = 1/ (1+torch.exp(-gamma*delta_diff)) # [bs,c]
        sorted_indexes_normal = torch.argsort(context_scores[:,:num_good_cls],dim=1,descending=True)
        sorted_indexes_abnormal = torch.argsort(context_scores[:,num_good_cls:],dim=1,descending=True)
        sorted_context_scores_normal = context_scores[0,:num_good_cls][sorted_indexes_normal[0]]
        sorted_context_scores_abnormal = context_scores[0,num_good_cls:][sorted_indexes_abnormal[0]]
        # normal_mask = context_scores[:,:num_good_cls]>eta
        # abnormal_mask = context_scores[:,num_good_cls:]<(1-eta)
        # valid_normal_idx = torch.where(normal_mask[0])[0]
        # valid_abnormal_idx = torch.where(abnormal_mask[0])[0]+num_good_cls
        topk_good = int(round(num_good_cls*topk))
        topk_bad = int(round(num_bad_cls*topk))
        if topk_good==0:
            topk_good = 1
        if topk_bad==0:
            topk_bad = 1
        valid_normal_idx = sorted_indexes_normal[0][:topk_good]
        valid_abnormal_idx = sorted_indexes_abnormal[0][:topk_bad]
        scores_normal = sorted_context_scores_normal[:topk_good]
        scores_abnormal = sorted_context_scores_abnormal[:topk_bad]
        
        valid_normal_embds = text_embedding[valid_normal_idx]
        valid_abnormal_embds = text_embedding[valid_abnormal_idx+int(num_good_cls)]
        # valid_indexes = torch.cat([valid_normal_idx,valid_abnormal_idx])

        valid_normal_embds /= valid_normal_embds.norm(dim=-1, keepdim=True)
        valid_normal_embds = valid_normal_embds.mean(dim=0).unsqueeze(0)
        valid_normal_embds /= valid_normal_embds.norm()
        text_embedding = torch.cat([valid_normal_embds,valid_abnormal_embds],dim=0)
        # valid_embds = torch.cat([valid_normal_embds,valid_abnormal_embds],dim=0)
        return text_embedding,valid_normal_idx,valid_abnormal_idx,scores_normal,scores_abnormal
    def context_scoringv2(self,inputs,topk=0.2, k=1.0):


        # print(inputs)
        num_good_cls= inputs['num_good_cls']
        num_bad_cls = inputs['text_embedding'].size(0)-num_good_cls
        image_emb = inputs['global_embedding']
        text_embedding = inputs['text_embedding'] # c,512
        normal_prompts = text_embedding[0:num_good_cls,:]
        abnormal_prompts= text_embedding[num_good_cls:,:]
        sim_normal = F.cosine_similarity(image_emb, normal_prompts, dim=-1)    # [N+]
        sim_abnormal = F.cosine_similarity(image_emb, abnormal_prompts, dim=-1) # [N-]
        

        min_normal, max_normal = sim_normal.min(), sim_normal.max()
        min_abnormal, max_abnormal = sim_abnormal.min(), sim_abnormal.max()
        

        def interval_distance(x, min_val, max_val):
            lower_dist = torch.relu(min_val - x)
            upper_dist = torch.relu(x - max_val)
            return torch.max(lower_dist, upper_dist)
        

        d_normal_to_normal = interval_distance(sim_normal, min_normal, max_normal)
        d_normal_to_abnormal = interval_distance(sim_normal, min_abnormal, max_abnormal)
        scores_normal = 1 / (1 + torch.exp(-k * (d_normal_to_normal - d_normal_to_abnormal)))
        

        d_abnormal_to_normal = interval_distance(sim_abnormal, min_normal, max_normal)
        d_abnormal_to_abnormal = interval_distance(sim_abnormal, min_abnormal, max_abnormal)
        scores_abnormal = 1 / (1 + torch.exp(-k * (d_abnormal_to_abnormal - d_abnormal_to_normal)))
        sorted_indexes_normal = torch.argsort(scores_normal,descending=True)
        sorted_indexes_abnormal = torch.argsort(scores_abnormal,descending=True)
        sorted_context_scores_normal = scores_normal[sorted_indexes_normal]
        sorted_context_scores_abnormal = scores_abnormal[sorted_indexes_abnormal]
        # print(num_good_cls)
        topk_good = int(round(num_good_cls*topk))
        # print(num_bad_cls)
        if isinstance(num_bad_cls, torch.Tensor):
            topk_bad = int(round(num_bad_cls.item()*topk))
        else:
            topk_bad = int(round(num_bad_cls * topk))
        if topk_good==0:
            topk_good = 1
        if topk_bad==0:
            topk_bad = 1
        valid_normal_idx = sorted_indexes_normal[:topk_good]
        valid_abnormal_idx = sorted_indexes_abnormal[:topk_bad]
        scores_normal = sorted_context_scores_normal[:topk_good]
        scores_abnormal = sorted_context_scores_abnormal[:topk_bad]

        valid_normal_embds = text_embedding[valid_normal_idx]
        valid_abnormal_embds = text_embedding[valid_abnormal_idx+int(num_good_cls)]
        # valid_indexes = torch.cat([valid_normal_idx,valid_abnormal_idx])

        valid_normal_embds /= valid_normal_embds.norm(dim=-1, keepdim=True)
        valid_normal_embds = valid_normal_embds.mean(dim=0).unsqueeze(0)
        valid_normal_embds /= valid_normal_embds.norm()
        text_embedding = torch.cat([valid_normal_embds,valid_abnormal_embds],dim=0)

        # selected_normal = torch.where(scores_normal > 0)[0].tolist()
        # selected_abnormal = torch.where(scores_abnormal > 0)[0].tolist()
        
        return text_embedding,valid_normal_idx,valid_abnormal_idx,scores_normal,scores_abnormal
    def feature_surgery(self,inputs):
        image_features = torch.cat([inputs['global_embedding'].unsqueeze(1),inputs['visual_embedding'].reshape(1,768,-1).permute(0,2,1)],dim=1)
        text_features = inputs['text_embedding'] 
        prob = image_features[:,:1,:] @ text_features.t()
        prob = (prob * 2).softmax(-1)
        w = prob / prob.mean(-1, keepdim=True)

        # element-wise multiplied features
        b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        feats *= w.reshape(1, 1, n_t, 1) # bs,1370,3,768
        redundant_feats = feats.mean(2, keepdim=True) # along cls dim
        feats = feats - redundant_feats # bs,1370,3,768
        
        # sum the element-wise multiplied features as cosine similarity
        similarity = feats.sum(-1) # 1,1370,3
        ft_anomaly_map = self.get_similarity_map(similarity[:, 1:, :], self.image_size).permute(0, 3, 1, 2)
        ft_anomaly_map = (torch.sum(ft_anomaly_map[:,1:],dim=1,keepdim=True)+1-ft_anomaly_map[:,0:1])/n_t
        ft_anomaly_map = ft_anomaly_map.squeeze(1)
        return ft_anomaly_map
    def forward_mask(self,lateral,cls_token,text_feat):
        qs = []
        attns = []
        maps_size = []

        q = self.q_proj(self.get_qs(text_feat, cls_token)) # b,c',512            
        q = q.transpose(0,1)

        for idx, decoder_ in enumerate(self.decoder):
            q_, attn_ = decoder_(q, lateral.transpose(0, 1))
            for q, attn in zip(q_, attn_):
                attn = attn.transpose(-1, -2) 
                attn = self.d3_to_d4(attn)
                maps_size.append(attn.size()[-2:])
                qs.append(q.transpose(0, 1))
                attns.append(attn)
        qs = torch.stack(qs, dim=0)

        outputs_seg_masks = []
        size = maps_size[-1]

        for i_attn, attn in enumerate(attns):
            if True:
                outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=self.align_corners)) # b,c',32,32
            else:
                outputs_seg_masks.append(outputs_seg_masks[i_attn - 1] +
                                        F.interpolate(attn, size=size, mode='bilinear', align_corners=self.align_corners))


        return outputs_seg_masks[-1]

    def get_similarity_map(self,sm, shape):
        side = int(sm.shape[1] ** 0.5)
        sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
        sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
        sm = sm.permute(0, 2, 3, 1)
        return sm


    def compute_similarity(self,image_features, text_features, t=2):
        # image_features:[b,1370,768] 
        if len(text_features.shape)==2: # text_features:[2,768]
            prob_1 = image_features[:, :1, :] @ text_features.t()
            b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
            feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
            similarity = feats.sum(-1)

        else: # text_features:[b,2,768]
            prob_1 = image_features[:, :1, :] @ text_features.permute(0,2,1) # [b,1,2]
            b, n_t, n_i, c = image_features.shape[0], text_features.shape[1], image_features.shape[1], image_features.shape[2]
            feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(b, 1, n_t, c) # [b,1370,2,768]
            similarity = feats.sum(-1) # [b,1370,2]
        return (similarity/0.07).softmax(-1), prob_1


    def d3_to_d4(self, t):
        n, hw, c = t.size()
        # if hw % 2 != 0:
        #     t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def get_qs(self, q, cls):
        # q = [q.cls, q]
        
        bs, _ = cls.shape
        if len(q.shape)==2:
            C, dim = q.shape
            q = q.expand(bs, -1, -1)

        q1 = torch.einsum("bd,bcd->bcd", cls, q)
        q_ = torch.concat((q1, q), dim=-1)
        return q_


class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None: # not do
            output = self.norm(output)

        return outputs, attns

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_hpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.use_hpa = use_hpa

        # Query 永远独立（来自文本侧）
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.k = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        if self.use_hpa:
            # === HPA 核心：共享一个投影矩阵 W ===
            self.shared_kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 输出 2×dim
            print("HPA enabled: shared projection for K and V")
        else:
            # 标准 ViT
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 用于保存注意力图（可视化用）
        self.attn_map = None

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size() # 1, 21, 512
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        # q = self.q(xq).reshape(B, Nq, self.num_heads,
        #                               C // self.num_heads).permute(0, 2, 1, 3)
        # k = self.k(xk).reshape(B, Nk, self.num_heads,
        #                               C // self.num_heads).permute(0, 2, 1, 3)
        # v = self.v(xv).reshape(B, Nv, self.num_heads,
        #                               C // self.num_heads).permute(0, 2, 1, 3)

        # Query 独立投影
        q = self.q_proj(xq).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.use_hpa:
            # 共享投影：对 memory (visual patches) 做 K 和 V
            kv = self.shared_kv_proj(xk)  # [B, 1369, 1536]
            k, v = kv.chunk(2, dim=-1)  # 各 [B, 1369, 768]
        else:
            k = self.k(xk)
            v = self.v(xv)

        k = k.reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        self.attn_map = attn.detach().cpu()  # 保存用于可视化
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

if __name__ == "__main__":
    # test
    base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    novel_class = [15, 16, 17, 18, 19]
    both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    model = ATMSingleHeadSeg(
        img_size=512,
        in_channels = 512,
        channels=512,
        num_classes = len(base_class),
        base_class=base_class,
        both_class=both_class,
        embed_dims=512,
        use_stages=1,            
        num_layers=3,
        num_heads=8,
        use_proj=False,
        crop_train=False
    )
    model.init_weights()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # global_embedding = torch.randn(4, 512).to(device)
    # visual_embedding=torch.randn(4, 512, 32, 32).to(device)
    # prompt_text_embeddings = torch.randn(4, 15,512).to(device)
    # inputs = [visual_embedding,global_embedding,prompt_text_embeddings]
    pred_masks = torch.randn(4, 15, 512, 512).to(device)
    contour_map  = model.forward_contour(pred_masks)
    # visualize one contour map
    target = contour_map[0].cpu().numpy()
    # transfer 1 to 255
    target = (1-target) * 255
    plt.imsave(fname='contour_map.png',arr=target, cmap='gray')

    print(contour_map.shape)