import torch
import torch.nn as nn
# import clip
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from pkg_resources import packaging
from copy import deepcopy
from typing import Union, List
from .utils import tokenize,_tokenizer
from generate_dataset_json.prompts.visa_parameters import gpt_prompts
from dataset import global_defect_classes
import numpy as np

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class LLMPromptsGenerator(nn.Module):
    def __init__(self,class_names,dataset,remove_background):
        super().__init__()
        self.class_names = class_names
        self.dataset = dataset
        self.remove_background=remove_background
        self.prompts = [global_defect_classes[self.dataset][name][1] for name in self.class_names]
        if self.remove_background:
            background_prompt = self.prompts[0]
            self.prompts.pop(0)
        self.num_classes = len(self.prompts)
    
    def forward(self,clip_model,agnostic_prompts):
        text_embeddings = []
        
        prompts =self.prompts
        dtype = clip_model.transformer.get_cast_dtype()
        tokenized_prompts = []
        for prompt in prompts:
            tokenized_prompt = tokenize(prompt).cuda()
            tokenized_prompts.append(tokenized_prompt)
        tokenized_prompts = torch.cat(tokenized_prompts).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        if len(agnostic_prompts.shape)==3:
            if not self.remove_background:
                agnostic_prompts_pos = agnostic_prompts[0].unsqueeze(0)
                agnostic_prompts_neg = agnostic_prompts[1].unsqueeze(0)
                embedding_pos = embedding[:1]
                embedding_neg = embedding[1:]
                
                embedding_pos[:,1:13,:]+=agnostic_prompts_pos[:,1:13,:]
                
                embedding_neg[:,1:13,:]+=agnostic_prompts_neg[:,1:13,:]
                prompts = torch.cat([embedding_pos,embedding_neg],dim=0)
            else:
                agnostic_prompts_neg = agnostic_prompts[1].unsqueeze(0)
                embedding[:,1:13,:]+=agnostic_prompts_neg[:,1:13,:]
                prompts = embedding
            text_embeddings = clip_model.forward_dynamic_prompts(prompts,tokenized_prompts)
        else:
            bs= agnostic_prompts.shape[0]
            agnostic_prompts_pos = agnostic_prompts[:,0:1] # bs,1,77,768
            agnostic_prompts_neg = agnostic_prompts[:,1:] # bs,1,77,768
            if not self.remove_background:
                embedding_pos = embedding[:1].unsqueeze(0).repeat(bs,1,1,1) # bs,1,77,768
                embedding_pos[:,:,1:13,:]+=agnostic_prompts_pos[:,:,1:13,:]
                embedding_neg = embedding[1:].unsqueeze(0).repeat(bs,1,1,1) # bs,c,77,768
                embedding_neg[:,:,1:13,:]+=agnostic_prompts_neg[:,:,1:13,:]
                prompts = torch.cat([embedding_pos,embedding_neg],dim=1)
            else:
                embedding = embedding.unsqueeze(0).repeat(bs,1,1,1) # bs,c,77,768
                embedding[:,:,1:13,:]+=agnostic_prompts_neg[:,:,1:13,:]
                prompts = embedding
            text_embeddings = []
            for pts_i in prompts:
                text_embedding = clip_model.forward_dynamic_prompts(pts_i, tokenized_prompts)
                text_embeddings.append(text_embedding)
            text_embeddings = torch.stack(text_embeddings, dim = 0)
        return text_embeddings
    def forward_test(self,clip_model,agnostic_prompts,prompts=None):
        text_embeddings = []
        if prompts is None:
            prompts = [global_defect_classes[self.dataset][name][1] for name in self.class_names]
            num_good_cls = 1
        else:
            normal_prompts = [prompt for prompt in prompts["normal_prompts"]]
            abnormal_prompts = [prompt for prompt in prompts["abnormal_prompts"].values()]
            num_good_cls = len(normal_prompts)
            prompts = normal_prompts + abnormal_prompts
        # prompts = [global_defect_classes[self.dataset][name][1] for name in self.class_names]
        dtype = clip_model.transformer.get_cast_dtype()
        tokenized_prompts = []
        for prompt in prompts:
            tokenized_prompt = tokenize(prompt).cuda()
            tokenized_prompts.append(tokenized_prompt)
        tokenized_prompts = torch.cat(tokenized_prompts).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        if len(agnostic_prompts.shape)==3:
            embedding_pos = embedding[:num_good_cls]
            embedding_neg = embedding[num_good_cls:]
            agnostic_prompts_pos = agnostic_prompts[0].unsqueeze(0)
            embedding_pos[:,1:13,:]+=agnostic_prompts_pos[:,1:13,:]
            agnostic_prompts_neg = agnostic_prompts[1].unsqueeze(0)
            embedding_neg[:,1:13,:]+=agnostic_prompts_neg[:,1:13,:]
            prompts = torch.cat([embedding_pos,embedding_neg],dim=0)
            text_embeddings = clip_model.forward_dynamic_prompts(prompts,tokenized_prompts)

        else:
            # prompts = embedding
            bs= agnostic_prompts.shape[0]
            agnostic_prompts_pos = agnostic_prompts[:,0:1] # bs,1,77,768
            agnostic_prompts_neg = agnostic_prompts[:,1:] # bs,1,77,768
            embedding_pos = embedding[:num_good_cls].unsqueeze(0).repeat(bs,1,1,1) # bs,1,77,768
            embedding_pos[:,:,1:13,:]+=agnostic_prompts_pos[:,:,1:13,:]
            embedding_neg = embedding[num_good_cls:].unsqueeze(0).repeat(bs,1,1,1) # bs,c,77,768
            embedding_neg[:,:,1:13,:]+=agnostic_prompts_neg[:,:,1:13,:]
            prompts = torch.cat([embedding_pos,embedding_neg],dim=1)
            text_embeddings = []
            for pts_i in prompts:
                text_embedding = clip_model.forward_dynamic_prompts(pts_i, tokenized_prompts)
                text_embeddings.append(text_embedding)
            text_embeddings = torch.stack(text_embeddings, dim = 0)
        return text_embeddings,num_good_cls
    def forward_single_object(self,clip_model,gt_object_type,compound_prompts_text):   
        class_names = [name.lower() for _,name in self.id2class_map[gt_object_type].items() if name.lower() != 'background']
        prompts = []#['a photo of a normal ' + self.class2name_map[gt_object_type]]
        for cls_name in class_names:
            prompt = gpt_prompts[gt_object_type][cls_name.lower()]
            prompts.append(prompt)
        if len(prompts) < self.c_max:
            for _ in range(self.c_max-len(prompts)):
                prompts.append('PLACEHOLDER')
        tokenized_prompts = []
        for prompt in prompts:
            tokenized_prompt = tokenize(prompt).cuda()
            tokenized_prompts.append(tokenized_prompt)
        tokenized_prompts = torch.cat(tokenized_prompts).cuda()
        text_embedding = clip_model.encode_text(tokenized_prompts,compound_prompts_text)
        return text_embedding

        


# Main script
if __name__ == "__main__":
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # clip_image_encoder = clip_model.visual

    # Dummy inputs
    batch_size, num_patches, hidden_dim, num_layers = 4, 16, 512, 4
    patch_tokens = [torch.randn(batch_size, num_patches, hidden_dim).to(device) for _ in range(num_layers)]

    # Initialize Prompt Learner
    num_tokens = 10
    prompt_learner = PromptLearner(num_tokens, hidden_dim, num_layers).to(device)

    # Generate prompts
    prompts = prompt_learner(patch_tokens)
    print("Generated prompts shape:", prompts.shape)  # [batch_size, num_tokens * 3, hidden_dim]

    # Combine with CLIP text encoder
    # text_features = clip_model.encode_text(prompts.view(-1, prompts.size(-1)))
    # print("Text features shape:", text_features.shape)  # [batch_size * num_tokens * 3, feature_dim]
    