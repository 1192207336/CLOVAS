from .model import *
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
from copy import deepcopy
def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class DynamicPromptCLIPTextEncoder(nn.Module):
    def __init__(self, 
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=512,
                 dtype=torch.float32,
                 pretrained=None, 
                 tpt_settings = None,
                 ):
        super().__init__()

        self.pretrained = pretrained
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.dtype = dtype
        if tpt_settings!=None:
            ctx_dim = self.ln_final.weight.shape[0]
            self.text_encoder_n_ctx = tpt_settings['n_ctx']
            self.compound_prompts_depth = tpt_settings['prompts_depth']
            self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                          for _ in range(self.compound_prompts_depth - 1)])
            for single_para in self.compound_prompts_text:
                # print("single_para", single_para.shape)
                nn.init.normal_(single_para, std=0.02)
            # single_layer = nn.Linear(ctx_dim, 896)
            # self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                design_details = tpt_settings,
                is_text_encoder=True
            )
        else:
            self.compound_prompts_text = None
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                is_text_encoder=True
            )

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if pretrained is None:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.positional_embedding, std=0.01)
            proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
            attn_std = self.transformer.width ** -0.5
            fc_std = (2 * self.transformer.width) ** -0.5
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            if self.text_projection is not None:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            print(f'fine tune text encoder, weights are randomly initialized.')
            return
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
        elif isinstance(pretrained, dict):
            checkpoint = pretrained
        state_dict = {}

        for k in checkpoint.keys():
            if k.startswith('transformer.'):
                state_dict[k] = checkpoint[k]

            if k == 'positional_embedding' or k == 'text_projection' or k.startswith(
                    'token_embedding') or k.startswith('ln_final'):
                if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                    checkpoint[k] = checkpoint[k][:self.context_length]
                    print('positional_embedding is tuncated from 77 to', self.context_length)
                state_dict[k] = checkpoint[k]

        u, w = self.load_state_dict(state_dict, False)
        print(f'pretrained text encoder weight loaded.')
        if len(u)>0:
            print(f'{u} are misaligned params in text encoder')
            


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding .type(self.dtype)
        x = x.permute(1, 0, 2)
        deep_compound_prompts_text = self.compound_prompts_text
        # pdb.set_trace()
        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    def forward_dynamic_prompts(self, prompts, tokenized_prompts):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts + self.positional_embedding.to(cast_dtype) #
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print("test", x.shape, len(deep_compound_prompts_text))
        deep_compound_prompts_text = self.compound_prompts_text
        # pdb.set_trace()
        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder = DynamicPromptCLIPTextEncoder(tpt_mode=True).to(device)
    
    pretrained = r"/mnt/ssd/home/jcheng/EfficientNet/YOLOXX/checkpoints/ViT-B-16.pt"
    text_encoder.init_weights(pretrained)
    prompts = torch.randn(2, 77, 512).to(device)
    tokenized_prompts = torch.randint(0, 49408, (2, 77)).to(device)
    text = torch.randint(0, 49408, (2, 77)).to(device)
    output = text_encoder(text)
    print(output.shape)
    output = text_encoder.forward_dynamic_prompts(prompts, tokenized_prompts)
    print(output.shape)