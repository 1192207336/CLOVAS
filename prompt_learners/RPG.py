import torch
import torch.nn as nn
from AnomalyCLIP_lib.CLIP import CLIP
from pkg_resources import packaging
from copy import deepcopy
from .rnn import AttentionDecoderGRU_modified
from .utils import tokenize,_tokenizer


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


class RNN_PromptLearner(nn.Module):
    def __init__(self,
                 clip_model,
                 decoder_num_heads=8,
                 decoder_drop_out=0.2,
                #  n_vis_feature = 4,
                 design_details=None
                 ):
        super().__init__()
        self.input_dim = self.decoder_hidden = clip_model.text_projection.shape[-1]
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        # self.n_vis_feature = design_details["n_vis_feature"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        self.classnames = classnames
        self.state_normal_list = [
            "{}",
        ]

        self.state_anomaly_list = [
            "damaged {}",
        ]
        
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num
        tokenized_cls_name_pos = torch.cat([tokenize(template.format(name)+ "." for template in self.state_normal_list for name in classnames)])
        tokenized_cls_name_neg = torch.cat([tokenize(template.format(name)+ "." for template in self.state_anomaly_list for name in classnames)])
        with torch.no_grad():
            cls_feature_pos = clip_model(tokenized_cls_name_pos)
            cls_feature_neg = clip_model(tokenized_cls_name_neg)
        self.register_buffer("cls_feature_pos",cls_feature_pos)# 1,d
        self.register_buffer("cls_feature_neg",cls_feature_neg)# 1,d

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
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if True:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
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


        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized [1,1,12,768]
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized [1,1,12,768]


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


        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] ) # 1,1,1,768
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :]) # 1,1,77-12-1,768
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos) # 1,1 77
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg) # 1,1 77
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)
        self.pos_decoder = AttentionDecoderGRU_modified(
            input_size =self.input_dim,
            hidden_size = self.decoder_hidden,
            output_size = ctx_dim,
            max_length = self.n_ctx_neg,
            num_heads = decoder_num_heads,
            dropout = decoder_drop_out
        )
        self.neg_decoder = AttentionDecoderGRU_modified(
            input_size =self.input_dim,
            hidden_size = self.decoder_hidden,
            output_size = ctx_dim,
            max_length = self.n_ctx_neg,
            num_heads = decoder_num_heads,
            dropout = decoder_drop_out
        )
       

    
    def forward(self, patch_features):
        # ctx_pos = self.ctx_pos
        # ctx_neg = self.ctx_neg
        # ctx_pos = self.ctx_pos
        # ctx_neg = self.ctx_neg

        # patch features: [batch_size, num_patches, hidden_dim]
        ctx_pos = self.pos_decoder(self.cls_feature_pos, patch_features) # [n_cls, n_ctx, dim]
        ctx_pos = ctx_pos.unsqueeze(0).repeat(self.normal_num,1,1,1) # [normal_num,n_cls, n_ctx, dim]
        ctx_neg = self.neg_decoder(self.cls_feature_neg, patch_features)
        ctx_neg = ctx_neg.unsqueeze(0).repeat(self.anormaly_num,1,1,1) # [anormaly_num,n_cls, n_ctx, dim]

        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        # print(prefix_pos.shape, prefix_neg.shape)

        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (normal_num,n_cls, 1, dim)
                ctx_pos,  # (1,n_cls, n_ctx, dim)
                suffix_pos,  # (normal_num,n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls,n_neg, 1, dim)
                ctx_neg,  # (n_cls,n_neg, n_ctx, dim)
                suffix_neg,  # (n_cls, n_neg,*, dim)
            ],
            dim=2,
        )
        _, _, l, d = prompts_pos.shape # 1,1,77,768
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d) 
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0) # 2,77,768


        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)


        return prompts, tokenized_prompts, self.compound_prompts_text
        


# Main script
if __name__ == "__main__":
     # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # clip_image_encoder = clip_model.visual

    model = CLIP(
        embed_dim=640,
        image_resolution=240, vision_layers=12, vision_width=896, vision_patch_size=16,
        context_length=77, vocab_size=49408, transformer_width=640, transformer_heads=10, transformer_layers=12
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # image_features = torch.randn(8, 640).to(device)

    # text_encoder = TextEncoder(model)
    # pspg = PsPG_LP(text_encoder)
    # pspg.set_device(device)
    # tokenized_pos,tokenized_neg,prompts_pos,prompts_neg=pspg(image_features)

    AnomalyCLIP_parameters = {
        "Prompt_length": 12,"n_vis_feature": 4, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    prompt_learner = RNN_PromptLearner(model,design_details=AnomalyCLIP_parameters)
    patch_features = [torch.randn(2, 100, 640).to(device) for _ in range(4)]
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(patch_features)
    print(prompts.shape)
    