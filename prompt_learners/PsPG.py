from copy import deepcopy
import torch
import torch.nn as nn
import pdb

# from .rnn import AttentionDecoderGRU, AttentionDecoderLSTM

from timm.models.layers import drop_path, trunc_normal_
from AnomalyCLIP_lib.CLIP import CLIP 
from .rnn import AttentionDecoderGRU, AttentionDecoderLSTM
from prompt_learners.utils import tokenize
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model_name = "clip"
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.output_dim = self.text_projection.shape[-1]

    def get_embedding(self, text):
        return self.token_embedding(text).type(
            self.dtype
        )  # [batch_size, n_ctx, transformer.width]

    def forward(self, text, embed=None):
        text = text
        if embed is not None:
            x = embed + self.positional_embedding.type(self.dtype)
        else:
            x = self.token_embedding(text).type(
                self.dtype
            ) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x1 = x @ self.text_projection
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x, x1

# modified from COOP & DualCOOP
class ContextOptimization(nn.Module):

    def __init__(
        self,
        
        text_encoder,
        classnames=['object','damaged object'],
        decoder_type = "gru",
        n_ctx_pos = 5,
        n_ctx_neg = 5,
        ctx_init_pos="",
        ctx_init_neg="",
        token_type = "BERT",
        COOP_CSC = False
    ):
        super().__init__()
        n_cls = len(classnames)
        ctx_dim = text_encoder.ln_final.weight.shape[0]
        self.token_type = token_type
        prefix_len = self.get_prefix_len()

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = text_encoder.get_embedding(prompt_pos)
                embedding_neg = text_encoder.get_embedding(prompt_neg)
            ctx_vectors_pos = embedding_pos[0, prefix_len : prefix_len + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, prefix_len : prefix_len + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if COOP_CSC:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if COOP_CSC:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx_neg, ctx_dim)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]

        tokenized_prompts_pos = tokenize(prompts_pos)
        tokenized_prompts_neg = tokenize(prompts_neg)
        with torch.no_grad():
            embedding_pos = text_encoder.get_embedding(tokenized_prompts_pos)
            embedding_neg = text_encoder.get_embedding(tokenized_prompts_neg)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        if prefix_len == 1:
            self.register_buffer("token_prefix_pos", embedding_pos[:, :prefix_len, :])
            self.register_buffer("token_prefix_neg", embedding_neg[:, :prefix_len, :])
        self.register_buffer(
            "token_suffix_pos", embedding_pos[:, prefix_len + n_ctx_pos :, :]
        )
        self.register_buffer(
            "token_suffix_neg", embedding_neg[:, prefix_len + n_ctx_neg :, :]
        )

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.tokenized_prompts_pos = tokenized_prompts_pos
        self.tokenized_prompts_neg = tokenized_prompts_neg

    def get_prefix_len(self):
        return 0 if "BERT" in self.token_type else 1

    @torch.jit.ignore
    def set_device(self, device):
        self.tokenized_prompts_pos = {
            "input_ids": self.tokenized_prompts_pos["input_ids"].to(device),
            "attention_mask": self.tokenized_prompts_pos["attention_mask"].to(device),
        }
        self.tokenized_prompts_neg = {
            "input_ids": self.tokenized_prompts_neg["input_ids"].to(device),
            "attention_mask": self.tokenized_prompts_neg["attention_mask"].to(device),
        }

    def forward(self):
        prefix_len = self.get_prefix_len()
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        ctx_pos = ctx_pos.expand(self.n_cls, -1, -1)
        ctx_neg = ctx_neg.expand(self.n_cls, -1, -1)

        prefix_pos = self.token_prefix_pos if prefix_len == 1 else None
        prefix_neg = self.token_prefix_neg if prefix_len == 1 else None
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        if prefix_len == 1:
            prompts_pos = torch.cat(
                [
                    prefix_pos,  # (n_cls, 1, dim)
                    ctx_pos,  # (n_cls, n_ctx, dim)
                    suffix_pos,  # (n_cls, *, dim)
                ],
                dim=1,
            )

            prompts_neg = torch.cat(
                [
                    prefix_neg,  # (n_cls, 1, dim)
                    ctx_neg,  # (n_cls, n_ctx, dim)
                    suffix_neg,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            prompts_pos = torch.cat(
                [
                    ctx_pos,  # (n_cls, n_ctx, dim)
                    suffix_pos,  # (n_cls, *, dim)
                ],
                dim=1,
            )

            prompts_neg = torch.cat(
                [
                    ctx_neg,  # (n_cls, n_ctx, dim)
                    suffix_neg,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        tokenized_prompts_pos = self.tokenized_prompts_pos
        tokenized_prompts_neg = self.tokenized_prompts_neg

        return (
            tokenized_prompts_pos,
            tokenized_prompts_neg,
            prompts_pos,
            prompts_neg,
            None,
            None,
        )


class PsPG_LP(nn.Module):

    def __init__(
        self,
        text_encoder,
        classnames=['object','damaged object'],
        context_length=77,
        decoder_max_length=16,
        decoder_hidden=640,
        decoder_num_heads=8,
        decoder_drop_out=0.2,
        decoder_layers=2,
        decoder_drop_path=0.1,
        enable_prefix=False
    ):
        super().__init__()
        self.input_dim = text_encoder.output_dim
        self.output_dim = text_encoder.ln_final.weight.shape[0]
        self.max_length = decoder_max_length
        self.hidden_size = decoder_hidden
        self.num_head = decoder_num_heads
        self.dropout = decoder_drop_out
        self.max_context = context_length
        self.layers = decoder_layers
        self.drop_path = decoder_drop_path
        self.enable_prefix = enable_prefix
        self.device = None
        self.tokenized_prompt = None

        tokenized_classnames = tokenize(classnames)
        print(classnames)
        if enable_prefix:
            self.register_buffer(
                "cls_length",
                tokenized_classnames.argmax(dim=-1) - 1,
                persistent=False,
            )
        with torch.no_grad():
            x, x1 = text_encoder(tokenized_classnames)
        self.register_buffer("cls_feature", x, persistent=False)
        self.register_buffer(
            "cls_allfeatures", x1[:, 1 : self.max_length + 1], persistent=False
        )
        temp_text = " ".join(["X"] * self.max_length)
        token = tokenize(temp_text)
        token_text = token
        with torch.no_grad():
            embedding = text_encoder.get_embedding(token) # [1,77,640]
        self.register_buffer(
            "prefix_embed",
            embedding[0, 0].repeat(self.cls_feature.shape[0], 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "suffix_embed",
            embedding[0, token_text.argmax(dim=-1) :].repeat(
                self.cls_feature.shape[0], 1, 1
            ),
            persistent=False,
        )
        if enable_prefix:
            prompts = [temp_text + " " + name + "." for name in classnames]
            tokenized_prompts = tokenize(prompts)
            self.tokenized_prompt = tokenized_prompts
            with torch.no_grad():
                embedding_template = text_encoder.get_embedding(tokenized_prompts)
            self.register_buffer(
                "prefix_template",
                embedding_template[:, :1, :],
                persistent=False,
            ) # [2,1,640]
            self.register_buffer(
                "suffix_template",
                embedding_template[:, 1 + self.max_length :, :],
                persistent=False,
            ) # [2,60,640]

        decoder_type = "gru"
        if decoder_type == "gru":
            self.pos_decoder = AttentionDecoderGRU(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
                self.num_head,
                self.dropout,
            )
            self.neg_decoder = AttentionDecoderGRU(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
                self.num_head,
                self.dropout,
            )
        elif decoder_type == "lstm":
            self.pos_decoder = AttentionDecoderLSTM(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
                self.num_head,
                self.dropout,
            )
            self.neg_decoder = AttentionDecoderLSTM(
                self.input_dim,
                self.hidden_size,
                self.output_dim,
                self.max_length,
                self.num_head,
                self.dropout,
            )


    @torch.jit.ignore
    def set_device(self, device):
        self.device = device
        self.to(device)
        if self.tokenized_prompt is not None:
            self.tokenized_prompt.to(device)

    @property
    def dtype(self):
        return self.cls_feature.dtype


    def forward(self, image_features):
        # image_features: (bs,dim)   cls_feature: (cls,dim)
        image_features = image_features.unsqueeze(0).repeat(
            self.cls_feature.shape[0], 1, 1
        )  # (cls,bs,dim)
        output_pos, _ = self.pos_decoder(self.cls_feature, image_features) # (cls,n_ctx,dim)
        output_neg, _ = self.neg_decoder(self.cls_feature, image_features) # (cls,n_ctx,dim)

        pos_texts = []
        neg_texts = []
        pos_indexs = torch.zeros(self.cls_feature.shape[0]).long()
        neg_indexs = torch.zeros(self.cls_feature.shape[0]).long()

        for i in range(self.cls_feature.shape[0]):
            pos_texts.append(" ".join(["X"] * self.max_length))
            neg_texts.append(" ".join(["X"] * self.max_length))
        tokenized_pos = tokenize(pos_texts)
        tokenized_neg = tokenize(neg_texts)
        tokenized_pos =tokenized_pos.to(self.device)
        tokenized_neg = tokenized_neg.to(self.device)
        if not self.enable_prefix:
            prompts_pos = torch.cat(
                [
                    self.prefix_embed,  # (cls, 1, dim)
                    output_pos,  # (cls, length, dim)
                    self.suffix_embed,  # (cls, *, dim)
                ],
                dim=1,
            )

            prompts_neg = torch.cat(
                [
                    self.prefix_embed,  # (cls, 1, dim)
                    output_neg,  # (cls, length, dim)
                    self.suffix_embed,  # (cls, *, dim)
                ],
                dim=1,
            )
        else:
            prompts_pos = torch.cat(
                [
                    self.prefix_template,  # (cls, 1, dim)
                    output_pos,  # (cls, length, dim)
                    self.suffix_template,  # (cls, *, dim)
                ],
                dim=1,
            )

            prompts_neg = torch.cat(
                [
                    self.prefix_template,  # (cls, 1, dim)
                    output_neg,  # (cls, length, dim)
                    self.suffix_template,  # (cls, *, dim)
                ],
                dim=1,
            )
            tokenized_pos = self.tokenized_prompt
            tokenized_neg = self.tokenized_prompt

        return (
            tokenized_pos,
            tokenized_neg,
            prompts_pos,
            prompts_neg,
        )

if __name__ == "__main__":

    # ViT-B-16-plus-240
    model = CLIP(
            embed_dim=640,
            image_resolution=240, vision_layers=12, vision_width=896, vision_patch_size=16,
            context_length=77, vocab_size=49408, transformer_width=640, transformer_heads=10, transformer_layers=12
        )
    text_encoder = TextEncoder(model)
    pspg = PsPG_LP(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pspg.set_device(device)
    image_features = torch.randn(8, 768).to(device)
    pspg(image_features)
    print("Pass")