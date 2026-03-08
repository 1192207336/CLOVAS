import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x1 = x
        self_attention_output, _ = self.self_attention(x, x, x)
        x = x + self_attention_output
        x = self.layer_norm(x)
        x = self.fc(x)
        x = x + x1
        x = self.layer_norm(x)

        return x


class AttentionDecoderGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        max_length=20,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.max_length = max_length
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.encoder_proj = nn.Linear(input_size, hidden_size)
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads=num_heads)
        self.encoder_attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, input_size)
        self.output_proj = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, decoder_input, encoder_outputs):
        # decoder_input: (cls,dim)=(2,640)  encoder_outputs: (cls,bs,dim)=(2,8,640)
        h = torch.zeros(decoder_input.size(0), self.gru.hidden_size).to(
            decoder_input.device
        ) # (cls,hidden_dim)=(2,640)
        encoder_outputs = self.encoder_proj(encoder_outputs) # (cls,bs,hidden_dim)=(2,8,640)
        attention_weights = []
        outputs = []

        for i in range(self.max_length):
            h = self.gru(decoder_input, h)
            h = self.self_attention(h) # (cls,hidden_dim)=(2,640)

            h1 = h.unsqueeze(1).permute(1, 0, 2) # 1,cls,640
            context = encoder_outputs.permute(1, 0, 2) # bs,cls,640
            context, attention_weights_t = self.encoder_attention(h1, context, context) # context: 1,cls,640   attention_weights_t: cls,1,bs
            attention_weights.append(attention_weights_t)

            output = h + context.squeeze(0)
            output = self.fc(output) # (cls,dim) = (2,640)

            outputs.append(output)
            decoder_input = output

        outputs = torch.stack(outputs, dim=1) # (cls,n_ctx,dim) = (2,16,640)
        outputs = self.dropout(outputs)
        outputs = self.output_proj(outputs)
        attention_weights = torch.stack(attention_weights, dim=0) # (cls,n_ctx,bs) = (2,16,bs)
        return outputs, attention_weights


class AttentionDecoderLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        max_length=20,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.max_length = max_length
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.encoder_proj = nn.Linear(input_size, hidden_size)
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads=num_heads)
        self.encoder_attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, input_size)
        self.output_proj = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, decoder_input, encoder_outputs):
        h = torch.zeros(decoder_input.size(0), self.lstm.hidden_size).to(
            decoder_input.device
        )
        c = torch.zeros(decoder_input.size(0), self.lstm.hidden_size).to(
            decoder_input.device
        )

        encoder_outputs = self.encoder_proj(encoder_outputs)
        attention_weights = []
        outputs = []

        for i in range(self.max_length):
            h, c = self.lstm(decoder_input, (h, c))
            h = self.self_attention(h)

            h1 = h.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
            attention_outputs, attention_weights_t = self.encoder_attention(
                encoder_outputs, h1, h1
            )
            attention_weights.append(attention_weights_t)
            context = torch.mean(attention_outputs, dim=1)

            output = h + context
            output = self.fc(output)

            outputs.append(output)
            decoder_input = output
        outputs = torch.stack(outputs, dim=1)
        outputs = self.dropout(outputs)
        outputs = self.output_proj(outputs)
        attention_weights = torch.stack(attention_weights, dim=0)
        return outputs, attention_weights

class AttentionDecoderGRU_modified(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        max_length=20,
        num_heads=8,
        dropout=0.0,
        n_layer = 4,
    ):
        super().__init__()
        self.max_length = max_length
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.encoder_proj = nn.Linear(input_size, hidden_size)
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads=num_heads)
        self.encoder_attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, input_size)
        self.output_proj = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fusion_projection = nn.Linear(n_layer, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def average_pooling(self,patch_feature):
        bs, N, d = patch_feature.shape
        n_ctx = self.max_length
        kernel_size = N // n_ctx

        padding = 0
        if N % n_ctx != 0:
            padding = n_ctx * kernel_size - N

        patch_feature = patch_feature.permute(0, 2, 1)
        pooled_feature = F.adaptive_avg_pool1d(patch_feature, n_ctx)  # (bs, d, n_ctx)
        pooled_feature = pooled_feature.permute(0, 2, 1)

        return pooled_feature
    def forward(self, decoder_input, patch_features):
        # decoder_input: [1,d] 
        # patch_features: [b,n,d]*4
        pooled_patch_features = []
        for patch_feature in patch_features:
            patch_feature = patch_feature/patch_feature.norm(dim=-1,keepdim=True)
            pooled_patch_feature = self.average_pooling(patch_feature)
            pooled_patch_features.append(pooled_patch_feature)
        prompts = []
        for patch_feature in pooled_patch_features:
            # attention_weights = []
            outputs = []
            patch_feature = patch_feature.permute(1,0,2) # n_ctx, B, D
            patch_feature = self.encoder_proj(patch_feature) # n_ctx, B, hidden_dim
            h = torch.zeros(decoder_input.size(0), self.gru.hidden_size).to(
                    decoder_input.device
                ) # (cls,hidden_dim)=(1,640)
            for encoder_outputs in patch_feature:
                # encoder_outputs: [b,d]
                encoder_outputs = encoder_outputs.unsqueeze(0).repeat(
                    decoder_input.shape[0], 1, 1
                )  # (cls,bs,d)

                encoder_outputs = self.encoder_proj(encoder_outputs)
                

                # for i in range(self.max_length):
                h = self.gru(decoder_input, h)
                h = self.self_attention(h)

                h1 = h.unsqueeze(1).permute(1, 0, 2)
                context = encoder_outputs.permute(1, 0, 2)
                context, attention_weights_t = self.encoder_attention(h1, context, context)
                # attention_weights.append(attention_weights_t)

                output = h + context.squeeze(0)
                output = self.fc(output)

                outputs.append(output)
                decoder_input = output

            outputs = torch.stack(outputs, dim=1) # cls,n_ctx,dim
            outputs = self.dropout(outputs)
            outputs = self.output_proj(outputs) # 1,n_ctx,640
            # attention_weights = torch.stack(attention_weights, dim=0) # n_ctx,cls,1,bs
            prompts.append(outputs)
        prompts = torch.stack(prompts, dim=0)
        prompts = prompts.permute(1, 2, 3, 0)
        prompts = self.fusion_projection(prompts).squeeze(-1)
        return prompts