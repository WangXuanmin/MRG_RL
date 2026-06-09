import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_queries=32,
        num_layers=2,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, output_dim) * 0.02)
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "query_norm": nn.LayerNorm(output_dim),
                        "context_norm": nn.LayerNorm(output_dim),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=output_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "ffn_norm": nn.LayerNorm(output_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(output_dim, output_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(output_dim * 4, output_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, image_tokens, attention_mask=None):
        batch_size = image_tokens.shape[0]
        context = self.input_proj(image_tokens)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        for layer in self.layers:
            residual = queries
            attn_out, _ = layer["cross_attn"](
                query=layer["query_norm"](queries),
                key=layer["context_norm"](context),
                value=layer["context_norm"](context),
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            queries = residual + attn_out
            queries = queries + layer["ffn"](layer["ffn_norm"](queries))

        return self.final_norm(queries)
