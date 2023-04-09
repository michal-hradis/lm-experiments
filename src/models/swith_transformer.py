from labml_nn.transformers.switch import SwitchTransformer, SwitchTransformerLayer, SwitchFeedForward
from labml_nn.transformers.feed_forward import FeedForward
import torch
from models.model_common import generate_square_subsequent_mask, PositionalEncoding
from labml_nn.transformers.utils import subsequent_mask


def build_switch_transformer(model_dim, head_count, layers, n_experts, dropout=0.1, capacity_factor=1.2):
    d_model = model_dim
    heads = head_count
    drop_tokens = True
    is_scale_prob = True
    #expert = FeedForward(d_model, d_model * 2, dropout)
    n_layers = layers
    return SwitchTransformer(
        SwitchTransformerLayer(d_model=d_model,
                               attn=torch.nn.MultiheadAttention(d_model, heads, dropout=dropout
                                                                , bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None),
                               feed_forward=SwitchFeedForward(capacity_factor=capacity_factor,
                                                              drop_tokens=drop_tokens,
                                                              is_scale_prob=is_scale_prob,
                                                              n_experts=n_experts,
                                                              expert=FeedForward(d_model, d_model*2, dropout),
                                                              d_model=d_model),
                               dropout_prob=dropout),
        n_layers)


class TextSwitchTransformer(torch.nn.Module):
    def __init__(self, token_count, model_dim=256, head_count=4, layers=6, n_experts=4, dropout=0.025, causal=False):
        super(TextSwitchTransformer, self).__init__()

        self.token_count = token_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.layers = layers
        self.n_experts = n_experts
        self.dropout = dropout
        self.causal = causal

        self.src_mask = subsequent_mask(1024).to('cuda') if causal else None

        self.embed = torch.nn.Embedding(token_count, embedding_dim=self.model_dim)

        self.position_encoder = PositionalEncoding(d_model=self.model_dim, dropout=dropout, max_len=1024)

        self.model = build_switch_transformer(self.model_dim, self.head_count, self.layers, self.n_experts, self.dropout)

        self.decoder = torch.nn.Linear(
            in_features=self.model_dim,
            out_features=self.token_count,
            bias=True)

    def forward(self, input_chars):
        emb = self.embed(input_chars)
        emb = self.position_encoder(emb)
        emb = torch.transpose(emb, 0, 1)

        if self.src_mask is not None:
            output, f, p, n_d, p_max = self.model(emb, self.src_mask[:emb.shape[0], :emb.shape[0]])
        else:
            output, f, p, n_d, p_max = self.model(emb, mask=None)
        output = torch.transpose(output, 0, 1)
        logits = self.decoder(output)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs, f, p, n_d, p_max

    def load_balancing_loss(self, counts, route_prob):
        total = counts.sum(dim=-1, keepdims=True)
        route_frac = counts / total
        route_prob = route_prob / total
        return self.n_experts * (route_frac * route_prob).sum()