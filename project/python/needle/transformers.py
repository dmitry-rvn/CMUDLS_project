"""
Transformer module.
"""
import needle as ndl
from needle import nn
from needle import init
from needle import ops


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, *, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** 0.5
        self.w_q = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=device)
        self.w_k = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=device)
        self.w_v = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=device)
        self.w_out = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=device)

    def forward(self, x, mask=None):
        batch_size, seq_length, features_dim = x.shape

        # TODO: repeat mask n_heads times if mask is passed?
        if not mask:
            mask = init.zeros(batch_size, self.n_heads, seq_length, seq_length, device=x.device)

        x = x.reshape((batch_size * seq_length, features_dim))
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # batch_size x seq_length x d -> batch_size x n_heads x seq_length x head_dim
        q, k, v = [a.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose((1, 2)) for a in (q, k, v)]
        attn = ops.softmax(ops.bmm(k, q.transpose((3, 2))) / self.scale + mask)  # batch_size x n_heads x seq_length x seq_length

        tmp = ops.bmm(attn, v).transpose((1, 2)).reshape((batch_size, seq_length, self.n_heads * self.head_dim))
        out = ops.stack([self.w_out(b) for b in ops.split(tmp, axis=0)], axis=0)  # batch_size x seq_length x hidden_dim
        return out, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, feed_forward_dim: int, *, dropout: float = 0.0, device=None):
        super().__init__()
        self.self_attention = MultiheadAttention(hidden_dim, n_heads, device=device)
        self.layer_norm_1 = nn.LayerNorm1d(hidden_dim, device=device)
        self.layer_norm_2 = nn.LayerNorm1d(hidden_dim, device=device)
        self.feed_forward_1 = nn.Linear(in_features=hidden_dim, out_features=feed_forward_dim, device=device)
        self.feed_forward_2 = nn.Linear(in_features=feed_forward_dim, out_features=hidden_dim, device=device)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length, features_dim = x.shape

        attn_out, _ = self.self_attention(x, mask)
        x_plus_attn = x + self.dropout_1(attn_out)
        x_plus_attn = x_plus_attn.reshape((batch_size * seq_length, features_dim))

        ln1_out = self.layer_norm_1(x_plus_attn)
        ff_out = self.feed_forward_2(ops.relu(self.feed_forward_1(ln1_out)))
        out = self.layer_norm_2(ln1_out + self.dropout_2(ff_out))
        out = out.reshape((batch_size, seq_length, features_dim))
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, max_length: int, hidden_dim: int, feed_forward_dim: int,
                 *, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1, device=None):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.token_embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=hidden_dim, batch_first=True, device=device)
        self.positional_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=hidden_dim, batch_first=True, device=device)
        layers = [TransformerEncoderLayer(hidden_dim, n_heads, feed_forward_dim, dropout=dropout, device=device)
                  for _ in range(n_layers)]
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.scale = hidden_dim ** 0.5

    def forward(self, x, mask=None):
        batch_size, seq_length = x.shape

        pos = ndl.Tensor([list(range(0, seq_length))] * batch_size, device=self.device)
        pos_emb = self.positional_embedding(pos)
        tok_emb = self.token_embedding(x) * self.scale

        out = self.dropout(tok_emb + pos_emb)
        out = self.layers(out, mask=mask)
        return out
