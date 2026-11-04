import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        return x @ self.weight.T + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.eps) * self.weight + self.bias


class AttentionHead(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.query = Linear(dim, head_dim)
        self.key = Linear(dim, head_dim)
        self.value = Linear(dim, head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attention = scores.softmax(dim=-1)
        return attention @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(dim, dim // num_heads) for _ in range(num_heads)
        ])
        self.projection = Linear(dim, dim)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = Linear(dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = 0.5 * x * (1 + torch.erf(x / (2 ** 0.5)))
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DeiTTiny(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.patch_projection = Linear(3 * 16 * 16, 192)
        self.cls_token = nn.Parameter(torch.empty(1, 1, 192))
        self.position = nn.Parameter(torch.empty(1, 197, 192))
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim=192, num_heads=3, hidden_dim=768)
            for _ in range(12)
        ])
        self.norm = LayerNorm(192)
        self.classifier = Linear(192, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, 3, 14, 16, 14, 16)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(batch, 196, 768)
        x = self.patch_projection(x)
        cls = self.cls_token.expand(batch, 1, 192)
        x = torch.cat([cls, x], dim=1) + self.position
        x = self.blocks(x)
        x = self.norm(x)
        return self.classifier(x[:, 0])
