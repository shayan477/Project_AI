import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ─── ViT BUILDING BLOCKS ───────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(in_features, hidden_features)
        self.act     = nn.GELU()
        self.fc2     = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.dropout(x)
        x = self.fc2(x); x = self.dropout(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads      = heads
        self.scale      = dim ** -0.5
        self.qkv        = nn.Linear(dim, dim * 3)
        self.attn_drop  = nn.Dropout(dropout)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(dropout)
    def forward(self, x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C//self.heads)
        q,k,v = qkv.permute(2,0,3,1,4)
        attn  = (q @ k.transpose(-2,-1)) * self.scale
        attn  = attn.softmax(dim=-1)
        attn  = self.attn_drop(attn)
        out   = (attn @ v).transpose(1,2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = AttentionBlock(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_dim, dim, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16,
                 num_classes=2, dim=512, depth=6,
                 heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches      = (image_size // patch_size) ** 2
        patch_dim        = 3 * patch_size * patch_size
        self.patch_size  = patch_size

        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token       = nn.Parameter(torch.randn(1,1,dim))
        self.pos_embedding   = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.dropout         = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B,C,H,W = x.shape; p = self.patch_size
        x = x.unfold(2,p,p).unfold(3,p,p)
        x = x.contiguous().view(B,C,-1,p,p)
        x = x.permute(0,2,1,3,4).reshape(B, -1, C*p*p)
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x[:,0])

# ─── PREDICTOR ────────────────────────────────────────────────

class HumanActionPredictor:
    def __init__(self, model_path, device=None):
        self.device = (torch.device(device)
                       if device in ("cpu","cuda")
                       else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict      = ckpt["state_dict"]
        self.idx_to_class = ckpt["idx_to_class"]

        self.model = ViT(num_classes=len(self.idx_to_class))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    def predict(self, image_input):
        img = (Image.open(image_input).convert("RGB")
               if isinstance(image_input, str)
               else image_input.convert("RGB"))
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = F.softmax(logits, dim=-1)[0]
            idx    = torch.argmax(probs).item()
        return self.idx_to_class[idx], probs[idx].item()
