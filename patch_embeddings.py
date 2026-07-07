import torch
from torch import nn
from patchify import patchify, unpatchify
from mask import mask_images

EMBED_DIM = 128

class PatchEmbedding(nn.Module):
    def __init__(self, N, patch_size):
        super().__init__()
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, EMBED_DIM)
        initializer = torch.zeros(1, N, EMBED_DIM)
        self.pos_embed = nn.Parameter(initializer)

    def forward(self, x):
        patched = self.patch_embed(x)
        positioned = self.pos_embed + patched
        return positioned

if __name__ == "__main__":
    paths = ["images/shuak.png"]
    img = patchify(paths)

    embedding = PatchEmbedding(256, 32)
    embedded = embedding.forward(img)

    kept, mask, restore = mask_images(embedded)
    
    assert(kept.shape == (1, 64, EMBED_DIM))