import torch
from torch import nn
from patchify import patchify
from mask import mask_images
from patch_embeddings import EMBED_DIM, PatchEmbedding

class PatchEncoder(nn.Module):
    def __init__(self, num_heads = 4, num_layers=4, mlp_ratio=4):
        super().__init__()
        ffw = EMBED_DIM * mlp_ratio
        encoder_layer = nn.TransformerEncoderLayer(d_model = EMBED_DIM, dim_feedforward=ffw, nhead = num_heads, batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)
        

if __name__ == "__main__":
    paths = ["images/shuak.png"]
    img = patchify(paths)

    embedding = PatchEmbedding(256, 32)
    embedded = embedding.forward(img)

    kept, mask, restore = mask_images(embedded)

    encoder = PatchEncoder()
    encoded = encoder.forward(kept)
    
    assert(encoded.shape == (1, 64, EMBED_DIM))