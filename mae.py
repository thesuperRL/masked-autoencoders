import torch
from torch import nn
from decoder import PatchDecoder
from encoder import PatchEncoder
from mask import mask_images
from patch_embeddings import PatchEmbedding

class MAE(nn.Module):
    def __init__(self, N, patch_size, len_keep):
        super().__init__()

        self.patchembedding = PatchEmbedding(N, patch_size)
        self.patchencoder = PatchEncoder()
        self.patchdecoder = PatchDecoder(N, patch_size, len_keep)

    def forward(self, patches):
        embedded = self.patchembedding(patches)
        kept, mask, restore = mask_images(embedded)
        encoded = self.patchencoder(kept)
        pred = self.patchdecoder(encoded, restore)
        return pred, mask