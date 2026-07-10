import torch
from torch import nn
from patchify import patchify
from mask import mask_images
from patch_embeddings import EMBED_DIM, PatchEmbedding
from encoder import PatchEncoder
from decoder import PatchDecoder, DECODE_DIM

def loss(target, pred, mask):
    # pixel-wise squared error
    diff = pred - target
    squared_error = diff * diff

    patch_loss = squared_error.mean(dim=-1)

    # remove counts over visible patches (i.e. what was given)
    weighted = patch_loss * mask

    error = torch.sum(weighted)
    num_masked = torch.sum(mask)    
    loss_val = error/num_masked
    
    return loss_val

if __name__ == "__main__":
    paths = ["images/shuak.png"]
    img = patchify(paths)

    embedding = PatchEmbedding(256, 32)
    embedded = embedding.forward(img)

    kept, mask, restore = mask_images(embedded)

    encoder = PatchEncoder()
    encoded = encoder.forward(kept)

    decoder = PatchDecoder(256, 32, kept.shape[1])
    pred = decoder.forward(encoded, restore)

    # loss tests
    loss_val = loss(img, img, mask)
    print(f"Equal Loss: {loss_val}")

    # your existing forward (embed → mask → encoder → decoder)
    loss_val = loss(img, pred, mask)

    assert(loss_val.dim() == 0)
    assert(loss_val > 0)

    loss_val.backward()
    print(f"Test Loss: {loss_val}")