import torch
from torch import nn
from patchify import patchify
from mask import mask_images
from patch_embeddings import EMBED_DIM, PatchEmbedding
from encoder import PatchEncoder

DECODE_DIM = EMBED_DIM // 2

class PatchDecoder(nn.Module):
    def __init__(self, N, patch_size, len_keep, num_heads = 4, num_layers=2, mlp_ratio=4):
        super().__init__()
        self.N = N
        self.len_keep = len_keep

        # 3 from RGB
        patch_dim = 3 * patch_size * patch_size
        ffw = DECODE_DIM * mlp_ratio

        self.project_in = nn.Linear(EMBED_DIM, DECODE_DIM)

        initializer = torch.zeros(1, 1, DECODE_DIM)
        self.mask_token = nn.Parameter(initializer)

        initializer = torch.zeros(1, N, DECODE_DIM)
        self.decoder_pos_embed = nn.Parameter(initializer)

        # use this to be perfectly accurate to MAE, because torch decoder has cross-attention but encoder doesnt
        decoder_layer = nn.TransformerEncoderLayer(d_model = DECODE_DIM, dim_feedforward=ffw, nhead = num_heads, batch_first=True, norm_first=True, activation='gelu')
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.project_out = nn.Linear(DECODE_DIM, patch_dim)

    def forward(self, x, ids_restore):
        projected = self.project_in(x)

        batch_size = x.shape[0]

        # concatenate masked and visible for unshuffling
        mask_tokens = torch.broadcast_to(self.mask_token, (batch_size, self.N-self.len_keep, DECODE_DIM))
        concatenated = torch.cat((projected, mask_tokens), dim = 1)

        # unshuffle
        _3d_ids = ids_restore.unsqueeze(-1)
        restore = _3d_ids.expand((-1, -1, DECODE_DIM))
        unshuffled = torch.gather(concatenated, dim = 1, index = restore)

        # decode
        positioned = self.decoder_pos_embed + unshuffled

        # transform
        transformed = self.decoder(positioned)

        # project out
        out = self.project_out(transformed)
        return out
        

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
    
    assert(pred.shape == (1, 256, 3072))