import torch
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt

def mask_images(images, mask_ratio = 0.75):
    IMGS = images.shape[0]
    N = images.shape[1]
    DIM = images.shape[2]
    len_keep = int(N * (1 - mask_ratio))

    random_generation = torch.rand((IMGS, N))
    ids_shuffle = torch.argsort(random_generation, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    mask_indicator = torch.ones((IMGS, N))
    mask_indicator[:, :len_keep] = 0
    mask = torch.gather(mask_indicator, dim = 1, index = ids_restore)

    ids_kept = ids_shuffle[:, :len_keep]
    ids_kept = ids_kept.unsqueeze(-1)
    ids_kept = ids_kept.expand(-1, -1, DIM) 
    patches_kept = torch.gather(images, dim=1, index = ids_kept)

    return patches_kept, mask, ids_restore

def visualize_mask(images, mask, patch_size = 32):
    DIM = images.shape[2]

    mask = mask.unsqueeze(-1)
    broadcast_mask = mask.expand(-1, -1, DIM) 

    blanked = images * (1 - broadcast_mask)

    unpatched = unpatchify(blanked, patch_size)
    unpatched = unpatched.squeeze(dim=0)

    permuted_image = unpatched.permute(1, 2, 0).detach().numpy()
    plt.imshow(permuted_image)
    plt.show()


if __name__ == "__main__":
    paths = ["images/shuak.png"]
    img = patchify(paths)
    kept, mask, restore = mask_images(img)

    visualize_mask(img, mask)

