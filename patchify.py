import torch
import math
from process_image import tensorify

def patchify_tensor(tensor_img, patch_size=32):
    assert tensor_img.shape[2] % patch_size == 0
    assert tensor_img.shape[3] == tensor_img.shape[2]
    grid = tensor_img.shape[2] // patch_size

    B, C = tensor_img.shape[0], tensor_img.shape[1]
    tensor_img = torch.reshape(tensor_img, (B, C, grid, patch_size, grid, patch_size))
    tensor_img = tensor_img.permute(0, 2, 4, 1, 3, 5)

    tensor_img = torch.flatten(tensor_img, start_dim=1, end_dim=2)
    tensor_img = torch.flatten(tensor_img, start_dim=2, end_dim=4)
    
    return tensor_img

def patchify(img_path_list, patch_size = 32):
    tensor_images = []
    for img_path in img_path_list:
        img, _ = tensorify(img_path)
        tensor_images.append(img)
    return patchify_tensor(torch.stack(tensor_images, dim=0), patch_size)
    
def unpatchify(img_tensor_list, patch_size = 32):
    GRID = int(math.sqrt(img_tensor_list.shape[1]))

    tensors = img_tensor_list.unflatten(1, (GRID, GRID))
    tensors = tensors.unflatten(3, (3, patch_size, patch_size))  

    tensors = tensors.permute(0, 3, 1, 4, 2, 5)

    B = tensors.shape[0]
    C = tensors.shape[1]  
    HW = GRID * patch_size
    tensors = torch.reshape(tensors, (B, C, HW, HW))

    return tensors

if __name__ == "__main__":
    path = "images/shuak.png"
    img, _ = tensorify(path)
    img = img.unsqueeze(0)

    patched = unpatchify(patchify([path]))

    assert(torch.allclose(img, patched))