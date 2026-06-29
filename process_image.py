import torch
import torchvision.transforms.functional as func
import torchvision
import matplotlib.pyplot as plt

def tensorify(image_path, resize = [512, 512], device = 'cpu'):
    raw_image = torchvision.io.decode_image(image_path, "RGB")
    float_image = (raw_image.to(torch.float32)) / 255.0
    if resize is not None:
        float_image = func.resize(float_image, resize)
    return float_image, float_image.shape

if __name__ == "__main__":
    image, shape = tensorify("images/shuak.png")

    print(image)
    print(shape)
    print(image.min())
    print(image.max())

    permuted_image = image.permute(1, 2, 0)
    plt.imshow(permuted_image)
    plt.show()