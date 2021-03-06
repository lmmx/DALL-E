import io
import os, sys
import requests
import PIL

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from dall_e import map_pixels, unmap_pixels, load_model, models_path

target_image_size = 256


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f"min dim for image {s} < {target_image_size}")

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


# This can be changed to a GPU, e.g. 'cuda:0'.
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# For faster load times, download these files locally and use the local paths instead.
enc_path = models_path / "encoder.pkl"
enc_url = "https://cdn.openai.com/dall-e/encoder.pkl"

dec_path = models_path / "decoder.pkl"
dec_url = "https://cdn.openai.com/dall-e/decoder.pkl"

enc = load_model(str(enc_path), dev)
dec = load_model(str(dec_path), dev)

penguin_url = (
    "https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg"
)
penguin_img = download_image(penguin_url)
x = preprocess(penguin_img)
# Display the original image
print("Original image:")
penguin_img.show()

z_logits = enc(x.to(dev))
z = torch.argmax(z_logits, axis=1)
z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()

x_stats = dec(z).float()
x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
x_rec = T.ToPILImage(mode="RGB")(x_rec[0])

print("Reconstructed image:")
# Display the reconstructed image
x_rec.show()
