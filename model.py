import numpy as np
import cv2
import torch
import torch.nn as nn

IMG_SIZE = 128

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=32, use_skips=True):
        super().__init__()
        f = [base, base * 2, base * 4, base * 8, base * 16]
        self.use_skips = use_skips
        self.enc = nn.ModuleList([
            ConvBlock(in_ch, f[0]),
            ConvBlock(f[0], f[1]),
            ConvBlock(f[1], f[2]),
            ConvBlock(f[2], f[3]),
        ])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBlock(f[3], f[4]), nn.Dropout2d(0.2))
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2)
            for i in (3, 2, 1, 0)
        ])
        mult = 2 if use_skips else 1
        self.dec = nn.ModuleList([
            ConvBlock(f[i] * mult, f[i]) for i in (3, 2, 1, 0)
        ])
        self.head = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            if self.use_skips:
                x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.head(x)


def prepare_input(image, size=IMG_SIZE):
    """Convert a grayscale/RGB/RGBA image to a resized RGB uint8 array."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        image = image[..., :3]
    else:
        raise ValueError("Unsupported image format. Please upload a grayscale or RGB MRI image.")

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def normalize(image):
    """Per-channel z-score normalization using non-zero pixels."""
    x = image.astype(np.float32)
    out = np.zeros_like(x)
    for channel in range(x.shape[2]):
        ch = x[..., channel]
        nonzero = ch[ch > 0]
        if nonzero.size > 10:
            out[..., channel] = np.where(
                ch > 0,
                (ch - nonzero.mean()) / (nonzero.std() + 1e-6),
                0.0,
            )
    return out.transpose(2, 0, 1)


def flair_view(image):
    channel = image[..., 1]
    return np.stack([channel, channel, channel], axis=-1)


def make_overlay(image, mask):
    """Red transparent predicted region with a yellow boundary."""
    base = flair_view(image).copy()
    red = base.copy()
    red[mask > 0] = (255, 0, 0)
    overlay = cv2.addWeighted(base, 0.65, red, 0.35, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)
    return overlay
