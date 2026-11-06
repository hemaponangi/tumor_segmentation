import os
import glob
import json
import random
import time

import cv2
import kagglehub
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import UNet, normalize, IMG_SIZE

# Adjust these values if your computer has limited memory.
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 1e-3
BASE_FILTERS = 32
SEED = 42
OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pair(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"Could not read image/mask: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return image, (mask > 0).astype(np.uint8)


def augment(image, mask):
    if random.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.9, 1.1)
    matrix = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, scale)
    image = cv2.warpAffine(
        np.ascontiguousarray(image), matrix, (IMG_SIZE, IMG_SIZE),
        flags=cv2.INTER_LINEAR, borderValue=0
    )
    mask = cv2.warpAffine(
        np.ascontiguousarray(mask), matrix, (IMG_SIZE, IMG_SIZE),
        flags=cv2.INTER_NEAREST, borderValue=0
    )
    factor = random.uniform(0.9, 1.1)
    image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return image, mask


class MRIDataset(Dataset):
    def __init__(self, images, masks, train=False):
        self.images = images
        self.masks = masks
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        if self.train:
            image, mask = augment(image, mask)
        x = torch.from_numpy(normalize(image).copy()).float()
        y = torch.from_numpy(mask[None].astype(np.float32))
        return x, y


def dice_loss(logits, target, eps=1.0):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((1, 2, 3))
    denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()


def criterion(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def validation_dice(model, loader, device):
    model.eval()
    tp = fp = fn = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prediction = torch.sigmoid(model(x)) > 0.5
        truth = y > 0.5
        tp += (prediction & truth).sum().item()
        fp += (prediction & ~truth).sum().item()
        fn += (~prediction & truth).sum().item()
    return (2 * tp) / (2 * tp + fp + fn + 1e-9)


def main():
    seed_everything(SEED)
    cv2.setNumThreads(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    print("Downloading public LGG MRI Segmentation dataset...")
    dataset_path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    mask_suffix = "_mask.tif"
    records = {}

    for mask_path in sorted(glob.glob(
        os.path.join(dataset_path, "**", "*" + mask_suffix), recursive=True
    )):
        image_path = mask_path[:-len(mask_suffix)] + ".tif"
        if not os.path.exists(image_path):
            continue
        key = os.path.basename(image_path)
        if key not in records:
            patient_id = os.path.basename(os.path.dirname(mask_path))
            records[key] = (image_path, mask_path, patient_id)

    if not records:
        raise RuntimeError("No MRI/mask pairs found in the downloaded dataset.")

    images, masks, patient_ids = [], [], []
    for count, (image_path, mask_path, patient_id) in enumerate(records.values(), 1):
        image, mask = load_pair(image_path, mask_path)
        images.append(image)
        masks.append(mask)
        patient_ids.append(patient_id)
        if count % 500 == 0:
            print(f"Loaded {count} MRI/mask pairs")

    images = np.stack(images)
    masks = np.stack(masks)
    patient_ids = np.asarray(patient_ids)

    # Split by patient to prevent slices from the same patient leaking across sets.
    patients = np.array(sorted(set(patient_ids)))
    rng = np.random.RandomState(SEED)
    rng.shuffle(patients)
    train_end = int(0.70 * len(patients))
    val_end = train_end + int(0.15 * len(patients))
    train_patients = patients[:train_end]
    val_patients = patients[train_end:val_end]
    test_patients = patients[val_end:]

    train_idx = np.where(np.isin(patient_ids, train_patients))[0]
    val_idx = np.where(np.isin(patient_ids, val_patients))[0]
    test_idx = np.where(np.isin(patient_ids, test_patients))[0]
    print(f"Patients: train={len(train_patients)}, validation={len(val_patients)}, test={len(test_patients)}")
    print(f"Slices: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")

    train_loader = DataLoader(
        MRIDataset(images[train_idx], masks[train_idx], train=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        MRIDataset(images[val_idx], masks[val_idx]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = UNet(base=BASE_FILTERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_dice = -1.0
    bad_epochs = 0
    best_path = os.path.join(OUTPUT_DIR, "unet_best.pt")
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(x)
            count += len(x)

        val_dice = validation_dice(model, val_loader, device)
        scheduler.step(val_dice)
        train_loss = running_loss / max(count, 1)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | validation Dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    elapsed = (time.time() - start) / 60
    metadata = {
        "image_size": IMG_SIZE,
        "input_channels": 3,
        "base_filters": BASE_FILTERS,
        "threshold": 0.5,
        "best_validation_dice": round(float(best_dice), 4),
        "training_minutes": round(elapsed, 1),
        "dataset": "LGG MRI Segmentation (mateuszbuda/lgg-mri-segmentation)",
        "train_slices": int(len(train_idx)),
        "validation_slices": int(len(val_idx)),
        "test_slices": int(len(test_idx)),
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining finished.")
    print("Best model saved to:", best_path)
    print("Metadata saved to artifacts/metadata.json")
    print("Copy the artifacts folder into your Streamlit project before deployment.")


if __name__ == "__main__":
    main()
