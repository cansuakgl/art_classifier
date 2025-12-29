import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import DATASET_PATH, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize(IMG_SIZE + 36),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class ArtworkDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_dataset(dataset_path=None):
    dataset_path = dataset_path or DATASET_PATH
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
        )
    
    image_paths = []
    labels = []
    artist_names = []
    artist_to_idx = {}

    for artist_name in sorted(os.listdir(dataset_path)):
        artist_dir = os.path.join(dataset_path, artist_name)
        if not os.path.isdir(artist_dir):
            continue

        artist_to_idx[artist_name] = len(artist_names)
        artist_names.append(artist_name)

        for img_file in os.listdir(artist_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(artist_dir, img_file))
                labels.append(artist_to_idx[artist_name])

    print(f"Loaded {len(image_paths)} images from {len(artist_names)} artists")
    return image_paths, labels, artist_names, artist_to_idx
