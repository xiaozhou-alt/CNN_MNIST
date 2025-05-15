import gzip
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # 随机应用数据增强
            if random.random() > 0.5:
                image = self.random_rotate(image)
            if random.random() > 0.5:
                image = self.random_scale(image)
            if random.random() > 0.5:
                image = self.random_shift(image)
                
            image = self.transform(image)
        return image, label
    
    def random_rotate(self, img, degree=15):
        angle = random.uniform(-degree, degree)
        return transforms.functional.rotate(img, angle)
    
    def random_scale(self, img, scale_range=(0.9, 1.1)):
        scale = random.uniform(*scale_range)
        h, w = img.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        return transforms.functional.resize(img, (new_h, new_w))
    
    def random_shift(self, img, max_shift=2):
        h_shift = random.randint(-max_shift, max_shift)
        v_shift = random.randint(-max_shift, max_shift)
        return transforms.functional.affine(img, angle=0, translate=(h_shift, v_shift), scale=1.0, shear=0)

def get_data_loaders(batch_size=64, transform=None):
    train_images = load_mnist_images('./MINIST-master/data/train-images-idx3-ubyte.gz')
    train_labels = load_mnist_labels('./MINIST-master/data/train-labels-idx1-ubyte.gz')
    test_images = load_mnist_images('./MINIST-master/data/t10k-images-idx3-ubyte.gz')
    test_labels = load_mnist_labels('./MINIST-master/data/t10k-labels-idx1-ubyte.gz')

    train_dataset = MNISTDataset(train_images, train_labels, transform)
    test_dataset = MNISTDataset(test_images, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader