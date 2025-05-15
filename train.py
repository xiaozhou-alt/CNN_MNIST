import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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
            image = self.transform(image)
        return image, label

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

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 使用data_loader获取数据
train_loader, test_loader = get_data_loaders(batch_size=64, transform=transform)

# 初始化模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化损失记录
loss_records = pd.DataFrame(columns=['Epoch', 'Loss'])

# 训练模型
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
            # 记录损失值
            global loss_records
            new_record = pd.DataFrame({'Epoch': [epoch], 'Loss': [loss.item()]})
            loss_records = pd.concat([loss_records, new_record], ignore_index=True)

# 测试模型
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# 训练和测试
for epoch in range(1, 6):
    train(epoch)
    test()
    
    # 每轮训练后保存模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_records
    }, f'./model_epoch_{epoch}.pth')

# 将损失记录保存到Excel文件
loss_records.to_excel('./loss_records.xlsx', index=False)

# 保存最终模型
torch.save(model.state_dict(), './final_model.pth')