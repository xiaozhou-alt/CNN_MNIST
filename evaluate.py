import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import gzip
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F

# 加载数据集的函数（与train.py中相同）
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

# 定义CNN模型（与train.py中相同）
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

# 加载测试数据
test_images = load_mnist_images('./MINIST-master/data/t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('./MINIST-master/data/t10k-labels-idx1-ubyte.gz')
test_dataset = MNISTDataset(test_images, test_labels, transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型
model = CNN()
# 加载训练好的模型
model.load_state_dict(torch.load('./final_model.pth'))
model.eval()

# 评估函数
def evaluate():
    correct = 0
    total = 0
    predictions = []
    
    # 初始化混淆矩阵
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.numpy())
            
            # 构建混淆矩阵
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # 计算各项指标
    accuracy = 100 * correct / total
    precision = confusion_matrix.diag() / confusion_matrix.sum(0).float()
    recall = confusion_matrix.diag() / confusion_matrix.sum(1).float()
    f1 = 2 * precision * recall / (precision + recall)
    
    # 打印结果
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Correct/Total: {correct}/{total}')
    
    # 可视化混淆矩阵
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix.numpy(), annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('./confusion_matrix.png')
    plt.close()
    
    print('\nConfusion matrix visualization saved to confusion_matrix.png')
    
    print('\nClass-wise Metrics:')
    for i in range(10):
        print(f'Class {i}:')
        print(f'  Precision: {precision[i]:.4f}')
        print(f'  Recall:    {recall[i]:.4f}')
        print(f'  F1 Score:  {f1[i]:.4f}')
    
    # 计算宏平均
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()
    
    print('\nMacro Averages:')
    print(f'Precision: {macro_precision:.4f}')
    print(f'Recall:    {macro_recall:.4f}')
    print(f'F1 Score:  {macro_f1:.4f}')
    

if __name__ == '__main__':
    evaluate()