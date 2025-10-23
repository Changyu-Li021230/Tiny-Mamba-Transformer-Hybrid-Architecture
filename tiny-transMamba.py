import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.distributions.beta import Beta

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 极简数据预处理
def build_transform(train=True):
    if train:
        return transforms.Compose([ 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2023, 0.1994, 0.2010]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2023, 0.1994, 0.2010]),
        ])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, 
                               transform=build_transform(train=True))
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, 
                              transform=build_transform(train=False))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# 极小卷积块 (简化后的卷积块)
class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 极小网络架构
class TinyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 输入stem
        self.stem = nn.Sequential(
            SimpleConvBlock(3, 16),  # 减小输入通道
        )
        
        # 主体结构
        self.blocks = nn.Sequential(
            self._make_layer(16, 32, num_blocks=1),
            nn.MaxPool2d(2),
            
            self._make_layer(32, 64, num_blocks=1),
            nn.MaxPool2d(2),
        )
        
        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(SimpleConvBlock(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 使用ReLU
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# 混合增强 (MixUp + CutMix)
class HybridAugment:
    def __init__(self, alpha=1.0, cutmix_prob=0.5):
        self.alpha = alpha
        self.cutmix_prob = cutmix_prob
        self.beta = Beta(alpha, alpha)
        
    def __call__(self, x, y):
        if np.random.rand() < self.cutmix_prob:
            # CutMix
            lam = self.beta.sample().item()
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            return x, y, y, lam  # 返回targets_a (y) 和 targets_b (y) 以及 lam
        else:
            # MixUp
            lam = self.beta.sample().item()
            x = lam * x + (1 - lam) * x.flip(0)
            return x, y, y, lam  # 返回targets_a (y) 和 targets_b (y) 以及 lam
        
    def _rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

# 训练函数
def train(model, loader, optimizer, criterion, augment, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 应用混合增强
        inputs, targets_a, targets_b, lam = augment(inputs, targets)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        
        # 混合损失计算
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # 计算指标
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).float() + 
                   (1 - lam) * predicted.eq(targets_b).float()).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

# 测试函数
def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

# 主训练流程
def main():
    # 初始化模型
    model = TinyNet().to(device)
    
    # 损失函数 (带标签平滑)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 数据增强
    augment = HybridAugment(alpha=1.0, cutmix_prob=0.5)
    
    # 训练循环
    best_acc = 0
    for epoch in range(20):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, augment)
        test_loss, test_acc = test(model, test_loader, criterion)
        
        scheduler.step()
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1:03d}: "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    print(f"Best Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
