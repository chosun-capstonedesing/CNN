import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ------------------
# 하이퍼파라미터 및 경로 설정
# ------------------
data_dir = "/home/limch/aa/data"  # 이미지들이 클래스별 폴더로 저장된 경로
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# ------------------
# 데이터 전처리 (그레이스케일로 변환 후 256x256 크기로 리사이즈)
# ------------------
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 입력 데이터를 그레이스케일로 변환
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ImageFolder를 이용하여 데이터셋 생성 (폴더명이 라벨)
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
num_classes = len(dataset.classes)
print("총 클래스 수:", num_classes)

# ------------------
# 데이터셋 분할: 70% 학습, 15% 검증, 15% 테스트
# ------------------
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------
# 간단한 CNN 모델 정의 (그레이스케일 입력: 채널=1)
# ------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # 입력 채널: 1 (그레이스케일)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32 x 256 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 32 x 128 x 128

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 128 x 128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 64 x 64 x 64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 128 x 64 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                            # 128 x 32 x 32
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# ------------------
# GPU 사용 설정: GPU가 있으면 사용, 없으면 CPU 사용
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용하는 장치:", device)
model = SimpleCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------
# 학습 및 검증 루프
# ------------------
for epoch in range(num_epochs):
    # 학습 단계
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (preds == labels).sum().item()
    
    train_loss = running_loss / total_train
    train_acc = correct_train / total_train
    
    # 검증 단계
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (preds == labels).sum().item()
    
    val_loss = running_val_loss / total_val
    val_acc = correct_val / total_val
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ------------------
# 테스트 단계
# ------------------
model.eval()
running_test_loss = 0.0
correct_test = 0
total_test = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_test_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (preds == labels).sum().item()

test_loss = running_test_loss / total_test
test_acc = correct_test / total_test

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
