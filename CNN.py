import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import time

# 하이퍼파라미터 및 데이터 경로 설정
data_dir = "/home/limch/aa/CNN/data"
batch_size = 32
num_epochs = 15
learning_rate = 1e-6

# SimpleCNN 모델 정의
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

# 학습 관련 코드를 train_model 함수로 분리
def train_model():
    # 데이터 전처리 및 데이터셋 생성
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    num_classes = len(dataset.classes)
    
    # 데이터셋 분할: 70% 학습, 15% 검증, 15% 테스트
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # GPU 사용 설정: GPU가 있으면 사용, 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용하는 장치:", device)
    model = SimpleCNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 전체 예상 시간 계산을 위한 변수
    start_time = time.time()
    total_batches = len(train_loader) + len(val_loader)
    
    print(f"총 에폭 수: {num_epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"총 배치 수: {total_batches} (train: {len(train_loader)}, val: {len(val_loader)})")
    
    # 학습 및 검증 루프
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 학습 단계
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
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
            
            # 진행 상황 업데이트
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # 검증 단계
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]')
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()
                
                # 진행 상황 업데이트
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val
        
        # 에폭당 소요 시간 계산
        epoch_time = time.time() - epoch_start
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time = epoch_time * remaining_epochs
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"에폭 소요 시간: {epoch_time:.1f}초")
        print(f"예상 남은 시간: {estimated_time/60:.1f}분")
        print("-" * 50)

    # 테스트 단계
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

    # 모델 저장
    model_save_path = 'model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'num_classes': num_classes
    }, model_save_path)
    
    return model_save_path

if __name__ == '__main__':
    train_model()
