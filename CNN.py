import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
import os

# 하이퍼파라미터 및 데이터 경로 설정
data_dir = "/home/limch/aa/CNN/exe"
batch_size = 32
num_epochs = 30
learning_rate = 1e-4
weight_decay_rate = 1e-5

# 저장 경로 설정
model_save_dir = './saved_models_v_experiment_ClassWeights' # 실험 결과 저장 디렉토리 변경
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

best_model_path = os.path.join(model_save_dir, 'best_model.pth')
final_model_path = os.path.join(model_save_dir, 'final_model.pth')


# SimpleCNN 모델 정의 (이전과 동일)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model():
    # 데이터 전처리 (이전과 동일)
    train_data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_test_data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_pil = datasets.ImageFolder(root=data_dir)
    num_classes = len(dataset_pil.classes)
    print(f"발견된 클래스: {dataset_pil.classes}")
    print(f"클래스-인덱스 매핑: {dataset_pil.class_to_idx}") # 클래스 순서 확인을 위해 출력
    print(f"총 클래스 수: {num_classes}")

    dataset_size = len(dataset_pil)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_indices, val_indices, test_indices = random_split(range(len(dataset_pil)), [train_size, val_size, test_size])
    
    class DatasetFromSubset(torch.utils.data.Dataset):
        def __init__(self, subset_indices, full_dataset_pil, transform=None):
            self.subset_indices = subset_indices
            self.full_dataset_pil = full_dataset_pil
            self.transform = transform

        def __getitem__(self, index):
            actual_index = self.subset_indices[index]
            x, y = self.full_dataset_pil.samples[actual_index]
            x = self.full_dataset_pil.loader(x)
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset_indices)

    train_dataset = DatasetFromSubset(train_indices.indices, dataset_pil, transform=train_data_transforms)
    val_dataset = DatasetFromSubset(val_indices.indices, dataset_pil, transform=val_test_data_transforms)
    test_dataset = DatasetFromSubset(test_indices.indices, dataset_pil, transform=val_test_data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용하는 장치:", device)
    model = SimpleCNN(num_classes).to(device)

    # --- 클래스 가중치 설정 ---
    # dataset_pil.class_to_idx 를 보고 Benign과 Malware의 인덱스를 확인해야 합니다.
    # 예: {'Benign': 0, 'Malware': 1} 이라고 가정
    # Malware 재현율을 높이기 위해 Malware 클래스(인덱스 1)에 더 높은 가중치를 부여합니다.
    # 이 가중치 값은 데이터 불균형(Benign: 11613, Malware: 26407)과 중요도에 따라 조절합니다.
    # Malware가 다수 클래스임에도 재현율이 낮으므로, Malware 오류에 대한 페널티를 높이는 것입니다.
    # weights = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device) # 예: Malware(인덱스 1)에 2배 가중치
    
    # class_to_idx를 기반으로 동적으로 가중치 설정 (Malware에 더 높은 가중치)
    # Malware가 소수 클래스라면 1/빈도수 비율로 가중치를 줄 수 있지만, 여기서는 다수 클래스임에도 재현율이 낮으므로
    # Malware 예측 실패에 대한 페널티를 더 강하게 주기 위해 임의의 비율로 가중치를 설정합니다.
    # (주의: 아래 코드는 dataset_pil.class_to_idx의 실제 내용을 보고, Benign과 Malware의 인덱스에 맞게 수정해야 할 수 있습니다.)
    malware_class_label = None
    benign_class_label = None
    for class_name, idx in dataset_pil.class_to_idx.items():
        if "malware" in class_name.lower(): # 클래스 이름에 "malware"가 포함되어 있으면
            malware_class_label = idx
        elif "benign" in class_name.lower(): # 클래스 이름에 "benign"이 포함되어 있으면
            benign_class_label = idx

    if malware_class_label is None or benign_class_label is None:
        print("경고: 'Malware' 또는 'Benign' 클래스를 찾을 수 없습니다. 기본 가중치를 사용합니다.")
        class_weights_tensor = None
    else:
        # Malware 오류에 더 큰 페널티 부여 (예: Benign 가중치 1, Malware 가중치 2)
        # weights_list 초기화 (클래스 수만큼)
        weights_list = [1.0] * num_classes 
        weights_list[benign_class_label] = 1.0  # Benign 가중치
        weights_list[malware_class_label] = 2.0 # Malware 가중치를 2배로 설정 (이 값을 조절하며 실험)
        
        class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
        print(f"사용된 클래스 가중치 ({dataset_pil.class_to_idx}): {class_weights_tensor.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # weight 파라미터에 가중치 텐서 전달
    # --------------------------

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False)

    start_time = time.time()
    best_val_acc = 0.0

    print(f"총 에폭 수: {num_epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"초기 학습률: {learning_rate}")
    # (이하 학습/검증/테스트 루프는 이전과 동일)

    for epoch in range(num_epochs):
        epoch_start = time.time()

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
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f"{optimizer.param_groups[0]['lr']:.1e}"})


        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]')
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels) # 가중치가 적용된 손실 계산
                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val
        
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if prev_lr != current_lr:
            print(f"Epoch {epoch+1}: 학습률 변경 감지: {prev_lr:.1e} -> {current_lr:.1e}")

        epoch_time = time.time() - epoch_start
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time = epoch_time * remaining_epochs

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"현재 학습률: {optimizer.param_groups[0]['lr']:.1e}")
        print(f"에폭 소요 시간: {epoch_time:.1f}초")
        if remaining_epochs > 0:
            print(f"예상 남은 시간: {estimated_time/60:.1f}분")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
                'num_classes': num_classes,
                'class_to_idx': dataset_pil.class_to_idx
            }, best_model_path)
            print(f"*** 최고 검증 정확도 갱신: {best_val_acc:.4f}. 모델 저장: {best_model_path} ***")
        print("-" * 60)

    print("\n최종 테스트 시작...")
    if os.path.exists(best_model_path):
        print(f"가장 성능이 좋았던 모델 로드: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("저장된 최고 성능 모델이 없습니다. 마지막 에폭 모델로 테스트합니다.")

    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    test_pbar = tqdm(test_loader, desc='[Test]')
    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels) # 테스트 시에도 가중치 적용된 손실 (평가 지표에는 직접 영향 없음)
            running_test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (preds == labels).sum().item()
            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    test_loss = running_test_loss / total_test
    test_acc = correct_test / total_test
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'num_classes': num_classes,
        'class_to_idx': dataset_pil.class_to_idx
    }, final_model_path)
    print(f"최종 모델 저장: {final_model_path}")
    
    total_execution_time = time.time() - start_time
    print(f"총 실행 시간: {total_execution_time/60:.2f} 분")

    return final_model_path

if __name__ == '__main__':
    train_model()