import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from CNN import SimpleCNN, train_model  # train_model 함수도 import

# 데이터 경로 설정
data_dir = "/home/limch/aa/CNN/data"
model_path = "/home/limch/aa/CNN/model.pth"  # 수정된 부분: 올바른 모델 파일 경로

# 모델 파일 존재 여부 확인
if not os.path.exists(model_path):
    print("저장된 모델이 없습니다. 모델 학습을 시작합니다...")
    model_path = train_model()  # subprocess 대신 직접 함수 호출
    if not os.path.exists(model_path):
        raise FileNotFoundError("모델 학습 후에도 model.pth 파일이 생성되지 않았습니다.")
else:
    print("저장된 모델을 불러옵니다.")

# 데이터 전처리 (CNN.py와 동일하게 설정)
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 데이터셋 로드
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
classes = dataset.classes
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 저장된 모델 불러오기
checkpoint = torch.load(model_path)
model = SimpleCNN(checkpoint['num_classes']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 평가 수행
correct = 0
total = 0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

print("클래스별 평가 시작...")
print("-" * 50)

with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 클래스별 정확도 계산
        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            class_correct[label] += (pred == label).item()
            class_total[label] += 1

# 전체 정확도 출력
print(f'전체 정확도: {100 * correct / total:.2f}%')
print("\n클래스별 정확도:")
print("-" * 50)

# 클래스별 정확도 출력
for i in range(len(classes)):
    if class_total[i] > 0:
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f'{classes[i]}: {class_accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')

print("-" * 50)