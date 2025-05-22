import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from CNN import SimpleCNN, train_model  # train_model 함수도 import
import tqdm

# 데이터 경로 설정
data_dir = "/home/limch/aa/CNN/exe"
# 🚨 중요: 실제로 평가할 학습된 모델 파일의 정확한 경로로 수정해주세요.
# 예: model_path = "./saved_models_v_experiment_ClassWeights/best_model.pth"
model_path = "/home/limch/aa/CNN/saved_models_v_experiment_ClassWeights/best_model.pth"  # 모델 경로 설정

# 모델 파일 존재 여부 확인
if not os.path.exists(model_path):
    print(f"지정된 경로에 모델이 없습니다: {model_path}")
    print("모델 학습을 시작합니다...")
    # train_model() 함수가 정의된 스크립트(CNN.py)가 동일한 디렉토리에 있거나,
    # 파이썬 경로에 설정되어 있어야 합니다.
    # train_model() 함수는 학습된 모델의 경로를 반환해야 합니다.
    model_path = train_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 학습 후에도 모델 파일({model_path})이 생성되지 않았습니다.")
    print(f"학습 완료. 평가할 모델 경로: {model_path}")
else:
    print(f"저장된 모델을 불러옵니다: {model_path}")

# 데이터 전처리 (Normalize 추가)
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 🚨 학습 때와 동일한 정규화 추가
])

# 데이터셋 로드
# ImageFolder는 data_dir 아래에 클래스별 서브디렉토리가 있다고 가정합니다.
try:
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
except FileNotFoundError:
    print(f"오류: 데이터 디렉토리 '{data_dir}'를 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()
except Exception as e:
    print(f"오류: 데이터셋 로드 중 문제가 발생했습니다: {e}")
    exit()

if not dataset.samples:
    print(f"오류: '{data_dir}' 디렉토리에서 이미지를 찾을 수 없습니다. 클래스별 하위 폴더에 이미지가 있는지 확인해주세요.")
    exit()

classes = dataset.classes
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True) # num_workers, pin_memory 추가

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"평가에 사용하는 장치: {device}")

# 저장된 모델 불러오기
try:
    checkpoint = torch.load(model_path, map_location=device) # map_location 추가 (CPU에서 GPU 모델 로드 등)
except FileNotFoundError:
    print(f"오류: 모델 파일 '{model_path}'를 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()
except Exception as e:
    print(f"오류: 모델 파일 로드 중 문제가 발생했습니다: {e}")
    exit()


# 'num_classes' 키 확인 및 모델 초기화
if 'num_classes' not in checkpoint:
    print("오류: 체크포인트에 'num_classes' 정보가 없습니다. 모델을 올바르게 저장했는지 확인해주세요.")
    # 대안: dataset.classes에서 num_classes를 가져올 수 있습니다.
    # num_classes_from_dataset = len(classes)
    # print(f"데이터셋에서 추론된 클래스 수: {num_classes_from_dataset}. 이 값을 사용합니다.")
    # model = SimpleCNN(num_classes_from_dataset).to(device)
    exit() # 또는 적절한 오류 처리

model = SimpleCNN(checkpoint['num_classes']).to(device)

if 'model_state_dict' not in checkpoint:
    print("오류: 체크포인트에 'model_state_dict' 정보가 없습니다.")
    exit()

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 평가 수행
correct = 0
total = 0
# 클래스 수가 로드된 모델과 일치하는지 확인
num_loaded_classes = checkpoint['num_classes']
if len(classes) != num_loaded_classes:
    print(f"경고: 데이터셋의 클래스 수({len(classes)})와 로드된 모델의 클래스 수({num_loaded_classes})가 일치하지 않습니다.")
    # 필요시 로드된 모델의 클래스 수를 기준으로 class_correct, class_total 초기화
    # 여기서는 dataset.classes를 기준으로 하지만, 불일치 시 문제가 될 수 있음

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

y_true = []
y_pred = []

print("\n클래스별 평가 시작...")
print("-" * 50)

with torch.no_grad():
     for images, labels in tqdm.tqdm(dataloader, desc="평가 진행"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        
        for i in range(len(labels)): # 배치 내 각 샘플에 대해
            label_scalar = labels[i].item() # 스칼라 값으로 변환
            pred_scalar = predicted[i].item() # 스칼라 값으로 변환
            if pred_scalar == label_scalar:
                class_correct[label_scalar] += 1
            class_total[label_scalar] += 1


# 전체 정확도 출력
if total == 0:
    print("평가할 데이터가 없습니다.")
else:
    print(f'전체 정확도: {100 * correct / total:.2f}% ({correct}/{total})')

print("\n클래스별 정확도:")
print("-" * 50)
for i in range(len(classes)):
    if class_total[i] > 0:
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f'{classes[i]}: {class_accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
    else:
        print(f'{classes[i]}: 데이터 없음 (0/0)')

print("-" * 50)

# sklearn.metrics를 사용하기 전에 y_true, y_pred가 비어있지 않은지 확인
if not y_true or not y_pred:
    print("Precision, Recall, F1-Score를 계산할 예측 결과가 없습니다.")
else:
    # === 추가: Precision, Recall, F1-Score 계산 ===
    print("\nPrecision, Recall, F1-Score 전체 지표:")
    print("-" * 50)

    # 평균 방식 (binary면 binary, 멀티클래스면 macro/weighted 선택)
    # 클래스가 2개인 경우 (Benign, Malware) 'binary'를 사용하려면 pos_label을 지정해야 할 수 있습니다.
    # 여기서는 target_names를 사용하므로 macro 또는 weighted가 더 일반적입니다.
    average_mode = "macro"  # 또는 "weighted" 가능

    # zero_division 파라미터 추가 (0으로 나누는 경우 경고 대신 특정 값 반환)
    precision = precision_score(y_true, y_pred, average=average_mode, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average_mode, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average_mode, zero_division=0)

    print(f'Precision (정밀도) ({average_mode} avg): {precision:.4f}')
    print(f'Recall (재현율) ({average_mode} avg): {recall:.4f}')
    print(f'F1-Score ({average_mode} avg): {f1:.4f}')

    # === 추가: 클래스별 세부 리포트 출력 ===
    print("\n클래스별 세부 리포트:")
    print("-" * 50)
    try:
        # target_names가 실제 클래스 이름과 순서가 맞는지 확인
        # dataset.classes는 ImageFolder가 자동으로 폴더명 순으로 정렬한 리스트를 제공
        # 모델이 학습될 때 사용된 class_to_idx의 순서와 일치해야 함
        # 체크포인트에 class_to_idx가 저장되어 있다면 그것을 사용하는 것이 더 안전
        loaded_class_to_idx = checkpoint.get('class_to_idx', None)
        target_names_for_report = classes # 기본값
        if loaded_class_to_idx:
            # class_to_idx의 value(인덱스)를 기준으로 key(클래스 이름)를 정렬
            sorted_class_names = sorted(loaded_class_to_idx.items(), key=lambda item: item[1])
            target_names_for_report = [name for name, idx in sorted_class_names]
            if len(target_names_for_report) != checkpoint['num_classes']:
                 print(f"경고: 로드된 class_to_idx의 클래스 수와 num_classes가 불일치합니다. dataset.classes를 사용합니다.")
                 target_names_for_report = classes


        print(classification_report(y_true, y_pred, target_names=target_names_for_report, zero_division=0))
    except ValueError as e:
        print(f"classification_report 생성 중 오류: {e}")
        print("y_true 또는 y_pred에 없는 레이블이 target_names에 포함되었거나, 데이터셋 클래스와 모델 출력 간 불일치가 있을 수 있습니다.")
        print("기본 숫자 레이블로 리포트 출력 시도:")
        print(classification_report(y_true, y_pred, zero_division=0))