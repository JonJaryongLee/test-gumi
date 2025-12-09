import os
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import random
import tensorflow as tf
import ai_edge_torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V3_Small_Weights

import time

from ai_edge_litert.interpreter import Interpreter

# 전체 테스트할 이미지 갯수
MAX_TEST_SAMPLES = 5

# 시드 고정
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CIFAR-10 의 평균과 표준편차
mean = torch.tensor([0.4914, 0.4822, 0.4465])
std = torch.tensor([0.2470, 0.2435, 0.2616])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

print(f"testset 이미지 갯수: {len(testset)}")
print(f"실제 테스트할 이미지 갯수: {MAX_TEST_SAMPLES}")

# 타임스탬프 방식이므로, 직접 수정 필요
TFLITE_MODEL_PATH = './converted_models/20251027_171157_mobilenet_v3_small_cifar10_finetuned.tflite'

# .tflite 작동을 위해선 배치사이즈 1 필수
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

correct = 0
total = 0
total_inference_time = 0.0

idx = 0

print("================================")
print("테스트 실행")
print("================================")

for inputs_torch, labels_torch in testloader:

    # MAX_TEST_SAMPLES 크기만큼만 테스트합니다.
    if idx >= MAX_TEST_SAMPLES:
        break

    input_data_fp32 = inputs_torch.numpy().astype(np.float32)

    # 입력 텐서 설정 및 추론 실행
    interpreter.set_tensor(input_details['index'], input_data_fp32)

    # 하나의 이미지의 실행시간 계산
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    total_inference_time += (end_time - start_time)

    print(f"img_idx: {idx}")
    print(f"time: {end_time - start_time}")
    print("================================")
    idx += 1

    # 출력 텐서 가져오기
    output_data_fp32 = interpreter.get_tensor(output_details['index'])
    
    # 예측 결과 및 정확도 계산
    predicted = np.argmax(output_data_fp32, axis=1) # 가장 높은 값의 인덱스 (클래스)
    labels_np = labels_torch.numpy()
    
    total += labels_np.shape[0]
    correct += (predicted == labels_np).sum()

print("테스트 종료")
print("================================")

# 최종 정확도 계산 및 출력
tflite_acc = 100 * correct / total
print(f"\n--- TFLite 모델 평가 결과 ---")
print(f"TFLite 모델 테스트 정확도: {tflite_acc:.2f}%")

if total > 0:
    avg_inference_time = total_inference_time / total
    print(f"평균 단일 이미지 추론 시간: {avg_inference_time:.4f}초")
