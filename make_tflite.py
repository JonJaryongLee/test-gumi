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

# cpu 로 가져오기
loaded_model = torchvision.models.mobilenet_v3_small(
    weights=MobileNet_V3_Small_Weights.DEFAULT,
    map_location=torch.device('cpu')
)

loaded_model.classifier[3] = nn.Linear(in_features=loaded_model.classifier[3].in_features, out_features=10)

MODEL_STATE_DICT_DIR_PATH = "./models_state_dict"
FILE_NAME = "20251209_082006_mobilenet_v3_small_cifar10_finetuned.pth"
FULL_PATH = os.path.join(MODEL_STATE_DICT_DIR_PATH, FILE_NAME)

# 모든 state 를 cpu 로 로드하기
loaded_model.load_state_dict(torch.load(FULL_PATH, map_location=torch.device("cpu")))
# cpu 로 모델을 보내야 함
loaded_model.to("cpu")
# 평가모드로 변경
loaded_model.eval()

# 모든 연산자는 cpu 에서 실행되어야 함
# 배치사이즈는 1 필수
sample_args = (torch.randn(1, 3, 224, 224).to("cpu"),)

tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT]}

tfl_drq_model = ai_edge_torch.convert(
    loaded_model, sample_args, _ai_edge_converter_flags=tfl_converter_flags
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델 가중치를 저장할 디렉터리 및 파일
CONVERTED_MODELS_DIR_PATH = "./converted_models"
BASE_FILE_NAME = "mobilenet_v3_small_cifar10_finetuned.tflite"
FILE_NAME = f"{timestamp}_{BASE_FILE_NAME}"

os.makedirs(CONVERTED_MODELS_DIR_PATH, exist_ok=True)

FULL_PATH = os.path.join(CONVERTED_MODELS_DIR_PATH, FILE_NAME)

tfl_drq_model.export(FULL_PATH)
