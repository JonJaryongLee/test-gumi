timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델 가중치를 저장할 디렉터리 및 파일
CONVERTED_MODELS_DIR_PATH = "./converted_models"
BASE_FILE_NAME = "mobilenet_v3_small_cifar10_finetuned.tflite"
FILE_NAME = f"{timestamp}_{BASE_FILE_NAME}"

os.makedirs(CONVERTED_MODELS_DIR_PATH, exist_ok=True)

FULL_PATH = os.path.join(CONVERTED_MODELS_DIR_PATH, FILE_NAME)

tfl_drq_model.export(FULL_PATH)
