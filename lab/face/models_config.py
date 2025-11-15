# lab/face/models_config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

# model paths
DETECTION_MODEL_PATH = str(MODELS_DIR / "scrfd.onnx")  # main detector
DETECTION_MODEL_LIGHT = str(MODELS_DIR / "det_10g.onnx")  # optional light
EMBEDDING_MODEL_PATH = str(MODELS_DIR / "w600k_r50.onnx")  # ArcFace
GENDERAGE_MODEL_PATH = str(MODELS_DIR / "genderage.onnx")  # aux

# image sizes
TARGET_DETECTION_SIZE = (640, 640)  # tune per model
ARCFACE_INPUT_SIZE = (112, 112)  # ArcFace standard

# thresholds
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45
COSINE_LOGIN_THRESHOLD = 0.40
