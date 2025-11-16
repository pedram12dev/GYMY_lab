from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODELS_DIR = ROOT / "models"
EMBEDDING_MODEL_PATH = str(MODELS_DIR / "w600k_r50.onnx")
ARCFACE_INPUT_SIZE = (112, 112)
DETECTION_MODEL_PATH = str(MODELS_DIR / "scrfd.onnx")      # optional
DETECTION_MODEL_LIGHT = str(MODELS_DIR / "det_10g.onnx")   # optional
GENDERAGE_MODEL_PATH = str(MODELS_DIR / "genderage.onnx")  # optional
TARGET_DETECTION_SIZE = (640, 640)
