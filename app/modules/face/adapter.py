from functools import lru_cache
import numpy as np, cv2, onnxruntime as ort
from app.modules.face.models_config import EMBEDDING_MODEL_PATH, ARCFACE_INPUT_SIZE


def _preprocess_arcface(img_bgr: np.ndarray) -> np.ndarray:
    dst = cv2.resize(img_bgr, ARCFACE_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB).astype(np.float32)
    dst = (dst / 127.5) - 1.0
    dst = np.transpose(dst, (2, 0, 1))[None, ...]
    return dst

@lru_cache
def _arcface_session():
    sess = ort.InferenceSession(EMBEDDING_MODEL_PATH, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name

def image_bytes_to_embedding(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("cannot decode image")
    blob = _preprocess_arcface(img)
    sess, input_name = _arcface_session()
    out = sess.run(None, {input_name: blob})[0][0].astype(np.float32)
    n = np.linalg.norm(out)
    if n == 0: raise ValueError("zero embedding")
    return out / n
