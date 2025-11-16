# lab/face/pipeline.py
"""
Pipeline for face detection and embedding extraction using ONNX models.
- SCRFD: Face detection
- ArcFace: Face embedding (512-dim vector)
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from lab.face.models_config import (
    ARCFACE_INPUT_SIZE,
    CONF_THRESHOLD,
    DETECTION_MODEL_PATH,
    EMBEDDING_MODEL_PATH,
    NMS_IOU_THRESHOLD,
    TARGET_DETECTION_SIZE,
)

# --- Global sessions ---
_det_session = None
_emb_session = None


# -------------------------------
# Utility functions
# -------------------------------
def _ensure_exists(path_str: str, label: str) -> str:
    """Check if model file exists, raise error if not."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at {p}")
    return str(p)


def _to_xyxy(box: np.ndarray) -> List[float]:
    """
    Convert a box to [x1, y1, x2, y2].
    If the provided box looks like [x, y, w, h] (i.e., x2<=x1 or y2<=y1),
    convert it to [x, y, x+w, y+h]. Otherwise, assume it's already [x1,y1,x2,y2].
    """
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if x2 <= x1 or y2 <= y1:
        # Interpret as [x, y, w, h]
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)
    return [x1, y1, x2, y2]


def _clip_box_xyxy(box_xyxy: List[float], img_shape: Tuple[int, int, int]) -> List[int]:
    """
    Clip box coordinates to image bounds and convert to ints.
    """
    h, w = img_shape[:2]
    x1 = int(max(0, min(box_xyxy[0], w - 1)))
    y1 = int(max(0, min(box_xyxy[1], h - 1)))
    x2 = int(max(0, min(box_xyxy[2], w - 1)))
    y2 = int(max(0, min(box_xyxy[3], h - 1)))

    # Ensure proper ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _valid_box(box_xyxy: List[int]) -> bool:
    """Check if the box has positive area."""
    x1, y1, x2, y2 = box_xyxy
    return (x2 - x1) > 0 and (y2 - y1) > 0


# -------------------------------
# Model initialization
# -------------------------------
def initialize_onnx_sessions():
    """Initialize ONNX sessions for detection and embedding models."""
    global _det_session, _emb_session

    if _det_session is None:
        det_path = _ensure_exists(DETECTION_MODEL_PATH, "SCRFD model")
        _det_session = ort.InferenceSession(
            det_path, providers=["CPUExecutionProvider"]
        )

    if _emb_session is None:
        emb_path = _ensure_exists(EMBEDDING_MODEL_PATH, "ArcFace model")
        _emb_session = ort.InferenceSession(
            emb_path, providers=["CPUExecutionProvider"]
        )

    return _det_session, _emb_session


# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_for_onnx(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize, normalize, and convert image to ONNX input format."""
    resized = cv2.resize(image, target_size)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_image.astype(np.float32) / 255.0
    model_input = np.expand_dims(normalized, axis=0)  # (1,H,W,C)
    model_input = np.transpose(model_input, (0, 3, 1, 2))  # (1,C,H,W)
    return model_input


# -------------------------------
# Detection helpers
# -------------------------------
def decode_scrfd_outputs(outputs, conf_threshold=0.5) -> List[np.ndarray]:
    """
    Decode SCRFD outputs (boxes, scores).
    SCRFD ONNX usually has multiple outputs: boxes, scores, landmarks.
    Returns filtered boxes (as float arrays) after confidence threshold and NMS.
    """
    all_boxes, all_scores = [], []

    for out in outputs:
        shape = out.shape
        if shape[-1] == 4:  # boxes
            all_boxes.append(out.reshape(-1, 4))
        elif shape[-1] == 1 or len(shape) == 1:  # scores
            all_scores.append(out.reshape(-1))

    if not all_boxes or not all_scores:
        return []

    boxes = np.vstack(all_boxes).astype(np.float32)  # (N,4)
    scores = np.hstack(all_scores).astype(np.float32)  # (N,)

    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=NMS_IOU_THRESHOLD,
    )

    if len(indices) == 0:
        return []

    # Normalize indices to a flat list
    flat_indices = []
    for i in indices:
        if isinstance(i, (list, tuple, np.ndarray)):
            flat_indices.append(int(i[0]))
        else:
            flat_indices.append(int(i))

    filtered_boxes = [boxes[idx] for idx in flat_indices]
    return filtered_boxes


def detect_faces(
    det_session: ort.InferenceSession, frame_bgr: np.ndarray
) -> List[List[int]]:
    """
    Run SCRFD detection on a frame, return raw bounding boxes as ints.
    Note: Boxes are returned without clipping to image size; clipping happens before crop.
    """
    inp = preprocess_for_onnx(frame_bgr, TARGET_DETECTION_SIZE)
    input_name = det_session.get_inputs()[0].name
    outputs = det_session.run(None, {input_name: inp})
    boxes = decode_scrfd_outputs(outputs, CONF_THRESHOLD)
    return [list(map(int, _to_xyxy(box))) for box in boxes]


# -------------------------------
# Embedding helpers
# -------------------------------
def align_face(face_img_bgr: np.ndarray) -> np.ndarray:
    """Resize cropped face to ArcFace input size."""
    return cv2.resize(face_img_bgr, ARCFACE_INPUT_SIZE)


def get_face_embedding(
    emb_session: ort.InferenceSession, face_img_bgr: np.ndarray
) -> np.ndarray:
    """Extract 512-dim embedding from a cropped face image."""
    aligned = align_face(face_img_bgr)
    input_name = emb_session.get_inputs()[0].name
    emb_input = preprocess_for_onnx(aligned, ARCFACE_INPUT_SIZE)
    emb = emb_session.run(None, {input_name: emb_input})[0]  # (1,512)
    return emb.flatten()


def extract_embeddings_from_frame(frame_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Detect faces in a frame and return embeddings for each.
    Includes box normalization and clipping to avoid empty crops.
    """
    det_session, emb_session = initialize_onnx_sessions()
    raw_boxes = detect_faces(det_session, frame_bgr)

    embeddings = []
    for box in raw_boxes:
        # Clip to image bounds and ensure positive area
        clipped = _clip_box_xyxy(box, frame_bgr.shape)
        if not _valid_box(clipped):
            continue

        x1, y1, x2, y2 = clipped
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        vec = get_face_embedding(emb_session, crop)
        if sanity_check_embedding(vec):
            embeddings.append(vec)

    return embeddings


# -------------------------------
# Similarity
# -------------------------------
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = vec_a.flatten()
    b = vec_b.flatten()
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


# -------------------------------
# Sanity check
# -------------------------------
def sanity_check_embedding(emb: np.ndarray) -> bool:
    """Validate embedding vector integrity."""
    if emb is None:
        return False
    if emb.ndim != 1 or emb.shape[0] != 512:
        return False
    if np.isnan(emb).any() or np.isinf(emb).any():
        return False
    return True
