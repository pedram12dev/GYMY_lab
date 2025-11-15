# Lab: Face — scenarios and acceptance

## Goals

- Image → embedding (512) with stable preprocessing
- Store/read integrity
- Ready for websocket real-time matching

## Models

- det_10g.onnx (detection, light)
- scrfd.onnx (detection, main)
- w600k_r50.onnx (ArcFace)
- genderage.onnx (aux)

## Thresholds (MVP)

- Detection confidence: 0.5
- NMS IoU: 0.45
- Cosine login threshold: 0.40 (tune with data)

## Test scenarios

- Single-face, good lighting
- Multi-face, pick highest score
- Profile view / low light
- No-face image → empty boxes
- Model path error → clear failure

## Pass criteria

- Embedding length==512, no NaN/Inf
- Repeat same image: cosine > 0.99
- Self vs others: > threshold vs < threshold
- DB save/read exact match
