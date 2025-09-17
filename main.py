from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import pickle
import numpy as np
from ultralytics import YOLO

# -------------- File Paths ---------------


MODEL_PATH = "best.pt"
PKL_PATH = "training_summary.pkl"

# -------------- Load Class Names ---------------
def load_class_names(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            summary = pickle.load(f)
        if "class_names" in summary:
            raw = summary["class_names"]
            return {int(k): v for k, v in raw.items()}
        elif "names" in summary and isinstance(summary["names"], (list, dict)):
            if isinstance(summary["names"], list):
                return {i: name for i, name in enumerate(summary["names"])}
            else:
                return summary["names"]
    except Exception as e:
        print(f"Error loading class names from pkl: {e}")
    return {}

class_names = load_class_names(PKL_PATH)

# -------------- Load Model ---------------
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


 # -------------- Start FastAPI ---------------
app = FastAPI(title="YOLO Object Detection API (JSON only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


 # -------------- Response Model ---------------
class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox: BBox

class PredictionResponse(BaseModel):
    detections: List[Detection]
    num_detections: int


 # -------------- Inference Function ---------------
def predict_yolo(image: Image.Image, conf_threshold: float = 0.25):
    # Convert PIL image to numpy array
    img = np.array(image.convert("RGB"))
    results = model.predict(source=img, conf=conf_threshold, verbose=False)
    detections = []
    if len(results) == 0:
        return []
    r = results[0]
    if hasattr(r, "boxes") and r.boxes is not None:
        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
        except Exception:
            xyxy = np.asarray(r.boxes.xyxy)
            confs = np.asarray(r.boxes.conf)
            cls_ids = np.asarray(r.boxes.cls)
        for box, conf, cls in zip(xyxy, confs, cls_ids):
            class_id = int(cls)
            label = class_names.get(class_id, str(class_id))
            detections.append(Detection(
                class_id=class_id,
                label=label,
                confidence=float(conf),
                bbox=BBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3])
                )
            ))
    return detections


 # -------------- API Endpoint ---------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return {"detections": [], "num_detections": 0}
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    detections = predict_yolo(image)
    return PredictionResponse(
        detections=detections,
        num_detections=len(detections)
    )
