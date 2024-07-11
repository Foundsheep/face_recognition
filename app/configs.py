import json
import torch
from pathlib import Path

class Config:
    MODEL_PATH = r"C:\Users\msi\Desktop\workspace\026_face_detection\face_recognition\app\models\blaze_face_short_range.tflite"
    CLASSIFICATION_MODEL_NAME = "resnet18"

    j = None
    j_path = None
    curdir = Path(__file__).absolute().parent
    try:
        j_path = curdir / Path("training_jsons") / "label_to_id.json"
        with open(j_path, "r") as f:
            j = json.load(f)
    except:
        print("error occurred when loading json for num_classes")
        print(f".......... [{j_path = }]")
    CLASSIFICATION_NUM_CLASSES = len(j.keys()) if j is not None else 5
    RESIZED_HEIGHT = 224
    RESIZED_WIDTH = 224
    SHUFFLE = True
    BATCH_SIZE = 4
    DL_NUM_WORKERS = 1
    MAX_EPOCHS = 3
    MIN_EPOCHS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    READ_STEP = 5
    LOG_EVERY_N_STEPS = 1
    USE_EARLY_STOP = "Y"
    TRAINING_LOG_DIR = curdir / "train_result"
    if not TRAINING_LOG_DIR:
        TRAINING_LOG_DIR.mkdir()
