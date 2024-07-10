import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from configs import Config
import datetime

FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the live stream mode:
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    print(f'face detector result: {len(result.detections)} / {timestamp_ms}')

def get_face_detector_options(model_path=Config.MODEL_PATH, mode="live_stream"):

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=Config.MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM if mode == "live_stream" else VisionRunningMode.IMAGE,
        result_callback=print_result if mode == "live_stream" else None,
        )
    
    return options


