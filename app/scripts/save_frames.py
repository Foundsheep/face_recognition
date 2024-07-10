import cv2
import sys
from pathlib import Path
import time
from PIL import Image

try:
    from models.face_detector import get_face_detector_options
except:
    sys.path.append(str(Path(__file__).absolute().parent.parent))
    print(f"[{str(Path(__file__).absolute().parent.parent)}] added")
    from models.face_detector import get_face_detector_options
import mediapipe as mp
import datetime

FaceDetector = mp.tasks.vision.FaceDetector
COLOR = (0, 255, 0)

def capture(name):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    # timestamp_ms = int(time.time() * 1000)

    face_detector_options = get_face_detector_options()
    with FaceDetector.create_from_options(face_detector_options) as detector:
        cap = cv2.VideoCapture(0)

        count = 0
        ms = 0

        save_dir = Path(f"./{name}_{timestamp}")
        if not save_dir.exists():
            save_dir.mkdir()
            print(f"[{save_dir}] made!!!!!!!!!!!")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            ms += 1
            results = detector.detect_async(mp_image, ms)
            image_copy = image.copy()

            if results is not None:
                for detection in results.detections:
                    bbox = detection.bounding_box
                    x = bbox.origin_x
                    y = bbox.origin_y
                    h = bbox.height
                    w = bbox.width
                    cropped = image[y:y+h, x:x+w, :]

                    cv2.rectangle(image_copy, (x, y), (x+w, y+h), COLOR, 10)
                
            else:
                print("not found.....")

            count += 1
            cv2.imwrite(f"{save_dir.absolute()}/{str(count).zfill(4)}.png", image_copy)
            cv2.imshow("image captured", cv2.flip(image_copy, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        print("DONE")

def detect_face(folder_name):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    # timestamp_ms = int(time.time() * 1000)

    face_detector_options = get_face_detector_options()
    with FaceDetector.create_from_options(face_detector_options) as detector:
        cap = cv2.VideoCapture(0)

        count = 0
        ms = 0

        save_dir = Path(f"./{name}_{timestamp}")
        if not save_dir.exists():
            save_dir.mkdir()
            print(f"[{save_dir}] made!!!!!!!!!!!")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # image.flags.writeable = False
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            ms += 1
            # results = detector.detect_async(mp_image, ms)
            results = detector.detect(mp_image)
            image_copy = image.copy()

            print(f"!!!!!!!!!!!!!!!!! {type(results)}")
            if results is not None:
                for detection in results.detections:
                    bbox = detection.bounding_box
                    x = bbox.origin_x
                    y = bbox.origin_y
                    h = bbox.height
                    w = bbox.width
                    cropped = image[y:y+h, x:x+w, :]

                    cv2.rectangle(image_copy, (x, y), (x+w, y+h), COLOR, 10)
                
            else:
                print("not found.....")

            count += 1
            # pil_image = Image.fromarray(image_copy)
            # pil_image.save(f"{save_dir.absolute()}/{str(count).zfill(4)}.png")
            cv2.imwrite(f"{save_dir.absolute()}/{str(count).zfill(4)}.png", image_copy)
            print("========= written!!!")
            cv2.imshow("image captured", cv2.flip(image_copy, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        print("DONE")
    


if __name__ == "__main__":
    capture("DJ")