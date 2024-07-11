import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
import datetime
from pathlib import Path
from PIL import Image


def main(person_name, person_id):
    # # For static images:
    # IMAGE_FILES = []
    # with mp_face_detection.FaceDetection(
    #     model_selection=1, min_detection_confidence=0.5) as face_detection:
    # for idx, file in enumerate(IMAGE_FILES):
    #     image = cv2.imread(file)
    #     # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    #     results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #     # Draw face detections of each face.
    #     if not results.detections:
    #     continue
    #     annotated_image = image.copy()
    #     for detection in results.detections:
    #     print('Nose tip:')
    #     print(mp_face_detection.get_key_point(
    #         detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
    #     mp_drawing.draw_detection(annotated_image, detection)
    #     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f"./image_saved/{person_name}_{person_id}_{timestamp}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        print(f"[{save_dir}] made!!!!!!!!!!!")

    count = 0

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image_copy = image.copy()

            H = image.shape[0]
            W = image.shape[1]

            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)

            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image_copy, detection)

                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * W)
                    y = int(bbox.ymin * H)
                    h = int(bbox.height * H)
                    w = int(bbox.width * W)
                    cropped = image[y:y+h, x:x+w, :]
                    cropped = Image.fromarray(cropped)
                    count += 1

                    cropped.save(f"{str(save_dir.absolute())}/{str(count).zfill(4)}.png")
                    # cv2.imwrite(f"{str(save_dir.absolute())}/{str(count).zfill(4)}.png", cropped)
            else:
                print("not found..........")

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image_copy, 1))
            if cv2.waitKey(5) & 0xFF == 27:
              break
        cap.release()

if __name__ == "__main__":
    person_name = "이재철"
    person_id = "101"
    main(person_name, person_id)