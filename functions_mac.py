from time import sleep
import cv2
import dlib
from fastai.vision.all import *
import torch


class GenderPrediction:
    def __init__(self, model_path, img_height, img_width):
        self.model_path = model_path
        self.img_height = img_height
        self.img_width = img_width
        self.model = load_learner(model_path)

    def preprocess_image(self, image):
        cropped_face = cv2.resize(image, (self.img_height, self.img_width))
        return torch.tensor(cropped_face)

    def predict(self, image):
        tensor = self.preprocess_image(image)
        pred, _, _ = self.model.predict(tensor)
        return pred


class EmotionClassifier:
    def __init__(self, model_path, img_height, img_width):
        self.model_path = model_path
        self.img_height = img_height
        self.img_width = img_width
        self.model = load_learner(model_path)

    def preprocess_image(self, image):
        cropped_face = cv2.resize(image, (self.img_height, self.img_width))
        # Convert face to gray scaled
        gray_scaled = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        tensor = torch.tensor(gray_scaled)
        return tensor

    def predict(self, img_path):
        tensor = self.preprocess_image(img_path)
        pred, _, _ = self.model.predict(tensor)
        return pred


if __name__ == "__main__":
    # Create a VideoCapture object to capture the video from the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        exit()

    # Initialize the Dlib face detector
    detector = dlib.get_frontal_face_detector()

    # Training was done on RGB images of 192x192
    gender_classifier = GenderPrediction('./models/trained_gender_convnext_tiny_hnf.pkl', 192, 192)

    # Training of this model was done on gray-scaled images 48x48
    emotion_classifier = EmotionClassifier('./models/emotion_detector_convnext_tiny_hnf.pkl', 48, 48)

    try:
        # Loop to display the camera feed with face bounding boxes
        while True:
            # Capture each frame
            ret, frame = cap.read()

            # If the frame is read successfully, process and display it
            if ret:
                # Detect faces using Dlib
                faces = detector(frame, 1)

                # Draw bounding boxes around detected faces
                for face in faces:
                    x, y, width, height = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

                    detected_face = frame[y:y + height, x:x + width]

                    # text to display
                    text = f'{gender_classifier.predict(detected_face)}, {emotion_classifier.predict(detected_face)}'

                    # Display the text above the bounding box
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Display the processed frame
                cv2.imshow('Camera Feed', frame)

                # CPU is slow
                if not torch.cuda.is_available():
                    sleep(1)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Frame could not be read.")
                break
    except KeyboardInterrupt:
        # If Ctrl+C is pressed, exit the loop and release resources
        print("Interrupted by user. Exiting...")
    finally:
        # Release the VideoCapture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
