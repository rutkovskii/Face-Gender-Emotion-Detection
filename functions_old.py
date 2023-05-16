from time import sleep

import numpy as np
import cv2
import dlib
from fastai.vision.all import *
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from cam import gstreamer_pipeline, gstreamer_pipeline_hard


def predict_gender(model, frame):
    # Extract the face and resize it to the size expected by the gender classifier - 192x192
    cropped_face = cv2.resize(frame[y:y + height, x:x + width], (192, 192))

    # transform cropped_face to tensor
    tensor = torch.tensor(cropped_face)
    # predict gender
    gender, _, _ = gender_classifier.predict(tensor)

    return gender


def predict_emotion(model, frame):
    # Extract the face and resize it to the size expected by the emotion classifier - 48x48
    cropped_face = cv2.resize(frame[y:y + height, x:x + width], (48, 48))
    # Convert face to gray scaled
    gray_scaled = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
    tensor = torch.tensor(gray_scaled)

    # prediction emotion
    emotion, _, _ = emotion_classifier.predict(tensor)
    return emotion
  
def main():
	# Create a VideoCapture object to capture the video from the camera
	# cap = cv2.VideoCapture(0)
	cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

	# Check if the camera is opened successfully
	if not cap.isOpened():
		print("Error: Camera could not be opened.")
		exit()

	# Initialize the Dlib face detector
	detector = dlib.get_frontal_face_detector()

	# Load the gender classification model
	gender_classifier = load_learner('./models/trained_gender_resnet26d.pkl')

	# Load the emotion classification model
	emotion_classifier = load_learner('./models/emotion_detector_convnext_tiny_hnf.pkl')

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

		            # text to display
		            text = f'{predict_gender(gender_classifier, frame)}, {predict_emotion(emotion_classifier, frame)}'

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


if __name__ == "__main__":
  main()
