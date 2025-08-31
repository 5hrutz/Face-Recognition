import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the pre-trained face mask detector model
model = load_model("mask_detector.model") 

# TF_ENABLE_ONEDNN_OPTS=0

def detect_and_predict_mask(frame, face_cascade, model):
    # Convert the frame to grayscalecc
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
     
    # Loop over the detected faces
    for (x, y, w, h) in faces: 
        # Extract the face ROI (Region of Interest)
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face ROI for the model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        
        # Predict whether the face has a mask or not
        (mask, withoutMask) = model.predict(face)[0]
        
        # Determine the label and color for the bounding box
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        # Display the label and bounding box on the frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    return frame 

# Initialize the video stream
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Detect and predict mask
    frame = detect_and_predict_mask(frame, face_cascade, model)
    
    # Display the resulting frame
    cv2.imshow('Face Mask Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()