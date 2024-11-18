import cv2
import numpy as np
import math
import time
import os
import requests
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from dotenv import load_dotenv

load_dotenv()

# def send_message(conf):
#     bot_message = f"An accident occurred with a confidence of {conf}%."
#     send_text = "https://api.telegram.org/bot" + os.getenv("BOT_TOKEN") + "/sendMessage?chat_id=" + os.getenv("BOT_CHATID") + "&parse_mode=Markdown&text=" + bot_message
#     requests.get(send_text)

endTime = time.time()

# Initialize video capture
video_path = "D:/Catalyst_Python/yolo_practice/videos/cctv-footage-of-car-accident-on-nizambad-hyderabad-national-highway-720-ytshorts.savetube.me.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Load Keras model
model_path = "D:/Catalyst_Python/yolo_practice/ch-2 yolo with webcam/my_resnet50_video_classifier_accident_dectition.keras"
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error: Unable to load model from {model_path}. Exception: {e}")
    exit()

# Image preprocessing parameters
input_size = (224, 224)  # Adjust based on your model's input size
class_names = ["No Accident", "Accident"]  # Update with your actual class names
tracker = Tracker()
count = 0

while cap.isOpened():
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("Error: Unable to read frame from video")
        break

    # Resize and preprocess the image for the Keras model
    img_resized = cv2.resize(img, input_size)
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize if required

    # Perform inference
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    conf = math.ceil(predictions[0][class_idx] * 100)

    # Display results
    cv2.putText(img, f'{class_names[class_idx]} {conf}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if class_names[class_idx] == "Accident" and conf > 40 and time.time() - endTime > 5:
        send_message(conf)
        endTime = time.time()

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
