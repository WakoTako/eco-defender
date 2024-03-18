import os
import cv2
from ultralytics import YOLO
from twilio.rest import Client

# Twilio credentials
account_sid = ""
auth_token = ""

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Directory containing the test images
TEST_IMAGES_DIR = r"D:\before_training\input_images\lol"
# Directory to save the output images with bounding boxes
OUTPUT_DIR = r'D:\before_training\output_images'

# Load the trained YOLO model
model_path = r"D:\before_training\runs\detect\train2\weights\best.pt"
model = YOLO(model_path)

# Set threshold for object detection confidence
threshold = 0.2

# Map class IDs to animal names
class_names = {0: "Boar", 1: "Elephant", 2: "Tiger"}  # Add more as needed

# Iterate through the test images
for filename in os.listdir(TEST_IMAGES_DIR):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        image_path = os.path.join(TEST_IMAGES_DIR, filename)
        image = cv2.imread(image_path)

        # Perform object detection
        results = model(image)

        # Initialize animal detected flag
        animal_detected = False

        # Iterate through detected objects
        for bbox, class_id, conf in zip(results[0].boxes.xyxy.tolist(), results[0].boxes.cls.tolist(), results[0].boxes.conf.tolist()):
            x1, y1, x2, y2 = bbox
            if conf > threshold:
                animal_detected = True
                animal_name = class_names[class_id]
                break  # Stop iterating if an animal is detected

        # Send notification if an animal is detected
        if animal_detected:
            # Send SMS message
            message = client.messages.create(
                body=f"A {animal_name} has been detected on your property.",
                from_="+16203373067",
                to="+918590682110"
            )
            print("Message sent:", message.sid)

            # Draw bounding box and save output image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, animal_name, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            output_image_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_image_path, image)

        else:
            print("No animals detected in", filename)

print("Testing completed.")
