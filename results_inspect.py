import os
import cv2
from ultralytics import YOLO
import time

# Directory containing the test images
TEST_IMAGES_DIR = r'D:\before_training\input_images\new_new'
# Directory to save the output images with bounding boxes
OUTPUT_DIR = r'D:\before_training\output_images'

# Load the trained YOLO model
model_path = r"D:\before_training\runs\detect\train2\weights\best.pt"
model = YOLO(model_path)

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set threshold for object detection confidence
threshold = 0.2

# Iterate through the test images
for filename in os.listdir(TEST_IMAGES_DIR):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        image_path = os.path.join(TEST_IMAGES_DIR, filename)
        image = cv2.imread(image_path)

        # Perform object detection
        results = model(image)

        # Draw bounding boxes around detected objects
        for bbox in results[0].boxes.xyxy:

            boxes = results[0].boxes.xyxy.cpu().tolist()
            class_id = results[0].boxes.cls.cpu().tolist()
            conf_=results[0].boxes.conf.cpu().tolist()
            # x1, y1, x2, y2 = boxes[0]
            # print("bbox=",bbox)
            # print("boxes=",boxes[0])
            # time.sleep(3)
            for i in range(len(boxes)):
                # print("i=",i)
                x1, y1, x2, y2 = boxes[i]
                if conf_[0] > threshold:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, model.names[int(class_id[i])], (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)


            # print(conf_)
            #print("boxes=",boxes)
            #print("clss=",clss)


            #x1, y1, x2, y2, conf, class_id = bbox

        # Save the output image with bounding boxes
        output_image_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_image_path, image)

print("Testing completed.")

