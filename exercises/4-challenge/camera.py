import torch
import cv2
import time
from utils.camera import CameraDisplay
import numpy as np

from tinyyolov2 import TinyYoloV2
from utils.yolo import nms, filter_boxes
from utils.viz import display_result
from pruned_tinyyolov2NoBN import PrunedTinyYoloV2NoBN

# make an instance with 20 classes as output
#model = TinyYoloV2(num_classes=20)
model = PrunedTinyYoloV2NoBN(num_classes=1)

# load pretrained weights
sd = torch.load("models/voc_pruned_r002_10_times_19_epochs_lr0001_decay0005.pt")
model.load_state_dict(sd)

#put network in evaluation mode
model.eval()

from typing import List 
# Function to preprocess image for YOLOv2
def preprocess(image):
    # Resize to model input size, normalize, etc.
    image = cv2.resize(image, (416, 416))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change to CHW
    image = torch.from_numpy(image).float().unsqueeze(0)  # Add batch dimension
    return image

# Function to postprocess YOLOv2 output
def postprocess(output, conf_thresh=0.5, iou_thresh=0.4):
    # Implement the postprocessing steps to get bounding boxes
    # This is a simplified placeholder, adjust according to your model's output
    boxes = []
    # Assume output is in the shape of (batch_size, num_boxes, 5+num_classes)
    output = output[0]  # Remove batch dimension
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_thresh:
            box = detection[:4]  # Extract bounding box coordinates
            x, y, w, h = box
            x1 = int((x - w / 2) * 416)
            y1 = int((y - h / 2) * 416)
            x2 = int((x + w / 2) * 416)
            y2 = int((y + h / 2) * 416)
            boxes.append((x1, y1, x2, y2, confidence, class_id))
    return boxes

def display_result_img(image: np.ndarray, output: List[torch.Tensor]) -> np.ndarray:
    ima_shape = image.shape[:2]
    
    if output:
        bboxes = torch.stack(output, dim=0)
        for i in range(bboxes.shape[1]):
            if bboxes[0, i, -1] >= 0:
                cx = int(bboxes[0, i, 0] * ima_shape[1])
                cy = int(bboxes[0, i, 1] * ima_shape[0])
                
                w = int(bboxes[0, i, 2] * ima_shape[1])
                h = int(bboxes[0, i, 3] * ima_shape[0])
                
                cv2.rectangle(image, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (0, 0, 255), 2)
                cv2.putText(image, f"Class {int(bboxes[0, i, 4])}", (cx - w // 2, cy - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
    return image


now = time.time()

def callback(image):
    global now

    fps = f"{int(1/(time.time() - now))}"
    now = time.time()
    # Change to CHW (channels, height, width)
    #image = image.transpose(2, 0, 1)
    #image = image[0:320, 0:320, :]
    
    
    # Preprocess the image
    input_image = preprocess(image)
    
    # Run the model
    #with torch.no_grad():
    #    output = model(input_image)
    
    # Postprocess the output
    #boxes = postprocess(output)
    
    #image = torch.from_numpy(image).float().unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_image)
    output = filter_boxes(output, 0.1)
    output = nms(output, 0.25)
    #boxes = postprocess(output)
    #boxes = output
    #display_result(input, output, target
    
    # Draw bounding boxes
    #for box in boxes:
        # Draw the box and label on the image
        #x1, y1, x2, y2, conf, cls = box
        #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.putText(image, f"{cls}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    image = display_result_img(image, output)
    
    # Draw FPS on the image
    cv2.putText(image, "fps=" + fps, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
    return image

cam = CameraDisplay(callback)

cam.start()