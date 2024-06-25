from collections import defaultdict
import cv2
import numpy as np
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Load the YOLO model for object detection with tracking capabilities
model = YOLO("../../models/yolov9t.pt")  # Use a detection model

# Open the video file
cap = cv2.VideoCapture('../../videos/test_videos/2.mp4')

# Retrieve video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Load the mask image
mask = cv2.imread('../../videos/test_videos/mask/mask2_1.png', cv2.IMREAD_GRAYSCALE)
if mask is None or mask.shape[:2] != (h, w):
    raise ValueError("Mask image size must match video frame size.")

# Initialize video writer to save the output video with the specified properties
out = cv2.VideoWriter("../../videos/predicition_output/video_2_demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# Initialize variables for FPS calculation
prev_time = time.time()

# Counters for people crossing the line
left_to_right_counter = 0
right_to_left_counter = 0

# Dictionary to store the last known positions of track IDs
last_positions = {}

while True:
    # Read a frame from the video
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Create an annotator object to draw on the frame
    annotator = Annotator(im0, line_width=2)

    # Perform object tracking on the current frame
    results = model.track(im0, persist=True)

    # Check if bounding boxes and IDs are present in the results
    if results[0].boxes.id is not None:
        # Extract bounding boxes, class IDs, and tracking IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Annotate each bounding box with its corresponding tracking ID and color
        for box, conf, class_id, track_id in zip(boxes, confs, class_ids, track_ids):
            label = f"{model.names[int(class_id)]} {conf:.2f} ID: {track_id}"
            annotator.box_label(box, label, color=colors(track_id, True))

            # Calculate the midpoint of the bottom edge of the bounding box
            bottom_midpoint_x = int((box[0] + box[2]) / 2)
            bottom_midpoint_y = int(box[3])

            # Ensure the midpoint is within the bounds of the mask image
            bottom_midpoint_x = np.clip(bottom_midpoint_x, 0, w - 1)
            bottom_midpoint_y = np.clip(bottom_midpoint_y, 0, h - 1)

            # Check if the person has crossed the line using the mask
            if track_id in last_positions:
                last_x, last_y, last_side = last_positions[track_id]
                current_pixel = mask[bottom_midpoint_y, bottom_midpoint_x]

                # Determine current side of the line (0 for black, 255 for white)
                current_side = 255 if current_pixel == 255 else 0

                # Check for left-to-right crossing
                if last_side == 0 and current_side == 255 and bottom_midpoint_x > last_x:
                    left_to_right_counter += 1
                # Check for right-to-left crossing
                elif last_side == 255 and current_side == 0 and bottom_midpoint_x < last_x:
                    right_to_left_counter += 1

                # Update the last position of the track ID
                last_positions[track_id] = (bottom_midpoint_x, bottom_midpoint_y, current_side)
            else:
                # Initialize the last position for new track IDs
                current_pixel = mask[bottom_midpoint_y, bottom_midpoint_x]
                current_side = 255 if current_pixel == 255 else 0
                last_positions[track_id] = (bottom_midpoint_x, bottom_midpoint_y, current_side)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Annotate FPS on the frame
    # cv2.putText(im0, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Annotate crossing counters on the frame
    cv2.putText(im0, f"Left-to-Right: {left_to_right_counter}", (w - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(im0, f"Right-to-Left: {right_to_left_counter}", (w - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Write the annotated frame to the output video
    out.write(im0)
    # Display the annotated frame
    cv2.imshow("object-detection-tracking", im0)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video writer and capture objects, and close all OpenCV windows
out.release()
cap.release()
cv2.destroyAllWindows()
