from collections import defaultdict
import cv2
import numpy as np
import time
import json
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    return cap

def get_video_properties(cap):
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FPS)))

def initialize_video_writer(output_path, fourcc, fps, width, height):
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def load_mask(mask_path, height, width):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[:2] != (height, width):
        raise ValueError("Mask image size must match video frame size.")
    return mask

def main(video_path, mask_path, model_path, output_path):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Open the video file and get properties
    cap = initialize_video_capture(video_path)
    w, h, fps = get_video_properties(cap)
    
    # Load the mask image
    mask = load_mask(mask_path, h, w)
    
    # Initialize video writer
    out = initialize_video_writer(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, w, h)
    
    # Initialize variables
    prev_time = time.time()
    left_to_right_counter = 0
    right_to_left_counter = 0
    last_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        
        annotator = Annotator(frame, line_width=2)
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, conf, class_id, track_id in zip(boxes, confs, class_ids, track_ids):
                label = f"{model.names[int(class_id)]} {conf:.2f} ID: {track_id}"
                annotator.box_label(box, label, color=colors(track_id, True))
                
                bottom_midpoint_x = int((box[0] + box[2]) / 2)
                bottom_midpoint_y = int(box[3])
                bottom_midpoint_x = np.clip(bottom_midpoint_x, 0, w - 1)
                bottom_midpoint_y = np.clip(bottom_midpoint_y, 0, h - 1)
                
                if track_id in last_positions:
                    last_x, last_y, last_side = last_positions[track_id]
                    current_pixel = mask[bottom_midpoint_y, bottom_midpoint_x]
                    current_side = 255 if current_pixel == 255 else 0
                    
                    if last_side == 0 and current_side == 255 and bottom_midpoint_x > last_x:
                        left_to_right_counter += 1
                    elif last_side == 255 and current_side == 0 and bottom_midpoint_x < last_x:
                        right_to_left_counter += 1

                    last_positions[track_id] = (bottom_midpoint_x, bottom_midpoint_y, current_side)
                else:
                    current_pixel = mask[bottom_midpoint_y, bottom_midpoint_x]
                    current_side = 255 if current_pixel == 255 else 0
                    last_positions[track_id] = (bottom_midpoint_x, bottom_midpoint_y, current_side)
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left-to-Right: {left_to_right_counter}", (w - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Right-to-Left: {right_to_left_counter}", (w - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        cv2.imshow("object-detection-tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # read all paths from config.json
    with open("config.json", "r") as f:
        config = json.load(f)
        video_path = config["video_path"]
        mask_path = config["mask_path"]
        model_path = config["model_path"]
        output_path = config["output_path"]

    main(video_path, mask_path, model_path, output_path)