# People Tracking with YOLO and ByteTrack

This project tracks people using the YOLO and ByteTrack algorithms, counting individuals passing a marker and categorizing their direction as right-to-left or left-to-right.

# Demo
Our crossing line is between the pole and the noticeboard bottom.

https://github.com/rushidarge/People-Tracking-and-Counting/assets/39642887/b0d8f22c-dd9a-44fa-a9d9-0cb92b07d74c



## Table of Contents
- [Description](#description)
- [Working](#Working)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Limitation](#Limitation)
- [Bibliography](#Bibliography)

## Description
This project utilizes the YOLO (You Only Look Once) object detection algorithm combined with the ByteTrack multi-object tracking algorithm to monitor and count people passing a specified marker. The direction of movement (right-to-left or left-to-right) is recorded, and counters are incremented accordingly.

## Working:

1. **Read Video**:
    - The code starts by loading a video or connecting to a live camera feed. This is like pressing play on a video player.

2. **Finding People (YOLO)**:
    - The code uses a deep learning model called YOLO (You Only Look Once) to find people in each frame of the video. Think of YOLO as a really smart pair of glasses that can spot people instantly in any picture.

3. **Tracking People (ByteTrack)**:
    - Once YOLO spots a person, ByteTrack takes over to follow that person as they move from one frame to the next. ByteTrack is like a high-tech tracking algorithm that keeps an eye on each person so it knows where they go.

4. **Counting People**:
    - The code has a "marker" or an imaginary line in the video (that it gets from the `mask.png` file). Whenever a person crosses this line, the code notes down which direction they are moving:
        - If they cross from right to left, one counter goes up.
        - If they cross from left to right, another counter goes up.

5. **Displaying Results**:
    - The code continuously updates and shows the counts for how many people have crossed the line in each direction. This is like a scoreboard that keeps track of the movement of people in real time.


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/rushidarge/People-Tracking-and-Counting.git
    cd People-Tracking-and-Counting
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your input video or camera feed.
2. Update your path for video, mask, and out in config.json
3. Run the main tracking script:
    ```bash
    python get_count_dynamic.py
    ```
4. View the results and counters in the console output or as specified in the configuration.
5. Your video is saved in videos/predicition_output/

## Features
- Person detection using YOLO
- Multi-object tracking with ByteTrack
- Direction-based counting
- Easy configuration and customization

## Limitation
- We need to tune our logic sometimes we miss a person from counting.
- If people are overlapping we lose track of them.
- Two people walking simultaneously then we miss that person in counting, we need to place the camera strategically.
- To make it real-time we need GPU.

## Bibliography
Yolo Model : https://github.com/WongKinYiu/yolov9
Bytetrack Algorithm: https://medium.com/tech-blogs-by-nest-digital/object-tracking-object-detection-tracking-using-bytetrack-0aafe924d292
Ultralytics: https://docs.ultralytics.com/modes/track/
