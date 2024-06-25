# People Tracking with YOLO and ByteTrack

This project tracks people using the YOLO and ByteTrack algorithms, counting individuals passing a marker and categorizing their direction as either right-to-left or left-to-right.

# Demo
https://github.com/rushidarge/People-Tracking-and-Counting/assets/39642887/b0d8f22c-dd9a-44fa-a9d9-0cb92b07d74c



## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Description
This project utilizes the YOLO (You Only Look Once) object detection algorithm combined with the ByteTrack multi-object tracking algorithm to monitor and count people passing a specified marker. The direction of movement (right-to-left or left-to-right) is recorded, and counters are incremented accordingly.

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

