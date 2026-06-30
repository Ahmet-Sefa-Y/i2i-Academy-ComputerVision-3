# Real-Time Finger Counter

This project was completed as part of the i2i Academy Computer Vision homework.

## Objective

The application uses a webcam to detect a human hand, identify hand landmarks, determine whether each finger is open or closed, and display the live number of open fingers on the video feed.

## Technologies Used

* Python 3.12
* OpenCV
* MediaPipe Hands

## Features

* Captures live frames from the default webcam.
* Detects one hand and its landmark points.
* Uses finger-joint logic to count open fingers.
* Displays hand landmarks and connections on the video frame.
* Shows the live open-finger count as text.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python finger_counter.py
```

## Controls

Press `Q` or `Esc` to close the application.
