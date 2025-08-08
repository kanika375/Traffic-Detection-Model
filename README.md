--------------------YOLOv8 Vehicle Detection and Speed Tracking--------------------------------

Overview:
Exploring real-time detection with YOLOv8 and OpenCV.
This project uses YOLOv8 for real-time object detection combined with the Deep SORT tracking algorithm to detect and track vehicles (such as cars, trucks, buses, and bicycles) in a video. It counts the number of vehicles passing a predefined line and calculates their speeds. 

Features
1. Real-time vehicle detection and tracking using YOLOv8 and Deep SORT.
2. Object speed calculation in km/h.
3. Dynamic vehicle count based on the frame height.
4. Displays vehicle type, detection confidence, and speed on the video feed.

Repository Structure
/YOLO-weights
    yolov8n.pt                  # Pretrained YOLOv8 model weights
/tracker.py                     # Main script for vehicle detection, tracking, and speed calculation
/1.mp4                          # Input videos
/2.mp4
/3.mp4

Requirements
1. Python 3.x
2. Libraries:
   ultralytics
   deep_sort_realtime
   opencv-python
   cvzone
   imgaug
   math

You can install the required dependencies using:
pip install ultralytics deep_sort_realtime opencv-python cvzone imgaug

How to Run the Code
1. Ensure you have the YOLOv8 model weights yolov8n.pt in the ../YOLO-weights/ directory. You can download the weights from the YOLOv8 GitHub page.
2. Place your input video (e.g., 3.mp4) in the same directory as the script.
3. Run the script using the following command:
python tracker.py
4. The script will:
   Load the video.
   Perform object detection on each frame.
   Track vehicles and calculate their speeds.
   Display the vehicle counts and tracking information.
5. To exit the video display, press q or Enter.

Key Code Components
1. YOLOv8 Model Setup
   The YOLOv8 model is loaded from the specified path to detect objects in each video frame.
   model = YOLO('../YOLO-weights/yolov8n.pt')
2. Video Processing & Augmentation
3. Object Detection & Deep SORT Tracker
   YOLOv8 detects the objects, and the Deep SORT tracker tracks these detections across frames.
4. Speed Calculation
   For each detected and tracked object, the script calculates the speed based on pixel movement over time, converting it to km/h.
   speed = (pixel_distance * fps) * 0.036
5. Vehicle Counting & Line Crossing Detection
   The script counts objects that cross a predefined line on the screen (positioned dynamically based on the frame height).
6. Output & Display
   The video with tracking, speed, and vehicle count information is saved and displayed.
   cv2.imshow("YOLOv8 Detection & Speed Tracking", resized_img)

License
This project is open-source and available for personal use and modification. Please refer to the license file for more information
