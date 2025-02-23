from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import cvzone
import math
import time
import imgaug.augmenters as iaa

# Load YOLOv8 Model
model = YOLO('../YOLO-weights/yolov8n.pt') 

# Load Video
cap = cv2.VideoCapture("3.mp4")

# Define augmentation (scale by 1.5x)
# augment = iaa.Affine(scale=1.5)  

# Video Writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 1.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1.5)
out = cv2.VideoWriter('augmented_video.mp4', fourcc, fps, (width, height))

# Initialize Deep SORT Tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Class Names
classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck", 
              "traffic light", "fire hydrant", "stop sign", "parking meter"]
target_class_ids = [0, 1, 2, 3, 5, 7, 9, 11, 13, 14]

# Color Map
color_map = {
    "person": (0, 255, 0),
    "bicycle": (255, 255, 0),
    "car": (255, 0, 0),
    "motorbike": (255, 0, 255),
    "bus": (0, 255, 255),
    "truck": (255, 165, 0),
    "traffic light": (0, 0, 255),
    "fire hydrant": (128, 0, 128),
    "stop sign": (0, 128, 128),
    "parking meter": (128, 128, 0)
}

def get_dynamic_count_line(img, position_ratio=0.5):
    """
    Dynamically sets the count line based on frame height.
    
    Args:
        frame: The video frame.
        position_ratio: Position of the line relative to frame height (0 to 1).
                        0.5 places it at the middle, 0.7 near the bottom.

    Returns:
        count_line_position: Integer Y-coordinate for the count line.
    """
    height, _, _ = img.shape
    return int(height * position_ratio)

success, img = cap.read()

# Counting Variables
count_line_position = get_dynamic_count_line(img, position_ratio=0.7)
offset = 15
car_count, bike_count, truck_count, bus_count = 0, 0, 0, 0
counted_ids = set()

# Speed Tracking Variables
object_speeds = {}
previous_positions = {}

# Frame Rate
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    success, img = cap.read()
    if not success:
        break

    # Apply augmentation
    # augmented_frame = augment(image=img)

    # Write and display
    # out.write(augmented_frame)
    # cv2.imshow('Augmented Video', augmented_frame)

    start_time = time.time()

    # Run YOLOv8 Detection
    results = model(img, stream=True)

    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])

            if cls in target_class_ids:
                class_name = classNames[target_class_ids.index(cls)]
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

    # Update Deep SORT Tracker
    tracks = tracker.update_tracks(detections, frame=img)

    # Draw Counting Line
    cv2.line(img, (0, count_line_position), (img.shape[1], count_line_position), (2, 127, 0), 3)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        class_name = track.get_det_class()

        # Draw Bounding Box and Label
        color = color_map.get(class_name, (255, 255, 255))
        # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorC=color, rt=1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        # Center Point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)

        # Speed Calculation
        if track_id not in object_speeds:
            object_speeds[track_id] = {"previous_center": (cx, cy), "speed": 0}
        else:
            prev_cx, prev_cy = object_speeds[track_id]["previous_center"]
            pixel_distance = math.hypot(cx - prev_cx, cy - prev_cy)
            speed = (pixel_distance * fps) * 0.036  # 0.036 converts pixels/frame to km/h (adjust if needed)
            object_speeds[track_id] = {"previous_center": (cx, cy), "speed": round(speed, 2)}


        # Confidence retrieval
        conf = 0
        for det in detections:
            if det[2] == class_name and (x1 <= det[0][0] <= x2):
                conf = det[1]
                break 

        # Get speed for current object
        current_speed = object_speeds[track_id]["speed"]

        # Rectangle with vehicle info 
        info_text = f'{class_name} | Conf: {conf:.2f} | {current_speed} km/h'
        color = color_map.get(class_name, (255, 255, 255))
        # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorC=color, rt=1)
        cvzone.putTextRect(img, info_text, (x1, y1 - 20), scale=1.2, thickness=2, colorR=(0, 0, 0)) 
        
        # Count Objects When Crossing Line
        if count_line_position - offset < cy < count_line_position + offset:
            if track_id not in counted_ids:
                counted_ids.add(track_id)

                if class_name == "car":
                    car_count += 1
                elif class_name in ["bicycle", "motorbike"]:
                    bike_count += 1
                elif class_name == "truck":
                    truck_count += 1
                elif class_name == "bus":
                    bus_count += 1

    # Display Counts
    cv2.putText(img, f'Cars: {car_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Bikes: {bike_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(img, f'Trucks: {truck_count}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
    cv2.putText(img, f'Buses: {bus_count}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show Video
    resized_img = cv2.resize(img, (1000, 800))
    cv2.imshow("YOLOv8 Detection & Speed Tracking", resized_img)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 13]:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


