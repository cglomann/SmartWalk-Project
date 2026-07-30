from ultralytics import YOLO
import cv2
import math
from pose_features import get_landmarks, get_pose_features
from collections import deque
from intent_features import build_feature_vector
from intent_model import PedestrianIntentLSTM
import torch
import time
import csv
import os
from vehicle_features import init_vehicle_history, update_vehicle_history
from trajectory import predict_future_position
from ttc import calculate_ttc, calculate_pedestrian_arrival, check_conflict
from risk_score import calculate_risk_score, normalize_vehicle_risk
from alert_logic import determine_alert_level, get_alert_output
import csv as csv_module 

# Create dictionary to story vehicle history data
vehicle_history = {}

script_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolo26s.pt") # pedestrian detection
vehicle_model = YOLO("yolo26s.pt") # vehicle detection
crosswalk_model = YOLO("yolov8n_crosswalk.pt") # crosswalk shape detection

import numpy as np

feature_mean = np.load(os.path.join(script_dir, "feature_mean.npy"))    
feature_std = np.load(os.path.join(script_dir, "feature_std.npy"))      

# Use trained PedestrianIntentLSTM Model
intent_model = PedestrianIntentLSTM(input_size=10, hidden_size=64)
intent_model.load_state_dict(torch.load(os.path.join(script_dir, "pedestrian_intent_lstm.pth")))  
intent_model.eval()

# Define video feed being used
SEQUENCE_LENGTH = 20
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture("videos/pedestrians.mp4")
#video = cv2.VideoCapture("videos/testing_video.mp4")
#video = cv2.VideoCapture("videos/training_video2.mov")
fps = video.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30 # assumption in case of division by zero
time_change = 1 / fps

# Record video
output_fps = fps  # use the SAME fps as your source video, not your processing speed
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # code for .mp4 output
video_writer = cv2.VideoWriter(
    os.path.join(script_dir, "demo_output.mp4"),
    fourcc,
    output_fps,
    (frame_width, frame_height)
)

# Create dictionary to store pedestrian history data
pedestrian_history = {}

# Create crosswalk polygons list
crosswalk_polygons = []

STATIONARY_SPEED_THRESHOLD = 5  # pixels/sec — tune based on video
HISTORY_LENGTH = 30 # keep only the most recent 30 values per pedestrian ID recorded

frame_count = 0
cv2.namedWindow("SmartWalk Pedestrian Detection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Driver Alert Display", cv2.WINDOW_NORMAL)

# Create CSV file to record data (IDs, frame numbers, distance, orientation, head direction, crossing probability, risk score, alert level)
#csv_file = open("pedestrian_features.csv", mode="w", newline="")
csv_file = open("pedestrian_features_test.csv", mode="w", newline="")
print("CSV file created at:", os.path.abspath("pedestrian_features_test.csv"))
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "track_id", "frame_number", "x", "y", "v", "a", "theta",
    "d_curb", "d_crosswalk", "zone", "orientation", "head_direction", 
    "crossing_probability", "risk_score", "alert_level"
])

# Manually define polygon of curb 
#CURB_POINT_A = (76, 659)   # left green dot
#CURB_POINT_B = (566, 572)   # red dot (the corner)
#CURB_POINT_C = (N1333, 556)   # right green dot

#CURB_POINT_A = (130, 731)
#CURB_POINT_B = (1696, 749)

#CONFLICT_POINT = (
    #(CURB_POINT_A[0] + CURB_POINT_B[0]) // 2,
    #(CURB_POINT_A[1] + CURB_POINT_B[1]) // 2
#)

def point_to_segment_distance(point, seg_a, seg_b):
    """Function to calculate the shortest distance from a point to a line segment"""
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    seg_len_sq = abx**2 + aby**2
    if seg_len_sq == 0:
        return math.sqrt(apx**2 + apy**2)
    t = max(0, min(1, (apx * abx + apy * aby) / seg_len_sq)) # projects point onto the line that passes through points A and B, and figures out where on that line the closest point falls
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def get_crosswalk_polygons(frame, conf=0): 
    """Function that runs the results of the crosswalk model on a frame and returns the polygons of the detected crosswalks"""
    results = crosswalk_model.predict(frame, conf=conf, verbose=False)
    polygons = []
    for r in results:
        if r.masks is not None:
            for mask_xy in r.masks.xy:  # list of (N, 2) arrays, one per detected crosswalk
                polygons.append(mask_xy.astype(np.int32)) # Concerts coordinates into whole number integers for OpenCV drawing functions
    return polygons

visible_alert_level = 0 # default alert level 
visible_alert_expiration = 0

# BGR format (OpenCV uses BGR, not RGB)
ALERT_COLORS = {
    0: (200, 200, 200),  # no warning - gray
    1: (0, 255, 255),    # Level 1 - yellow
    2: (0, 165, 255),    # Level 2 - orange
    3: (0, 0, 255),      # Level 3 - red
}

def build_driver_display(alert_level, width=640, height=400):
    """Builds a simple driver-facing alert screen as its own image."""
    display = np.full((height, width, 3), 30, dtype=np.uint8)  # dark background

    color = ALERT_COLORS[alert_level]
    messages = {
        0: "ROAD CLEAR",
        1: "PEDESTRIAN NEARBY",
        2: "CAUTION - PEDESTRIAN MAY CROSS",
        3: "STOP - PEDESTRIAN CROSSING"
    }

    # Large colored circle, like a traffic signal light
    cv2.circle(display, (width // 2, height // 2 - 40), 100, color, -1)

    # Message text, centered-ish
    text = messages[alert_level]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 60

    cv2.putText(display, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return display

def build_split_screen(annotated_frame, driver_display, target_height=480, cam_width=854, disp_width=640):
    # Resize camera frame to target height, preserving aspect ratio
    cam_h, cam_w = annotated_frame.shape[:2]
    cam_scale = target_height / cam_h
    cam_resized = cv2.resize(annotated_frame, (cam_width, target_height))

    # Resize driver display to the same target height
    disp_h, disp_w = driver_display.shape[:2]
    disp_scale = target_height / disp_h
    disp_resized = cv2.resize(driver_display, (disp_width, target_height))

    # Stack side by side (horizontal concatenation)
    combined = cv2.hconcat([cam_resized, disp_resized])
    return combined

confidence_log = []  # will store (frame_number, confidence) pairs

while video.isOpened():
    success, frame = video.read()
    frame_count = frame_count + 1

    # Once enough frames have passed since the last alert, reset the alert level to the default (grey) unless a new alert arised before the expiration reached
    if frame_count > visible_alert_expiration:
        visible_alert_level = 0

    if not success:
        break

    # Using ByteTrack to track each individual object detected in the frame
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0] # Only track pedestrians (class 0 in COCO dataset)
    )

    # Display pedestrian IDs
    boxes = results[0].boxes

    # Draw the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    #cv2.imshow("SmartWalk Pedestrian Detection", annotated_frame)

    # VEHICLE TRACKING AND DETECTION
    vehicle_results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck (COCO classes)
        conf=0.4,
        iou=0.5
    )
    vehicle_boxes = vehicle_results[0].boxes

    # Draw vehicle boxes on top of the pedestrian-annotated frame, then display
    annotated_frame = vehicle_results[0].plot(img=annotated_frame)
    if len(crosswalk_polygons) == 0:
        polygons = get_crosswalk_polygons(frame, conf=0.1)  # run the crosswalk model once at startup
        if 'CONFLICT_POINT' not in dir() and not crosswalk_polygons:
            pass  # skip risk scoring this frame — no conflict point defined yet
        if polygons:
            crosswalk_polygons = polygons  # only update if there is a new detection of this frame
            # if no new detection, crosswalk_polygons retains its previous value
    cv2.polylines(annotated_frame, crosswalk_polygons, isClosed=True, color=(0, 255, 0), thickness=2) # draws detecte crosswalk outline as green lines onto the frame
    #cv2.imshow("SmartWalk Pedestrian Detection", annotated_frame)

    if vehicle_boxes.id is not None:
        for v_box, v_track_id in zip(vehicle_boxes, vehicle_boxes.id):
            v_track_id = int(v_track_id)
            vx1, vy1, vx2, vy2 = v_box.xyxy[0]
            vehicle_center = (int((vx1 + vx2) / 2), int((vy1 + vy2) / 2))
            update_vehicle_history(vehicle_history, v_track_id, vehicle_center, time_change)

    # Extract coordinates of each detected pedestrian and print them
    if boxes.id is not None:
        for box, track_id in zip(boxes, boxes.id):
            track_id = int(track_id)
            print(f"\nPedestrian ID: {int(track_id)}")

            x1, y1, x2, y2 = box.xyxy[0]

            # Inside your pedestrian loop, right after you get `box`:
            confidence = float(box.conf[0])  # YOLO's confidence score for this detection
            confidence_log.append((frame_count, confidence))

        
            foot_point = (
                int((x1 + x2) / 2),
                int(y2)
            )
            
            print()
            print(f"\n*****FRAME COUNT*****\n")
            print(f"{frame_count}")
            # time.sleep(0.2)
            print()
            print(f"ID: {int(track_id)}")
            print(f"Box: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
            print(f"Foot Point: {foot_point}")

        # Create dictionary to store pedestrian history
            if track_id not in pedestrian_history:
                pedestrian_history[int(track_id)] = {
                    "positions": deque(maxlen=HISTORY_LENGTH),
                    "speeds": deque(maxlen=HISTORY_LENGTH),
                    "directions": deque(maxlen=HISTORY_LENGTH),
                    "accelerations": deque(maxlen=HISTORY_LENGTH),
                    "curb_distances": deque(maxlen=HISTORY_LENGTH),
                    "pose_features": deque(maxlen=HISTORY_LENGTH),
                    "crossing_probabilities": deque(maxlen=HISTORY_LENGTH)
                    }   

            # Store pedestrian positions
            pedestrian_history[int(track_id)]["positions"].append(foot_point)
            print(
                f"ID {track_id} has "
                f"{len(pedestrian_history[int(track_id)]['positions'])} positions stored."
            )

            # Calculate speed 
            if len(pedestrian_history[int(track_id)]["positions"]) > 1:
                current_position = pedestrian_history[int(track_id)]["positions"][-1]
                previous_position = pedestrian_history[int(track_id)]["positions"][-2]
                pedestrian_history[int(track_id)]["speeds"].append((math.sqrt((current_position[0] - previous_position[0])**2 + (current_position[1] - previous_position[1])**2))/time_change)
            else:
                pedestrian_history[int(track_id)]["speeds"].append(0)

            print(
                f"Pedestrian ID {track_id} has "
                f"{len(pedestrian_history[int(track_id)]['speeds'])} speed values stored."
            )

            print(
                f"Pedestrian ID {track_id}'s speed values (in pixels per second): {pedestrian_history[int(track_id)]['speeds']}"
            )

            # Calculate direction
            if len(pedestrian_history[int(track_id)]["positions"]) > 1:
                current_position = pedestrian_history[int(track_id)]["positions"][-1]
                previous_position = pedestrian_history[int(track_id)]["positions"][-2]
                dx = current_position[0] - previous_position[0]
                dy = current_position[1] - previous_position[1]
                direction = math.atan2(dy, dx)
                pedestrian_history[int(track_id)]["directions"].append(direction)
            else:
                pedestrian_history[int(track_id)]["directions"].append(0)

            print(
                f"Pedestrian ID {track_id} has "
                f"{len(pedestrian_history[int(track_id)]['directions'])} direction values stored."
            )

            print(
                f"Pedestrian ID {track_id}'s direction values (in radians): {pedestrian_history[int(track_id)]['directions']}"
            )

            # Calculate acceleration
            if len(pedestrian_history[int(track_id)]["speeds"]) > 1:
                current_speed = pedestrian_history[int(track_id)]["speeds"][-1]
                previous_speed = pedestrian_history[int(track_id)]["speeds"][-2]
                acceleration = (current_speed - previous_speed) / time_change
                pedestrian_history[int(track_id)]["accelerations"].append(acceleration)
            else:
                pedestrian_history[int(track_id)]["accelerations"].append(0)

            print(
                f"Pedestrian ID {track_id} has "
                f"{len(pedestrian_history[int(track_id)]['accelerations'])} acceleration values stored."
            )

            print(
                f"Pedestrian ID {track_id}'s acceleration values (in pixels per second squared): {pedestrian_history[int(track_id)]['accelerations']}"
            )

            # Curb distance calculation - manual fixed points
            #curb_dist_1 = point_to_segment_distance(foot_point, CURB_POINT_A, CURB_POINT_B)
            #curb_dist_2 = point_to_segment_distance(foot_point, CURB_POINT_B, CURB_POINT_C)
            #curb_dist = min(curb_dist_1, curb_dist_2)

            # Curb distance calculation - polygon based
            min_distance = float("inf")
            for polygon in crosswalk_polygons:
                for i in range(len(polygon)):
                    CURB_POINT_A = polygon[i]
                    CURB_POINT_B = polygon[(i+1) % len(polygon)]
                    distance = point_to_segment_distance(foot_point, CURB_POINT_A, CURB_POINT_B)
                    if distance < min_distance:
                        min_distance = distance
                        CONFLICT_POINT = [((CURB_POINT_A[0] + CURB_POINT_B[0]) // 2), ((CURB_POINT_A[1] + CURB_POINT_B[1]) // 2)]

            pedestrian_history[track_id]["curb_distances"].append(min_distance)

            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            person_crop_bgr = frame[y1i:y2i, x1i:x2i]

            if person_crop_bgr.size > 0:
                person_crop_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
                landmarks = get_landmarks(person_crop_rgb)
            else:
                landmarks = None

            if landmarks is not None:
                pose_result = get_pose_features(landmarks)

                recent_speed = pedestrian_history[track_id]["speeds"][-1] if pedestrian_history[track_id]["speeds"] else 0
                leg_separation = pose_result["leg_separation"]

                if recent_speed < STATIONARY_SPEED_THRESHOLD and leg_separation is not None and leg_separation < 1.0:
                    posture = "standing"
                elif recent_speed >= STATIONARY_SPEED_THRESHOLD or (leg_separation is not None and leg_separation >= 1.0):
                    posture = "walking"
                else:
                    posture = "unknown"

                pose_result["posture"] = posture
            else:
                pose_result = None

            pedestrian_history[track_id]["pose_features"].append(pose_result)
            print(f"Pose Features: {pedestrian_history[track_id]['pose_features'][-1]}")

            # crossing-intention prediction
            history = pedestrian_history[track_id]

            current_vector = build_feature_vector(track_id, pedestrian_history, -1)  # -1 = this frame, just appended
            output_row = [track_id, frame_count] + current_vector
            if len(history["positions"]) >= SEQUENCE_LENGTH:
                sequence = [
                    build_feature_vector(track_id, pedestrian_history, i)
                    for i in range(-SEQUENCE_LENGTH, 0)
                ]

                sequence = ((np.array(sequence) - feature_mean) / feature_std).tolist() 
                input_tensor = torch.tensor([sequence], dtype=torch.float32)

                with torch.no_grad():
                    crossing_probability = intent_model(input_tensor).item()

                output_row.append(crossing_probability)

                pedestrian_history[track_id]["crossing_probabilities"].append(crossing_probability)
                print(f"Pedestrian ID {track_id} crossing probability: {crossing_probability:.2f}")
            else:
                pedestrian_history[track_id]["crossing_probabilities"].append(None)

            # Vehicle conflict + risk scoring
            if len(history["positions"]) >= SEQUENCE_LENGTH and vehicle_history:
                ped_position = history["positions"][-1]
                ped_speed = history["speeds"][-1] if history["speeds"] else 0

                highest_risk = 0.0
                path_conflict = False

                for v_id, v_hist in vehicle_history.items():
                    print(f"DEBUG: vehicle {v_id} has {len(v_hist['positions'])} positions, latest speed = {v_hist['speeds'][-1] if v_hist['speeds'] else 'N/A'}")
                    if not v_hist["positions"] or v_hist["speeds"][-1] == 0:
                        continue
                    v_position = v_hist["positions"][-1]
                    v_speed = v_hist["speeds"][-1]

                    ttc = calculate_ttc(v_position, v_speed, CONFLICT_POINT)
                    t_ped = calculate_pedestrian_arrival(ped_position, ped_speed, CONFLICT_POINT)
                    conflict = check_conflict(ttc, t_ped, delta=1.5)

                    if conflict:
                        path_conflict = True

                    v_risk = normalize_vehicle_risk(v_speed)
                    score = calculate_risk_score(
                        p_cross=crossing_probability,
                        p_path_conflict=1.0 if conflict else 0.0,
                        s_vehicle=v_risk
                    )
                    highest_risk = max(highest_risk, score)

                output_row.append(highest_risk)
                #print(f"DEBUG: track_id {track_id} highest_risk = {highest_risk:.3f}")  

                alert_level, confirmed_alert_level = determine_alert_level(track_id, crossing_probability, highest_risk)
                alert_message, alert_sound = get_alert_output(confirmed_alert_level, True)

                display_level = confirmed_alert_level # only shows real alert level on screen (if 10 consecutive risky frames have been recorded)
                # display_level = alert_level if confirmed else 0 # only shows real alert level on screen (if 10 consecutive risky frames have been recorded)

                if display_level > visible_alert_level:
                    visible_alert_level = display_level
                    visible_alert_expiration = frame_count + 30

                color = ALERT_COLORS[display_level]

                # Print alert to screen
                text = f"ID {track_id}: {alert_message}"
                text_position = (int(x1), int(y1) - 10)  # just above the pedestrian's bounding box

                cv2.putText(
                    annotated_frame,
                    text,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,        # font scale
                    color,
                    2,          # thickness
                    cv2.LINE_AA
                )

                output_row.append(alert_level)

                print(f"Pedestrian ID {track_id} risk score: {highest_risk:.2f} | Alert: {alert_message}")

            # current_vector = build_feature_vector(track_id, pedestrian_history, -1)  # -1 = this frame, just appended
            # prob_to_log = crossing_probability if len(history["positions"]) >= SEQUENCE_LENGTH else None
            # csv_writer.writerow([track_id, frame_count] + current_vector + [prob_to_log])
            csv_writer.writerow(output_row)
    
    circle_color = ALERT_COLORS[visible_alert_level]
    cv2.circle(annotated_frame, (60, 60), 30, circle_color, -1)

    video_writer.write(annotated_frame)
    cv2.imshow("SmartWalk Pedestrian Detection", annotated_frame)

    driver_display = build_driver_display(visible_alert_level)  
    cv2.imshow("Driver Alert Display", driver_display)

    split_screen_width = 854 + 640   # = 1494
    split_screen_height = 480

    split_writer = cv2.VideoWriter(
        os.path.join(script_dir, "demo_split_screen.mp4"),
        fourcc,
        output_fps,
        (split_screen_width, split_screen_height)
    )

    # NEW: build and save the combined split-screen frame
    split_frame = build_split_screen(annotated_frame, driver_display, target_height=480, cam_width=854, disp_width=640)
    split_writer.write(split_frame)
    cv2.imshow("Split Screen Preview", split_frame)   # optional — lets you watch the combined result live too

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

with open(os.path.join(script_dir, "yolo_confidence_log.csv"), "w", newline="") as f:
    writer = csv_module.writer(f)
    writer.writerow(["frame", "confidence"])
    writer.writerows(confidence_log)

video.release()
video_writer.release()
split_writer.release()
cv2.destroyAllWindows()
csv_file.close()

with open(os.path.join(script_dir, "yolov26s_confidence_log.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "confidence"])
    writer.writerows(confidence_log)

print(f"Confidence log saved with {len(confidence_log)} entries.")