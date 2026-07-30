import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options = base_options,
    running_mode = vision.RunningMode.IMAGE # feeds individual cropped frames
)

landmarker = vision.PoseLandmarker.create_from_options(options)

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

def get_landmarks(person_crop_rgb):
    """Returns raw MediaPipe landmarks for one cropped pedestrian or None"""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=person_crop_rgb)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]

def get_foot_direction(landmarks):
    """Returns direction of foot"""
    left_heel, left_toe = landmarks[LEFT_HEEL], landmarks[LEFT_FOOT_INDEX]
    right_heel, right_toe = landmarks[RIGHT_HEEL], landmarks[RIGHT_FOOT_INDEX]

    left_angle = math.atan2(left_toe.y - left_heel.y, left_toe.x - left_heel.x)
    right_angle = math.atan2(right_toe.y - right_heel.y, right_toe.x - right_heel.x)

    # Average using vector sum
    avg_x = math.cos(left_angle) + math.cos(right_angle)
    avg_y = math.sin(left_angle) + math.sin(right_angle)
    foot_direction = math.atan2(avg_y, avg_x)

    return foot_direction

def get_head_direction(landmarks):
    """Returns direction of head"""
    left_ear, right_ear = landmarks[LEFT_EAR], landmarks[RIGHT_EAR]
    # Position - turned toward left ear side, negative - turned toward right ear side
    head_turn = right_ear.x - left_ear.x
    return head_turn

def get_body_orientation(landmarks):
    """Returns body orientation based on shoulder and nose visibility"""
    ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
    nose = landmarks[NOSE]

    shoulder_depth_difference = rs.z - ls.z 

    if nose.visibility < 0.5:
        orientation = "facing_away"
    elif abs(shoulder_depth_difference) < 0.1:
        orientation = "facing_camera"
    elif shoulder_depth_difference > 0:
        orientation = "facing_left"
    else:
        orientation = "facing_right"

    return orientation

def get_leg_separation(landmarks):
    """Returns distance between ankles and hip width"""
    left_ankle, right_ankle = landmarks[LEFT_ANKLE], landmarks[RIGHT_ANKLE]
    left_hip, right_hip = landmarks[LEFT_HIP], landmarks[RIGHT_HIP]

    ankle_distance = math.sqrt((left_ankle.x - right_ankle.x)**2 + (left_ankle.y - right_ankle.y)**2)
    hip_width = math.sqrt((left_hip.x - right_hip.x)**2 + (left_hip.y - right_hip.y)**2)
    
    # Normalize by hip width so this works regardless of how close/far the person is from camera
    if hip_width == 0:
        return None
    
    return ankle_distance / hip_width

def get_pose_features(landmarks):
    """
    person_crop_rgb: cropped image of one pedestrian in RGB format
    Take landmarks (NOT an image), returns the feature dict.
    Returns dictionary of pose features or none if no pose is detected
    """
    ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
    lh, rh = landmarks[LEFT_HIP], landmarks[RIGHT_HIP]

    # Shoulder angle relative to horizontal
    shoulder_angle = math.atan2(rs.y - ls.y, rs.x - ls.x)
    hip_angle = math.atan2(rh.y - lh.y, rh.x - lh.x)

    return { 
        "shoulder_angle": shoulder_angle,
        "hip_angle": hip_angle,
        "foot_direction": get_foot_direction(landmarks),
        "head_direction": get_head_direction(landmarks),
        "body_orientation": get_body_orientation(landmarks),
        "leg_separation": get_leg_separation(landmarks)        
    }


