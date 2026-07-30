import cv2
from pose_features import get_landmarks, get_pose_features

img_bgr = cv2.imread("images/man_crossing_street.jpg")

if img_bgr is None:
    print("ERROR: Could not load the image. Check the path.")
else:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    landmarks = get_landmarks(img_rgb)
    result = get_pose_features(img_rgb)
    print(result)