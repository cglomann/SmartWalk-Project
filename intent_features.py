ORIENTATION_MAP = {
    "facing_camera": 0.0,
    "facing_left": 1.0,
    "facing_right": -1.0,
    "facing_away": 2.0 
}

def build_feature_vector(track_id, pedestrian_history, frame_index):
    """
    Builds one X_t feature vector for a given pedestrian at a given
    point in their stored history (frame_index = index into the deques).
    """
    history = pedestrian_history[track_id]

    x, y = history["positions"][frame_index]
    v = history["speeds"][frame_index]
    a = history["accelerations"][frame_index]
    theta = history["directions"][frame_index]
    d_curb = history["curb_distances"][frame_index]

    pose = history["pose_features"][frame_index]
    if pose is not None:
        o = ORIENTATION_MAP.get(pose["body_orientation"], 0.0)
        h = pose["head_direction"]
    else:
        o = 0.0
        h = 0.0

    # Placeholders until crosswalk zones (Step 4) are implemented
    d_crosswalk = 0.0
    z = 0.0

    return [x, y, v, a, theta, d_curb, d_crosswalk, z, o, h]

