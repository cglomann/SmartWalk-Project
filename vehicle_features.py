import math
from collections import deque

VEHICLE_HISTORY_LENGTH = 30

def init_vehicle_history():
    return {
        "positions": deque(maxlen=VEHICLE_HISTORY_LENGTH),
        "speeds": deque(maxlen=VEHICLE_HISTORY_LENGTH),
        "directions": deque(maxlen=VEHICLE_HISTORY_LENGTH)
    }

def update_vehicle_history(vehicle_history, track_id, center_point, time_change):
    if track_id not in vehicle_history:
        vehicle_history[track_id] = init_vehicle_history()

    history = vehicle_history[track_id]
    history["positions"].append(center_point)

    if len(history["positions"]) > 1:
        p1 = history["positions"][-1]
        p0 = history["positions"][-2]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        speed = math.sqrt(dx**2 + dy**2) / time_change
        direction = math.atan2(dy, dx)
    else:
        speed = 0
        direction = 0

    history["speeds"].append(speed)
    history["directions"].append(direction)
    return history
