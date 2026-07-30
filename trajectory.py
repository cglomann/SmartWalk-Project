import math 

def predict_future_position(position, speed, direction, k_frames, time_change):
    """Constant velocity prediction, k_frames into the future."""
    x, y = position
    vx = speed * math.cos(direction)
    vy = speed * math.sin(direction)
    future_x = x + k_frames * time_change * vx
    future_y = y + k_frames * time_change * vy
    return(future_x, future_y)
