# Time-to-Collision (TTC) Calculation
conflict_point = 1000 # fixed point I define

def distance_to_point(p1, p2):
    return((p1[0] - p2[0]**2) + (p1[1] - p2[1])**2) ** 0.5

def calculate_ttc(vehicle_position, vehicle_speed, conflict_point):
    if vehicle_speed <= 0:
        return float("inf")
    
    d_vehicle = distance_to_point(vehicle_position, conflict_point)
    return d_vehicle / vehicle_speed

def calculate_pedestrian_arrival(pedestrian_position, pedestrian_speed, conflict_point):
    if pedestrian_speed <= 0:
        return float("inf")
    d_pedestrian = distance_to_point(pedestrian_position, conflict_point)
    return d_pedestrian / pedestrian_speed

def check_conflict(ttc, t_pedestrian, delta=1.5):
    if ttc == float("inf") or t_pedestrian == float("inf"):
        return False
    return abs(ttc - t_pedestrian) < delta
