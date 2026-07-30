import csv

track_frame_ranges = {}
with open("pedestrian_features_2.csv", "r", newline="") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        track_id = int(row[0])
        frame_number = int(row[1])
        if track_id not in track_frame_ranges:
            track_frame_ranges[track_id] = [frame_number, frame_number]
        else:
            track_frame_ranges[track_id][0] = min(track_frame_ranges[track_id][0], frame_number)
            track_frame_ranges[track_id][1] = max(track_frame_ranges[track_id][1], frame_number)

for track_id, (min_f, max_f) in sorted(track_frame_ranges.items()):
    print(f"track_id {track_id}: frames {min_f} to {max_f}")