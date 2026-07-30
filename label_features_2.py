print("SCRIPT STARTED")

import csv

# Fill this in once you've watched the second video and noted track_id/frame ranges
annotations = [
    # Pedestrian 13 -> 101 (did cross: True)
    (13, 150, 790, True),
    (101, 412, 443, True),
    # Pedestrian 23 -> 15 (did cross: True)
    (15, 158, 176, True),
    (23, 182, 497, True),
    # Pedestrian 60 (did not cross: False)
    (60, 270, 336, False),
    # Pedestrian 73 (did cross: True)
    (73, 318, 352, True),
    # Pedestrian 134 -> 97 -> 178 (did cross: True)
    (97, 381, 533, True),  
    (134, 547, 689, True),
    (178, 703, 790, True),
]

def get_label(track_id, frame_count):
    for ann_track_id, start, end, crossed in annotations:
        if ann_track_id == track_id and start <= frame_count <= end:
            return 1 if crossed else 0
    return None

with open("pedestrian_features_2.csv", "r", newline="") as infile, \
     open("labeled_pedestrian_features_2.csv", "w", newline="") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    writer.writerow(header + ["label"])

    row_count = 0
    labeled_count = 0

    for row in reader:
        track_id = int(row[0])
        frame_count = int(row[1])
        label = get_label(track_id, frame_count)
        writer.writerow(row + [label])
        row_count += 1
        if label is not None:
            labeled_count += 1

print(f"Processed {row_count} rows. {labeled_count} rows received a label.")