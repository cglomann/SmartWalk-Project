print("SCRIPT STARTED")

import csv

# Your manual annotations from watching the video:
# (track_id, start_frame, end_frame, crossed)
annotations = [
    # Pedestrian 9 (crossed at crosswalk: True) — only track_id 9 falls within the pre-crossing window
    (9, 33, 440, True),

    # Pedestrian 80 (crossed at crosswalk: True) — only track_id 80 falls within the pre-crossing window
    (80, 375, 518, True),

    # Pedestrian 111 -> 363 -> 506 (did not cross: False)
    (111, 518, 695, False),
    (363, 945, 1013, False),
    (506, 1059, 1125, False),

    # Pedestrian 618 -> 633 -> 653 (jaywalked, not at crosswalk: False) — track_id 684 excluded, no overlap
    (618, 1647, 1666, False),
    (633, 1711, 1832, False),
    (653, 1863, 1902, False),
]

def get_label(track_id, frame_count):
    for ann_track_id, start, end, crossed in annotations:
        if ann_track_id == track_id and start <= frame_count <= end:
            return 1 if crossed else 0
    return None  # no matching annotation — pedestrian wasn't tracked/labeled

with open("pedestrian_features.csv", "r", newline="") as infile, \
     open("labeled_pedestrian_features.csv", "w", newline="") as outfile:

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