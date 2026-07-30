import csv

OFFSET = 100000  # ensures video 2's track_ids never collide with video 1's

with open("labeled_pedestrian_features.csv", "r", newline="") as f1, \
     open("labeled_pedestrian_features_2.csv", "r", newline="") as f2, \
     open("labeled_pedestrian_features_combined.csv", "w", newline="") as out:

    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    writer = csv.writer(out)

    header = next(reader1)
    next(reader2)  # skip second file's header
    writer.writerow(header)

    row_count = 0
    for row in reader1:
        writer.writerow(row)
        row_count += 1

    for row in reader2:
        row[0] = str(int(row[0]) + OFFSET)  # track_id is column 0
        writer.writerow(row)
        row_count += 1

print(f"Combined CSV created: labeled_pedestrian_features_combined.csv ({row_count} total rows)")