# plot_confidence_comparison.py
import csv
import matplotlib.pyplot as plt

def load_confidence_log(path):
    frames, confidences = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            frames.append(int(row[0]))
            confidences.append(float(row[1]))
    return frames, confidences

v8_frames, v8_conf = load_confidence_log("yolov8_confidence_log.csv")
v11_frames, v11_conf = load_confidence_log("yolov11_confidence_log.csv")
v11s_frames, v11s_conf = load_confidence_log("yolov11s_confidence_log.csv")   
v26_frames, v26_conf = load_confidence_log("yolov26_confidence_log.csv")
v26s_frames, v26s_conf = load_confidence_log("yolov26s_confidence_log.csv")   

plt.figure(figsize=(10, 5))
plt.scatter(v8_frames, v8_conf, s=5, alpha=0.5, label="YOLOv8n")
plt.scatter(v11_frames, v11_conf, s=5, alpha=0.5, label="YOLOv11n")
plt.scatter(v26_frames, v26_conf, s=5, alpha=0.5, label="YOLOv26n")
plt.scatter(v11s_frames, v11s_conf, s=5, alpha=0.5, label="YOLOv11s")   
plt.scatter(v26s_frames, v26s_conf, s=5, alpha=0.5, label="YOLOv26s")
plt.xlabel("Frame Number")
plt.ylabel("Detection Confidence")
plt.title("The Effect of Different YOLO Models on Pedestrian Detection Confidence")
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("confidence_comparison.png")

print(f"YOLOv8n average confidence: {sum(v8_conf)/len(v8_conf):.3f}")
print(f"YOLOv11n average confidence: {sum(v11_conf)/len(v11_conf):.3f}")
print(f"YOLOv11s average confidence: {sum(v11s_conf)/len(v11s_conf):.3f}")
print(f"YOLOv26n average confidence: {sum(v26_conf)/len(v26_conf):.3f}")
print(f"YOLOv26s average confidence: {sum(v26s_conf)/len(v26s_conf):.3f}")

plt.show()