import cv2
import numpy as np
import csv
from ultralytics import YOLO

# Settings
TOTAL_SAMPLES = 15
SITTING_FRAMES_REQUIRED = 4
KP_CONF_THRESHOLD = 0.40

HIP_HORIZONTAL_TOLERANCE = 0.25
HIP_VERTICAL_TOLERANCE = 0.35

OBJECT_OVERLAP_THRESHOLD = 0.20

RESULT_FILE = "evaluation_results.csv"

# Classes
SEAT_CLASSES = ["chair", "bench"]

OBJECT_CLASSES = [
    "backpack",
    "handbag",
    "suitcase",
    "remote",
    "book",
    "laptop",
    "bottle",
    "cell phone",
]

# COCO keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

# Load models
print()
print("Loading YOLO models...")

seat_model = YOLO("yolo11n.pt")
pose_model = YOLO("yolo11n-pose.pt")

print("Models loaded.")
print()

# Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()


def get_overlap(box1, box2):
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])

    width = max(0, right - left)
    height = max(0, bottom - top)

    return width * height


def midpoint(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def point_inside_expanded_box(point, box, horizontal_ratio, vertical_ratio):
    x, y = point
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    expanded_x1 = x1 - width * horizontal_ratio
    expanded_x2 = x2 + width * horizontal_ratio

    expanded_y1 = y1 - height * vertical_ratio
    expanded_y2 = y2 + height * vertical_ratio

    return expanded_x1 <= x <= expanded_x2 and expanded_y1 <= y <= expanded_y2


def is_sitting(kpts_xy, kpts_conf, seat_box):
    needed = [
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
    ]

    for idx in needed:
        if kpts_conf[idx] < KP_CONF_THRESHOLD:
            return False

    left_hip = kpts_xy[LEFT_HIP]
    right_hip = kpts_xy[RIGHT_HIP]

    left_knee = kpts_xy[LEFT_KNEE]
    right_knee = kpts_xy[RIGHT_KNEE]

    left_shoulder = kpts_xy[LEFT_SHOULDER]
    right_shoulder = kpts_xy[RIGHT_SHOULDER]

    hip_mid = midpoint(left_hip, right_hip)
    knee_mid = midpoint(left_knee, right_knee)
    shoulder_mid = midpoint(left_shoulder, right_shoulder)

    # Both hips should be around the chair
    left_hip_on_seat = point_inside_expanded_box(
        left_hip, seat_box, HIP_HORIZONTAL_TOLERANCE, HIP_VERTICAL_TOLERANCE
    )
    right_hip_on_seat = point_inside_expanded_box(
        right_hip, seat_box, HIP_HORIZONTAL_TOLERANCE, HIP_VERTICAL_TOLERANCE
    )
    hip_mid_on_seat = point_inside_expanded_box(
        hip_mid, seat_box, HIP_HORIZONTAL_TOLERANCE, HIP_VERTICAL_TOLERANCE
    )

    if not (left_hip_on_seat and right_hip_on_seat and hip_mid_on_seat):
        return False

    # Knees should be below hips
    if knee_mid[1] <= hip_mid[1]:
        return False

    # Torso / leg geometry
    torso_length = abs(hip_mid[1] - shoulder_mid[1])
    hip_knee_gap = abs(knee_mid[1] - hip_mid[1])

    if torso_length <= 10:
        return False

    bent_ratio = hip_knee_gap / torso_length

    if bent_ratio > 1.15:
        return False

    # Hip should not be far above chair
    sx1, sy1, sx2, sy2 = seat_box
    seat_height = sy2 - sy1

    if hip_mid[1] < sy1 - seat_height * 0.50:
        return False

    return True


def calculate_metrics(actual, predicted):
    classes = ["EMPTY", "OCCUPIED", "OBJECT"]

    total = len(actual)
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    accuracy = correct / total if total > 0 else 0

    results = {}

    for cls in classes:
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for a, p in zip(actual, predicted):
            if a == cls and p == cls:
                true_positive += 1
            elif a != cls and p == cls:
                false_positive += 1
            elif a == cls and p != cls:
                false_negative += 1

        if (true_positive + false_positive) > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if (true_positive + false_negative) > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        results[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Macro F1
    overall_f1 = sum(results[cls]["f1"] for cls in classes) / len(classes)

    return accuracy, overall_f1, results, correct


def detect_frame(frame, locked_seat):
    seat_results = seat_model(frame, verbose=False, conf=0.35)

    detected_seats = []
    objects = []

    for box in seat_results[0].boxes:
        class_id = int(box.cls[0])
        class_name = seat_model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        current_box = (x1, y1, x2, y2)

        if class_name in SEAT_CLASSES:
            detected_seats.append((current_box, confidence))
        elif class_name in OBJECT_CLASSES:
            objects.append((current_box, class_name, confidence))

    # Lock chair if not already locked
    if locked_seat is None and len(detected_seats) > 0:
        detected_seats.sort(key=lambda x: x[1], reverse=True)
        locked_seat = detected_seats[0][0]

    if locked_seat is None:
        return "EMPTY", locked_seat, frame

    seat = locked_seat

    # Pose detection
    pose_results = pose_model(frame, verbose=False, conf=0.35)

    people = []

    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
        all_xy = pose_results[0].keypoints.xy.cpu().numpy()
        all_conf = pose_results[0].keypoints.conf

        if all_conf is not None:
            all_conf = all_conf.cpu().numpy()
        else:
            all_conf = np.ones(all_xy.shape[:2])

        for person_xy, person_conf in zip(all_xy, all_conf):
            people.append((person_xy, person_conf))

    # Check occupied
    person_sitting = False

    for person_xy, person_conf in people:
        if is_sitting(person_xy, person_conf, seat):
            person_sitting = True
            break

    if person_sitting:
        return "OCCUPIED", locked_seat, frame

    # Check object
    for obj_box, obj_name, obj_conf in objects:
        overlap = get_overlap(seat, obj_box)

        obj_width = obj_box[2] - obj_box[0]
        obj_height = obj_box[3] - obj_box[1]
        obj_area = obj_width * obj_height

        if obj_area > 0 and (overlap / obj_area) > OBJECT_OVERLAP_THRESHOLD:
            return "OBJECT", locked_seat, frame

    return "EMPTY", locked_seat, frame


# Start
print("=" * 50)
print("SMART SEAT LIVE EVALUATION")
print("=" * 50)

print()
print("You will collect 15 samples.")
print()
print("Controls:")
print("  E = Actual EMPTY")
print("  O = Actual OCCUPIED")
print("  B = Actual OBJECT")
print("  Q = Quit")
print()
print("Recommended:")
print("  5 Empty")
print("  5 Occupied")
print("  5 Object")
print()
print("Wait for the detector to stabilize before labeling.")
print()

input("Press ENTER to start...")

# Data
actual_labels = []
predicted_labels = []

locked_seat = None
sitting_counter = 0

# CSV file
csv_file = open(RESULT_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Sample", "Actual", "Predicted", "Correct"])

# Main evaluation loop
while len(actual_labels) < TOTAL_SAMPLES:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    prediction, locked_seat, output = detect_frame(frame, locked_seat)

    # Draw fixed seat
    if locked_seat is not None:
        sx1, sy1, sx2, sy2 = locked_seat

        cv2.rectangle(output, (sx1, sy1), (sx2, sy2), (255, 255, 0), 3)
        cv2.putText(
            output, "SEAT", (sx1, max(25, sy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

    # Prediction display
    cv2.putText(
        output, "PREDICTION: " + prediction, (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3
    )

    cv2.putText(
        output,
        "Sample: " + str(len(actual_labels)) + " / " + str(TOTAL_SAMPLES),
        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    cv2.putText(
        output, "E=Empty  O=Occupied  B=Object  Q=Quit", (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )

    cv2.imshow("Smart Seat Evaluation", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print()
        print("Evaluation stopped by user.")
        break

    actual = None

    if key == ord("e"):
        actual = "EMPTY"
    elif key == ord("o"):
        actual = "OCCUPIED"
    elif key == ord("b"):
        actual = "OBJECT"

    if actual is not None:
        sample_number = len(actual_labels) + 1
        correct = actual == prediction

        actual_labels.append(actual)
        predicted_labels.append(prediction)

        csv_writer.writerow([sample_number, actual, prediction, correct])
        csv_file.flush()

        print()
        print("-" * 50)
        print("Sample:", sample_number, "/", TOTAL_SAMPLES)
        print("Actual:", actual)
        print("Predicted:", prediction)

        if correct:
            print("Result: CORRECT")
        else:
            print("Result: INCORRECT")

        correct_count = sum(
            a == p for a, p in zip(actual_labels, predicted_labels)
        )
        current_accuracy = (correct_count / len(actual_labels)) * 100

        print("Current Accuracy:", f"{current_accuracy:.2f}%")
        print("-" * 50)

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()

# Final results
if len(actual_labels) == 0:
    print()
    print("No samples were collected.")
    exit()

accuracy, overall_f1, results, correct = calculate_metrics(
    actual_labels, predicted_labels
)

print()
print()
print("=" * 50)
print("SMART SEAT MODEL EVALUATION")
print("=" * 50)

print()
print("Total samples:", len(actual_labels))
print("Correct:", correct)
print()

print(f"{'Class':<15}{'Precision':>12}{'Recall':>12}{'F1':>10}")
print("-" * 50)

for cls in ["EMPTY", "OCCUPIED", "OBJECT"]:
    precision = results[cls]["precision"]
    recall = results[cls]["recall"]
    f1 = results[cls]["f1"]

    print(f"{cls:<15}{precision:>12.2f}{recall:>12.2f}{f1:>10.2f}")

print("-" * 50)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print(f"Overall F1 Score: {overall_f1 * 100:.2f}%")
print("-" * 50)

print()
print("Results saved to:", RESULT_FILE)
print()
print("Evaluation complete.")