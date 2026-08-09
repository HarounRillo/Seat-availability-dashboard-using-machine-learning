import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO

# Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://seat-18d45-default-rtdb.asia-southeast1.firebasedatabase.app/"
})
seat_ref = db.reference("Seats/Seat1")
print("Firebase connected.")

# Models
seat_model = YOLO("yolo11n.pt")
pose_model = YOLO("yolo11n-pose.pt")

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()
print("Smart Chair Monitor started. Press Q to quit.")

# Classes
SEAT_CLASSES = ["chair", "bench"]
OBJECT_CLASSES = [
    "backpack", "handbag", "suitcase", "remote",
    "book", "laptop", "bottle", "cell phone"
]

# COCO keypoint indices
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
KP_CONF_THRESHOLD = 0.40

# Settings
LOCK_SEAT_AFTER_DETECTION = True
HIP_HORIZONTAL_TOLERANCE = 0.25
HIP_VERTICAL_TOLERANCE = 0.35
SITTING_FRAMES_REQUIRED = 4
OBJECT_OVERLAP_THRESHOLD = 0.20

# State
locked_seat = None
sitting_counter = 0


def get_overlap(box1, box2):
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    return max(0, right - left) * max(0, bottom - top)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def point_inside_expanded_box(point, box, horizontal_ratio, vertical_ratio):
    x, y = point
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (
        x1 - w * horizontal_ratio <= x <= x2 + w * horizontal_ratio
        and y1 - h * vertical_ratio <= y <= y2 + h * vertical_ratio
    )


def is_sitting(kpts_xy, kpts_conf, seat_box):
    needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE]

    if any(kpts_conf[i] < KP_CONF_THRESHOLD for i in needed):
        return False

    left_hip, right_hip = kpts_xy[LEFT_HIP], kpts_xy[RIGHT_HIP]
    left_knee, right_knee = kpts_xy[LEFT_KNEE], kpts_xy[RIGHT_KNEE]
    left_shoulder, right_shoulder = kpts_xy[LEFT_SHOULDER], kpts_xy[RIGHT_SHOULDER]

    hip_mid = midpoint(left_hip, right_hip)
    knee_mid = midpoint(left_knee, right_knee)
    shoulder_mid = midpoint(left_shoulder, right_shoulder)

    # Both hips must be over the seat area
    if not all(
        point_inside_expanded_box(p, seat_box, HIP_HORIZONTAL_TOLERANCE, HIP_VERTICAL_TOLERANCE)
        for p in [left_hip, right_hip, hip_mid]
    ):
        return False

    # Knees must be below hips
    if knee_mid[1] <= hip_mid[1]:
        return False

    # Sitting geometry: hip-to-knee gap should be smaller relative to torso
    torso_length = abs(hip_mid[1] - shoulder_mid[1])
    if torso_length <= 10:
        return False
    if (abs(knee_mid[1] - hip_mid[1]) / torso_length) > 1.15:
        return False

    # Hip should not be dramatically above the chair
    sx1, sy1, sx2, sy2 = seat_box
    if hip_mid[1] < sy1 - (sy2 - sy1) * 0.50:
        return False

    return True


def update_firebase(occupied):
    try:
        seat_ref.update({"occupied": occupied})
    except Exception as e:
        print("Firebase update error:", e)


def draw_person_skeleton(frame, kpts_xy, kpts_conf):
    connections = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_HIP, LEFT_KNEE),
        (RIGHT_HIP, RIGHT_KNEE),
    ]

    for p1, p2 in connections:
        if kpts_conf[p1] >= KP_CONF_THRESHOLD and kpts_conf[p2] >= KP_CONF_THRESHOLD:
            x1, y1 = map(int, kpts_xy[p1])
            x2, y2 = map(int, kpts_xy[p2])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE]:
        if kpts_conf[idx] >= KP_CONF_THRESHOLD:
            x, y = map(int, kpts_xy[idx])
            cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read camera")
        break

    output = frame.copy()

    seat_results = seat_model(frame, verbose=False, conf=0.35)

    detected_seats = []
    objects = []

    for box in seat_results[0].boxes:
        class_name = seat_model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        coords = tuple(map(int, box.xyxy[0]))

        if class_name in SEAT_CLASSES:
            detected_seats.append((coords, confidence, class_name))
        elif class_name in OBJECT_CLASSES:
            objects.append((coords, class_name, confidence))

    if locked_seat is None and detected_seats:
        detected_seats.sort(key=lambda x: x[1], reverse=True)
        locked_seat = detected_seats[0][0]
        print("Chair detected and locked:", locked_seat)

    pose_results = pose_model(frame, verbose=False, conf=0.35)

    people = []
    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
        all_xy = pose_results[0].keypoints.xy.cpu().numpy()
        all_conf = pose_results[0].keypoints.conf
        all_conf = all_conf.cpu().numpy() if all_conf is not None else np.ones(all_xy.shape[:2])
        people = list(zip(all_xy, all_conf))

    status = "NO SEAT DETECTED"
    detected = "None"
    occupied = False

    if locked_seat is not None:
        sx1, sy1, sx2, sy2 = locked_seat

        cv2.rectangle(output, (sx1, sy1), (sx2, sy2), (255, 255, 0), 3)
        cv2.putText(output, "SEAT", (sx1, max(25, sy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        person_sitting = False
        for person_xy, person_conf in people:
            draw_person_skeleton(output, person_xy, person_conf)

            if is_sitting(person_xy, person_conf, locked_seat):
                person_sitting = True
                hip_mid = midpoint(person_xy[LEFT_HIP], person_xy[RIGHT_HIP])
                cv2.putText(output, "SEATED",
                            (int(hip_mid[0]), int(hip_mid[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                break

        sitting_counter = sitting_counter + 1 if person_sitting else 0

        if sitting_counter >= SITTING_FRAMES_REQUIRED:
            status, detected, occupied = "OCCUPIED", "PERSON", True
        else:
            object_detected = False
            for obj_box, obj_name, obj_conf in objects:
                overlap = get_overlap(locked_seat, obj_box)
                obj_area = (obj_box[2] - obj_box[0]) * (obj_box[3] - obj_box[1])

                if obj_area > 0 and (overlap / obj_area) > OBJECT_OVERLAP_THRESHOLD:
                    object_detected = True
                    detected = obj_name.upper()
                    ox1, oy1, ox2, oy2 = obj_box
                    cv2.rectangle(output, (ox1, oy1), (ox2, oy2), (0, 165, 255), 2)
                    cv2.putText(output, detected, (ox1, max(25, oy1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    break

            status = "OBJECT DETECTED" if object_detected else "EMPTY"
            if not object_detected:
                detected = "None"

    update_firebase(occupied)

    color_map = {
        "OCCUPIED": (0, 0, 255),
        "OBJECT DETECTED": (0, 165, 255),
        "EMPTY": (0, 255, 0),
    }
    text_color = color_map.get(status, (0, 0, 255))

    cv2.putText(output, "STATUS: " + status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)
    cv2.putText(output, "DETECTED: " + detected, (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    cv2.putText(output, "CHAIR LOCKED", (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Smart Chair Monitor", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()