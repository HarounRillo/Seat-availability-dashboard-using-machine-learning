import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO


# ============================================================
# FIREBASE INITIALIZATION
# ============================================================

cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://seat-18d45-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

seat_ref = db.reference("Seats/Seat1")

print("Firebase connected.")


# ============================================================
# YOLO MODELS
# ============================================================

seat_model = YOLO("yolo11n.pt")
pose_model = YOLO("yolo11n-pose.pt")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

print("Smart Chair Monitor started.")
print("Press Q to quit.")


# ============================================================
# CLASSES
# ============================================================

SEAT_CLASSES = ["chair", "bench"]

OBJECT_CLASSES = [
    "backpack",
    "handbag",
    "suitcase",
    "remote",
    "book",
    "laptop",
    "bottle",
    "cell phone"
]


# ============================================================
# COCO KEYPOINT INDICES
# ============================================================

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

KP_CONF_THRESHOLD = 0.5


# ============================================================
# FUNCTIONS
# ============================================================

def get_overlap(box1, box2):

    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])

    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])

    width = max(0, right - left)
    height = max(0, bottom - top)

    return width * height


def midpoint(p1, p2):

    return (
        (p1[0] + p2[0]) / 2,
        (p1[1] + p2[1]) / 2
    )


def is_sitting(kpts_xy, kpts_conf, seat_box):

    needed = [
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_SHOULDER,
        RIGHT_SHOULDER
    ]

    for idx in needed:

        if kpts_conf[idx] < KP_CONF_THRESHOLD:
            return False

    hip_mid = midpoint(
        kpts_xy[LEFT_HIP],
        kpts_xy[RIGHT_HIP]
    )

    knee_mid = midpoint(
        kpts_xy[LEFT_KNEE],
        kpts_xy[RIGHT_KNEE]
    )

    shoulder_mid = midpoint(
        kpts_xy[LEFT_SHOULDER],
        kpts_xy[RIGHT_SHOULDER]
    )

    sx1, sy1, sx2, sy2 = seat_box

    # Check if the person's hip is actually over the seat
    hip_over_seat = (
        sx1 <= hip_mid[0] <= sx2
        and
        sy1 <= hip_mid[1] <= sy2
    )

    if not hip_over_seat:
        return False

    # Check sitting posture
    torso_length = abs(
        hip_mid[1] - shoulder_mid[1]
    )

    hip_knee_gap = (
        knee_mid[1] - hip_mid[1]
    )

    if torso_length <= 0:
        return False

    bent_ratio = hip_knee_gap / torso_length

    return bent_ratio < 0.5


def update_firebase(occupied):

    try:

        seat_ref.update({
            "occupied": occupied
        })

    except Exception as e:

        print("Firebase update error:", e)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera")
        break


    # ========================================================
    # RUN YOLO MODELS
    # ========================================================

    seat_results = seat_model(
        frame,
        verbose=False
    )

    pose_results = pose_model(
        frame,
        verbose=False
    )

    output = seat_results[0].plot()


    # ========================================================
    # FIND SEATS AND OBJECTS
    # ========================================================

    seats = []
    objects = []

    for box in seat_results[0].boxes:

        class_id = int(box.cls[0])

        class_name = seat_model.names[class_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        current_box = (
            x1,
            y1,
            x2,
            y2
        )

        if class_name in SEAT_CLASSES:

            seats.append(current_box)

        elif class_name in OBJECT_CLASSES:

            objects.append(
                (
                    current_box,
                    class_name
                )
            )


    # ========================================================
    # FIND PEOPLE
    # ========================================================

    people = []

    if (
        pose_results[0].keypoints is not None
        and
        len(pose_results[0].keypoints) > 0
    ):

        all_xy = (
            pose_results[0]
            .keypoints
            .xy
            .cpu()
            .numpy()
        )

        all_conf = pose_results[0].keypoints.conf

        if all_conf is not None:

            all_conf = (
                all_conf
                .cpu()
                .numpy()
            )

        else:

            all_conf = np.ones(
                all_xy.shape[:2]
            )

        for person_xy, person_conf in zip(
            all_xy,
            all_conf
        ):

            people.append(
                (
                    person_xy,
                    person_conf
                )
            )


    # ========================================================
    # DEFAULT STATUS
    # ========================================================

    status = "NO SEAT DETECTED"
    detected = "None"

    occupied = False


    # ========================================================
    # PROCESS FIRST DETECTED SEAT
    # ========================================================

    if len(seats) > 0:

        seat = seats[0]

        sx1, sy1, sx2, sy2 = seat


        # ----------------------------------------------------
        # Draw seat
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (sx1, sy1),
            (sx2, sy2),
            (255, 255, 0),
            2
        )

        cv2.putText(
            output,
            "SEAT",
            (sx1, sy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Check for sitting person
        # ----------------------------------------------------

        person_sitting = False

        for person_xy, person_conf in people:

            if is_sitting(
                person_xy,
                person_conf,
                seat
            ):

                person_sitting = True
                break


        # ====================================================
        # PERSON SITTING
        # ====================================================

        if person_sitting:

            status = "OCCUPIED"
            detected = "PERSON"

            occupied = True


        # ====================================================
        # NO PERSON — CHECK OBJECT
        # ====================================================

        else:

            object_detected = False

            for obj_box, obj_name in objects:

                overlap = get_overlap(
                    seat,
                    obj_box
                )

                obj_width = (
                    obj_box[2] -
                    obj_box[0]
                )

                obj_height = (
                    obj_box[3] -
                    obj_box[1]
                )

                obj_area = (
                    obj_width *
                    obj_height
                )

                if (
                    obj_area > 0
                    and
                    (overlap / obj_area) > 0.20
                ):

                    object_detected = True

                    detected = obj_name.upper()

                    break


            if object_detected:

                status = "OBJECT DETECTED"

            else:

                status = "EMPTY"
                detected = "None"


    # ========================================================
    # UPDATE FIREBASE
    # ========================================================

    update_firebase(occupied)


    # ========================================================
    # STATUS COLOR
    # ========================================================

    if status == "OCCUPIED":

        text_color = (
            0,
            0,
            255
        )

    elif status == "OBJECT DETECTED":

        text_color = (
            0,
            165,
            255
        )

    elif status == "EMPTY":

        text_color = (
            0,
            255,
            0
        )

    else:

        text_color = (
            0,
            0,
            255
        )


    # ========================================================
    # DISPLAY STATUS
    # ========================================================

    cv2.putText(
        output,
        "STATUS: " + status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        text_color,
        3
    )

    cv2.putText(
        output,
        "DETECTED: " + detected,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Smart Chair Monitor",
        output
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()