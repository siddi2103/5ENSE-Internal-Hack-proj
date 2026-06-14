import cv2
from ultralytics import YOLO


MIN_CONFIDENCE = 0.40
MIN_AREA = 25000
BLOCKED_THRESHOLD = 60000  # Adjust this: area needed to consider a zone "blocked"

IMPORTANT_OBJECTS = [
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "bottle",
    "backpack",
    "bicycle",
    "car",
    "motorcycle"
]


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)

print("Press ESC to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    #mirror that shi
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    left_zone = 0
    center_zone = 0
    right_zone = 0

    detected_objects = []
    valid_detections = []

    results = model(frame, verbose=False)

    for result in results:

        for box in result.boxes:

            conf = float(box.conf[0])

            if conf < MIN_CONFIDENCE:
                continue

            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in IMPORTANT_OBJECTS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2 - x1) * (y2 - y1)

            
            if area < MIN_AREA:
                continue

            
            valid_detections.append({
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": area
            })


    valid_detections.sort(key=lambda x: x["area"], reverse=True)

    for i, det in enumerate(valid_detections):
        
        label = det["label"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        area = det["area"]

        if label not in detected_objects:
            detected_objects.append(label)

        color = (0, 0, 255) if i == 0 else (0, 255, 0)
        thickness = 3 if i == 0 else 2

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness
        )

        if x1 < w / 3:
            left_zone += area

        if x2 > w / 3 and x1 < 2 * w / 3:
            center_zone += area

        if x2 > 2 * w / 3:
            right_zone += area

    if left_zone > BLOCKED_THRESHOLD and center_zone > BLOCKED_THRESHOLD and right_zone > BLOCKED_THRESHOLD:
        safest = "STOP"
    elif center_zone == 0:
        safest = "CENTER"
    else:
        safest = "LEFT" if left_zone <= right_zone else "RIGHT"

    if safest == "STOP":
        action = "ALL BLOCKED - STOP"
        arrow = "[ X ]"
        current_text = "L:+2.0mA  R:+2.0mA"  # Strong bilateral stop signal
        path_color = (0, 0, 255)  # Red 
        
    elif safest == "LEFT":
        action = "SAFE PATH LEFT"
        arrow = "<<<"
        current_text = "L:+1.5mA  R:0.2mA"
        path_color = (0, 255, 0)  # Green 
        
    elif safest == "RIGHT":
        action = "SAFE PATH RIGHT"
        arrow = ">>>"
        current_text = "L:0.2mA  R:+1.5mA"
        path_color = (0, 255, 0)

    else:
        action = "SAFE PATH FORWARD"
        arrow = "^^^"
        current_text = "L:_mA  R:_mA"
        path_color = (0, 255, 0)

    
    cv2.line(frame, (w // 3, 0), (w // 3, h), (255, 0, 0), 2)
    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (255, 0, 0), 2)

    cv2.putText(
        frame,
        "LEFT",
        (50, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "CENTER",
        (w // 2 - 60, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "RIGHT",
        (w - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

  

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (330, 440),
        (30, 30, 30),
        -1
    )

    alpha = 0.60
    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (330, 440),
        (0, 255, 255),
        1
    )

  

    cv2.putText(frame, "5ENSE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Safe Path Info (Color changes based on STOP vs GO)
    cv2.putText(frame, "SAFE PATH", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, safest, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.1, path_color, 2)

    # Arrow & Action
    cv2.putText(frame, arrow, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.5, path_color, 3)
    cv2.putText(frame, action, (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # GVS Simulation
    cv2.putText(frame, "SIMULATED GVS OUTPUT", (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, current_text, (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    # Detected Objects
    detected_text = ", ".join(detected_objects)
    if not detected_text:
        detected_text = "None"
    
    # Truncate text if it gets too long for the box
    if len(detected_text) > 28:
        detected_text = detected_text[:25] + "..."

    cv2.putText(frame, "DETECTED OBJECTS", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, detected_text, (20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)

    # Zone Areas
    cv2.putText(frame, "ZONE SCORES", (20, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(
        frame,
        f"L:{left_zone}  C:{center_zone}  R:{right_zone}",
        (20, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    # =====================================
    # DISPLAY
    # =====================================

    cv2.imshow(
        "5ENSE",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()