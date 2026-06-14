# 👁️5ENSE: Real-Time Spatial Navigation & Obstacle Avoidance

This demo is a computer vision script that uses a webcam and YOLOv8 object detection to analyze a user's surroundings in real-time. It divides the camera feed into three spatial zones (Left, Center, Right), calculates obstacle density, and determines the safest path forward. 

The system also simulates Galvanic Vestibular Stimulation (GVS) outputs (in milliamperes), conceptualizing a wearable interface that could physically guide a visually impaired user or an autonomous system around everyday obstacles.

## ✨Features

* **Real-Time Object Detection:** Utilizes Ultralytics YOLOv8 to detect common environmental obstacles (people, furniture, vehicles, etc.).
* **Spatial Zoning:** Divides the camera frame into Left, Center, and Right zones to calculate where obstacles are clustered.
* **Intelligent Pathfinding:** Automatically recommends a safe path (`LEFT`, `RIGHT`, `CENTER`, or `STOP`) based on zone blockage thresholds.
* **Simulated GVS Feedback:** Generates simulated electrical current readouts (mA) designed for Left/Right directional guidance.
* **Custom HUD (Heads-Up Display):** Features an on-screen overlay displaying safe paths, detected objects, zone scores, and bounding boxes.

## 🛠️Prerequisites

To run this project, you will need **Python 3.7+** and a connected webcam.

### 📦 Dependencies

Install the required Python libraries using `pip`:

```bash
pip install opencv-python ultralytics
```

## 🚀 Usage
Save the code to a Python file, for example, 5ense.py.

Run the script from your terminal:
```bash
python 5ense.py
```
A window titled "5ENSE" will open, displaying your mirrored webcam feed with the detection overlay. 📹

Press ESC to close the window and terminate the program.

## ⚙️ Configuration
You can easily adjust the sensitivity and behavior of the system by modifying the constants at the top of the script:

* MIN_CONFIDENCE (Default: 0.40): The minimum confidence score required for YOLO to register an object. Increase this to reduce false positives.

* MIN_AREA (Default: 25000): The minimum bounding box area (in pixels) an object must have to be considered a valid obstacle. Prevents distant/tiny objects from triggering navigational changes.

* BLOCKED_THRESHOLD (Default: 60000): The cumulative area of objects required to flag a specific zone as "blocked".

* IMPORTANT_OBJECTS: A list of class labels the system actively looks out for. You can add or remove objects (e.g., "dog", "cat", "stop sign") based on the standard COCO dataset classes. 
