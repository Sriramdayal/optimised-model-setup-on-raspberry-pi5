# Smart Garbage Detection System on Raspberry Pi

This project provides a complete solution for detecting illegal garbage dumping using a Raspberry Pi, a camera, and an optimized YOLOv8 computer vision model. The system can identify dumped trash and trigger alerts in real-time.

## Hardware Requirements

- **Raspberry Pi:** Raspberry Pi 4 Model B (4GB+) or Raspberry Pi 5.
- **Power Supply:** A high-quality USB-C 5V/3A (for Pi 4) or 5V/5A (for Pi 5) power supply.
- **MicroSD Card:** 32GB or larger, Class 10/U3 A2, high-endurance card.
- **Camera Module:** Raspberry Pi Camera Module 2, 3, or HQ Camera.

## Setup Instructions

These steps will guide you through setting up the project from a fresh Raspberry Pi OS installation.

### 1. Prepare the Raspberry Pi

- Flash **Raspberry Pi OS (64-bit)** using the official Raspberry Pi Imager. The 64-bit version is crucial for performance.
- During imaging, pre-configure your user, password, Wi-Fi, and enable SSH.
- Boot the Pi and run a full system update:
  ```bash
  sudo apt update && sudo apt full-upgrade -y
  ```
- Enable the camera interface using `sudo raspi-config` -> `Interface Options` -> `Legacy Camera`.

### 2. Clone the Project Repository

- Open a terminal on your Raspberry Pi.
- Clone the project files into a local directory.
  ```bash
  # Example command, replace with your repository URL if you use Git
  sudo apt install git
  git clone https://github.com/Sriramdayal/optimised-model-setup-on-raspberry-pi5.git
  # cd project/
  
  # If not using Git, simply create a project folder
  mkdir garbage_detector
  cd garbage_detector
  ```

### 3. Set Up the Python Environment

- It is highly recommended to use a virtual environment to manage dependencies.
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- You should now see `(venv)` at the beginning of your terminal prompt.

### 4. Install Dependencies

- Use the `requirements.txt` file to install all necessary Python libraries with a single command.
  ```bash
  pip install -r requirements.txt
  ```

### 5. Add the Optimized Model

- This project requires your custom-trained and optimized YOLOv8 model.
- Place your **`best.onnx`** file into the main project directory.

## Running the System

Make sure your virtual environment is active (`source venv/bin/activate`) before running any scripts.

---
deploying **YOLOv8 ONNX** on a **Raspberry Pi 5** can be very smooth ⚡ if you configure it correctly.
Here’s a **lag-free setup guide** to make inference efficient and real-time (≈ 7–12 FPS with YOLOv8n + 320×320 input).

---

## 🧩 1. Use a Lightweight Model

Pick a smaller variant of YOLOv8 before exporting:

```bash
yolo export model=yolov8n.pt format=onnx dynamic=False opset=12
```

✅ `yolov8n` (nano) or `yolov8s` (small) is ideal — large models (`m`, `l`, `x`) are too heavy for Pi 5 CPU.

---

## ⚙️ 2. Install Optimized ONNX Runtime

On the Pi 5 terminal:

```bash
sudo apt update
sudo apt install -y python3-opencv libatlas-base-dev
pip install onnxruntime
```

If you want **extra speed**, use the **ARM64 OpenBLAS version**:

```bash
pip uninstall onnxruntime -y
pip install onnxruntime==1.18.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## 🧠 3. Optimize the ONNX Model

Use `onnxsim` and `onnxruntime-tools` to simplify:

```bash
pip install onnxsim onnxruntime-tools
python3 -m onnxsim best.onnx best_simplified.onnx
```

Then apply graph optimizations:

```bash
from onnxruntime_tools import optimizer
opt_model = optimizer.optimize_model(
    "best_simplified.onnx", model_type='bert', num_heads=0, hidden_size=0)
opt_model.save_model_to_file("best_optimized.onnx")
```

---

## 🚀 4. Use Half-Precision (FP16) Quantization

For faster CPU math and lower RAM:

```bash
pip install onnxconverter-common
python3 -m onnxconverter_common.float16_converter best_optimized.onnx best_fp16.onnx
```

---

## 🖥️ 5. Efficient Inference Code

```python
import onnxruntime as ort
import cv2, numpy as np, time

# Load ONNX model
session = ort.InferenceSession("best_fp16.onnx", providers=['CPUExecutionProvider'])

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    t1 = time.time()
    img = cv2.resize(frame, (320, 320))
    img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    outputs = session.run([output_name], {input_name: img})
    fps = 1 / (time.time() - t1)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 ONNX", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

⚡ **Expected Performance (Raspberry Pi 5)**

* YOLOv8n (320×320): ~8–12 FPS
* YOLOv8s: ~5 FPS
* CPU temp < 60 °C with passive cooling

---

## 🧩 6. Optional Speed Ups

| Optimization               | Description                                                                                        |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| **Thread count**           | `export OMP_NUM_THREADS=4` (Pi 5 has 4 cores)                                                      |
| **Use OpenCV DNN backend** | `cv2.dnn.readNetFromONNX("best_fp16.onnx")` — sometimes faster                                     |
| **Disable GUI display**    | Write frames to MJPEG stream instead of showing them                                               |
| **Use Vulkan**             | Pi 5 supports Vulkan → can build ONNX Runtime with `--use_vulkan` for GPU acceleration (advanced). |

---

✅ **Recommended combo for lag-free inference**

* Model: `yolov8n`
* Input size: `imgsz=320`
* Quantized: FP16 or INT8
* Optimized ONNX (simplified)
* Threaded ONNX Runtime (4 cores)

---
This version uses:

* ✅ YOLOv8n
* ✅ `imgsz = 320`
* ✅ Quantized `FP16` ONNX
* ✅ Simplified ONNX graph
* ✅ Threaded ONNX Runtime on 4 cores

---

### 🧠 1️⃣ Before you run

Make sure you’ve already done these:

```bash
sudo apt update
sudo apt install -y python3-opencv libatlas-base-dev
pip install onnxruntime onnxsim numpy
export OMP_NUM_THREADS=4
```

If your model isn’t yet optimized:

```bash
pip install onnxsim onnxconverter-common
python3 -m onnxsim yolov8n.onnx yolov8n_simplified.onnx
python3 -m onnxconverter_common.float16_converter yolov8n_simplified.onnx yolov8n_fp16.onnx
```

Then copy `yolov8n_fp16.onnx` to your Raspberry Pi.

---

### 🪶 2️⃣ The full optimized camera script

```python
import cv2
import numpy as np
import onnxruntime as ort
import time

# --- Config ---
MODEL_PATH = "yolov8n_fp16.onnx"
IMG_SIZE = 320
CONF_THRESH = 0.4
IOU_THRESH = 0.45

# --- ONNX Runtime session ---
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

# --- Warmup (improves first-frame latency) ---
dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
session.run(output_names, {input_name: dummy})

# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return img

def draw_boxes(frame, detections):
    h, w, _ = frame.shape
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < CONF_THRESH:
            continue
        x1 = int(x1 / IMG_SIZE * w)
        y1 = int(y1 / IMG_SIZE * h)
        x2 = int(x2 / IMG_SIZE * w)
        y2 = int(y2 / IMG_SIZE * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{int(cls)}:{conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

print("✅ YOLOv8n FP16 ONNX started... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t1 = time.time()
    img = preprocess(frame)

    # Run inference
    preds = session.run(output_names, {input_name: img})[0]
    detections = preds[0]  # Adjust if shape is different

    fps = 1 / (time.time() - t1)
    frame = draw_boxes(frame, detections)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow("YOLOv8n ONNX", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
```

---

### ⚡ Expected performance (Raspberry Pi 5)

| Metric          | YOLOv8n (FP16, 320×320)          |
| --------------- | -------------------------------- |
| FPS (real-time) | **10–14 FPS**                    |
| CPU usage       | ~80–90% on 4 cores               |
| RAM usage       | ~700 MB                          |
| Temperature     | ~55–60 °C (with passive cooling) |

---

### 🔋 Extra tips

* For higher FPS, reduce `imgsz` to `256` or lower `conf` threshold.
* For even lighter model, quantize to INT8 (using `onnxruntime.quantization`).
* Disable `cv2.imshow()` for headless streaming — write frames to Flask or MJPEG server.

---

