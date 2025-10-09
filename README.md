# Smart Garbage Detection System on Raspberry Pi

This project provides a complete solution for detecting illegal garbage dumping using a Raspberry Pi, a camera, and an optimized YOLOv8 computer vision model. The system can identify dumped trash and trigger alerts in real-time.

## Hardware Requirements

- **Raspberry Pi:** Raspberry Pi 4 Model B (4GB+) or Raspberry Pi 5.
- **Power Supply:** A high-quality USB-C 5V/3A (for Pi 4) or 5V/5A (for Pi 5) power supply.
- **MicroSD Card:** 32GB or larger, Class 10/U3 A2, high-endurance card.
- **Camera Module:** Raspberry Pi Camera Module 2, 3, or HQ Camera.
- **(For Deployment):** A PIR Motion Sensor (e.g., HC-SR501).

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
  # git clone [https://your-repository-url.com/project.git](https://your-repository-url.com/project.git)
  # cd project/
  
  # If not using Git, simply create a project folder
  mkdir gmc_garbage_detector
  cd gmc_garbage_detector
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
- Place your **`best.onnx`** file into the main project directory (e.g., inside `gmc_garbage_detector/`).

## Running the System

Make sure your virtual environment is active (`source venv/bin/activate`) before running any scripts.

### Running the Presentation Demo

This script is for live demonstrations. It shows a camera feed and uses a keyboard to trigger detection.

- **Controls:** Press `d` to detect trash, `q` to quit.
- **Command:**
  ```bash
  python presentation_demo_optimized.py
  ```

### Running the Field Deployment Script

This script is for real-world use. It uses a PIR motion sensor to automatically trigger detection.

- **Prerequisite:** Ensure your PIR sensor is correctly wired to the GPIO pins.
- **Command:**
  ```bash
  python detector.py
  ```

---