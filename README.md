# Driver Monitoring System

## Overview
The **Driver Monitoring System** is an AI-powered surveillance system designed to enhance road safety by detecting driver behaviors and passenger activities in public transport vehicles. This system currently focuses on identifying:
- **Smoking Detection**: Determines if the driver is smoking.
- **Aisle Monitoring**: Detects if passengers are standing in the aisle.
- **Yawning Detection**: Identifies if the driver is yawning, which may indicate fatigue.
- **Mobile Usage Detection**: Detects if the driver is holding or using a mobile phone.

This is the initial version of the system, and future iterations may include additional features.

## Features
- Real-time monitoring using trained AI models
- Detection using advanced deep learning models
- Custom training on specialized datasets for high accuracy
- Modular and scalable architecture for easy feature additions

## Tech Stack
The system is built using various deep learning and computer vision frameworks, including:

### **Model Training**
- **YOLO** (You Only Look Once) - for object detection
- **Ultralytics** - implementation of YOLO models
- **Torch** - deep learning framework for training models
- **Roboflow** - for dataset preprocessing and augmentation

### **Processing & Inference**
- **OpenCV** - for image and video processing
- **Supervision** - for managing inference workflows
- **Hugging Face** - for leveraging pre-trained models and model deployment

### **Standard Libraries**
- **NumPy** - for numerical computations
- **Matplotlib** - for data visualization
- **PIL** (Pillow) - for image processing
- **Time** - for time-based operations
- **Etc.** - for other standard libraries and utilities

## Installation
To set up the Driver Monitoring System, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/driver-monitoring-system.git
   cd driver-monitoring-system
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python roboflow torch torchvision numpy matplotlib pillow supervision transformers
   ```

3. Download the pre-trained model and place it in the designated directory.

4. Run inference on test data:
   ```python
   from ultralytics import YOLO
   model = YOLO("trained_model.pt")  # Load pre-trained model
   results = model.predict(source="test_video.mp4", save=True)
   ```

## Future Enhancements
- **Drowsiness Detection**: Combining yawning with eye-tracking for fatigue detection.
- **Gesture Recognition**: Detecting distracted driving behaviors beyond mobile usage.
- **Emergency Alert System**: Notifying authorities in case of critical detections.

## Contribution
Feel free to contribute to the project by:
- Improving detection accuracy
- Adding new features
- Enhancing model efficiency

## License
This project is licensed under the MIT License.


