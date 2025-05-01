# Driver Monitoring System

## Objective
This system is designed to enhance road safety by monitoring driver behavior and passenger actions in real-time. The system detects **driver distractions** (e.g., mobile phone usage) and **unsafe passenger behavior** (e.g., passengers standing in the aisle when the bus is in motion).

## Features
- **Driver Distraction Detection**: Detects whether the driver is using a mobile phone using a pre-trained model from **Roboflow**.
- **Smoking Detection**: Uses a custom-trained CNN model to detect smoking behavior in the driver's cabin.
- **Yawning Detection**: Monitors the driver’s facial expressions to detect yawning, which can indicate fatigue.
- **Aisle Standing Detection**: Detects if passengers are standing in the aisle of the bus when the vehicle is in motion, promoting safety and compliance.

## Tools & Technologies
- **Programming Languages**: Python, C++
- **Libraries & Frameworks**:
  - **OpenCV**: For image processing and real-time video feed analysis.
  - **TensorFlow/PyTorch**: For training custom deep learning models.
  - **Flask**: For deploying the system as a web application for real-time inference.
  - **Roboflow**: For the mobile phone detection model.
  - **GitHub**: For version control and project management.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/djhx3/Driver-Monitoring-System.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the system:
   ```bash
   python web1.py
   ```

## Usage
Once the system is running, it will use your camera to detect driver behavior and passenger safety. The results for **smoking**, **yawning**, **mobile phone usage**, and **aisle standing** will be displayed in real-time.

## How it Works
- **Face Detection**: The system first detects the driver’s face using OpenCV.
- **Behavior Detection**: Depending on the detected face, the system applies pre-trained or custom models to identify behaviors (e.g., smoking, yawning).
- **Mobile Phone Detection**: A Roboflow model detects whether the driver is using a mobile phone.
- **Aisle Detection**: The system checks the bus interior and identifies whether passengers are standing in the aisle during startup.

## Contributions
- Custom-trained CNN for **smoking** and **aisle detection**.
- Integrated **Roboflow’s mobile phone detection model** for real-time inference.
- Deployed the system using **Flask** to make it accessible through a web interface.
  
## Future Improvements
- Implement more driver behavior monitoring features.
- Improve model accuracy and optimize real-time inference.
- Extend the system to monitor additional distractions (e.g., talking, eating).

---

### Customizing for Your Project:
- **Dependencies**: Ensure that your `requirements.txt` file includes all necessary dependencies like **TensorFlow**, **PyTorch**, **OpenCV**, **Flask**, etc.
- **Model Training**: Update the section on **training** if you want to provide more details on how the **smoking** model was trained or how you handled the **aisle detection**.
