**1.0 BlueROV2 Rope Detection Navigation System**

This repository contains the implementation of a machine learning–integrated rope detection and navigation assistance system developed for the BlueROV2 platform.  
The system combines YOLO-based object detection, HSV colour segmentation, geometric validation, and navigation guidance to improve situational awareness during underwater inspection.

This project was developed as part of a Final Year Project at Universiti Teknologi PETRONAS.


**2.0 Project Overview**

Underwater inspection using Remotely Operated Vehicles (ROVs) is challenging due to poor visibility, turbidity, and unstable vehicle motion.  
This system detects guide ropes in real time and provides directional assistance to help the operator maintain alignment during navigation.

**2.1 Main features:**

- YOLO-based underwater object detection
- HSV-based rope validation
- Geometric line verification
- Navigation guidance (Left / Center / Right)
- Real-time dashboard display
- Designed for BlueROV2 + BlueOS environment


**3.0 Requirements**

Python 3.9 or above

Required libraries:

- opencv-python
- numpy
- torch
- ultralytics
- flask (for dashboard)
- matplotlib

Install dependencies:

```bash
pip install -r requirements.txt

** 4.0 Model Training**

The detection model was trained using YOLOv11n with transfer learning.

**4.1 Training settings:**

Image size: 640 × 640

Epochs: 120

Optimizer: SGD

Data augmentation:

HSV adjustment

Rotation

Scaling

Mosaic

MixUp

**5.0 Running the Navigation Dashboard System**

Run:
python main.py

**6.0 Navigation Assistance**

The camera frame is divided into three regions:

Left

Centre

Right

Bounding box centre determines navigation direction.

Example:
Left  → steer left
Center → aligned
Right → steer right

**6.0 BlueOS / BlueROV2 Deployment**

The system can be packaged using Docker and deployed as a BlueOS Extension.

Reference:

https://blueos.cloud/docs/stable/development/extensions/

https://blueos.cloud/docs/stable/development/overview/#docker

For stable performance, a higher-performance onboard computer is recommended instead of Raspberry Pi.

Author

Muhammad Ilham Bin Mohammad Faisal
Universiti Teknologi PETRONAS
Final Year Project

Supervisor

Dr. Mohamed Nordin Bin Zakaria
Universiti Teknologi PETRONAS

License

This repository is for academic and research purposes only.
