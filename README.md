# Face Distance Estimation and Hand Gesture Control:
A python repository to fast and reliable face and hands landmarks detections and relative applications using mediapipe.

# Pre-requisits:
The code is tested on ubuntu 20.04 with python3.6.

pip3 install -r requirements.txt

# Inference
Simply run main.py after setting up your webcame or if you want to test the code on recorded video simply give its correct path in line 6.

# Face Distance Estimation:
The Face distance from camera is estimated on the base of a pre-defined value and current width of face detected.
The Value is set to 30 inches in line 21. The program start detecting distance of the face once you press q on the keyboard.
Shortly, when you feel that you are at almost 30 inches from the camera, press q, that will set a focal length and start measuring distance afterwards.

# Gesture Control:
The Gesture Control Algorithm is based on points cloud density which sets the levels according to status of hand.
Fully Open Palm: Level 0
Fully Closed Fist: Level 10

# Terminate Inference:
Press s on the keyboard to exit the process.
