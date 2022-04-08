import cv2
import mediapipe as mp

from utils import distance_to_camera, plot_face, plot_hands

cap = cv2.VideoCapture(0)

mp_hand_mesh = mp.solutions.hands
hands = mp_hand_mesh.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
colors = [(0,255,0), (0,235,15), (0,220,30), (0,205,50), (0,185,70), (0,155,100), (0,125,135), (0,90,165), (0,60,195), (0,30,225), (0,0,255)]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6)

INITIAL_DISTANCE = 30.0
FACE_WIDTH = 8.0
fl = False
focal_length = 0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
save_video = True
if save_video == True:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))
else:
    out = None

while True:
    success, img = cap.read()
    if not success:
        break
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(imgrgb)
    results_face = face_mesh.process(imgrgb)
    width_box = 0
    if results_hand.multi_hand_landmarks:
        level, img = plot_hands(results_hand, img, w, h)
        cv2.putText(img,f'Level: {level}', (10,50), cv2.FONT_HERSHEY_PLAIN, 3, colors[level], 3)

    if results_face.multi_face_landmarks:
        width_box, img = plot_face(results_face, img, w, h)
        if fl == False:
            cv2.putText(img, 'FL not Set',
                        (0, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        else:
            inches = distance_to_camera(FACE_WIDTH, focal_length, width_box)
            cv2.putText(img, "Face Distance: %.2fft" % (inches / 12),
                        (0, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
    if save_video == True:
        out.write(img)
    else:
        pass
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q') and width_box != 0:
        focal_length = (width_box * INITIAL_DISTANCE) / FACE_WIDTH
        fl = True
    if key == ord('s'):
        break

if save_video == True:
    out.release()
cap.release()
cv2.destroyAllWindows()