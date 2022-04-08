import math
import cv2
import numpy as np
import scipy.spatial as spatial

def distance_to_camera(knownWidth, focalLength, perWidth):
	return (knownWidth * focalLength) / perWidth

def distance_formula(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def determine_levels(final_density, thresh=0.02):
    if final_density <= thresh:
        return 0
    elif final_density <= thresh * 2:
        return 1
    elif final_density <= thresh * 3:
        return 2
    elif final_density <= thresh * 4:
        return 3
    elif final_density <= thresh * 5:
        return 4
    elif final_density <= thresh * 6:
        return 5
    elif final_density <= thresh * 7:
        return 6
    elif final_density <= thresh * 8:
        return 7
    elif final_density <= thresh * 9:
        return 8
    elif final_density <= thresh * 10:
        return 9
    else:
        return 10

def plot_hands(results, img, w, h):
    list_one = []
    for handLms in results.multi_hand_landmarks:

        for id, lm in enumerate(handLms.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            list_one.append((lm.x * 480, lm.y * 640))

            cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
    radius = distance_formula(list_one[0], list_one[12])
    tree = spatial.KDTree(np.array(list_one))
    neighbors = tree.query_ball_tree(tree, radius)
    frequency = np.array([len(i) for i in neighbors])
    density = frequency / radius ** 2
    final_density = sum(density) / len(density) * 100
    level = determine_levels(final_density, thresh=0.02)
    return level, img

def plot_face(results_face, img, w, h):
    mesh_points = np.array(
        [np.multiply([p.x, p.y], [w, h]).astype(int) for p in results_face.multi_face_landmarks[0].landmark])
    color = [0, 255, 255]
    thickness = 1
    for i in mesh_points:
        cv2.circle(img, i, 1, color, thickness, cv2.LINE_AA)
    cx_min = w
    cy_min = h
    cx_max = cy_max = 0
    for faceLms in results_face.multi_face_landmarks:
        for id, lm in enumerate(faceLms.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if cx < cx_min:
                cx_min = cx
            if cy < cy_min:
                cy_min = cy
            if cx > cx_max:
                cx_max = cx
            if cy > cy_max:
                cy_max = cy
    width_box = cx_max - cx_min
    cv2.rectangle(img, (cx_min, cy_min), (cx_max, cy_max), (255, 0, 0), 2)
    return width_box, img