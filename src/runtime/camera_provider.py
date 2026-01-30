import os
import cv2
import numpy as np

_CAP = None


def get_camera(index: int = 0):
    cap = cv2.VideoCapture(index)  # kein GStreamer!
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")

    # optional: direkt ein paar Settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def get_frame_bgr():
    cap = get_camera(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # kaputter Frame, skip

        # Prüfen, ob Frame nicht komplett schwarz ist
        if frame is None or np.mean(frame) < 10:  # Threshold anpassen
            continue  # zu dunkel/black, skip

        cv2.waitKey(1)  # nötig, sonst zeigt OpenCV nichts
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def close_camera():
    global _CAP
    if _CAP is not None:
        _CAP.release()
        _CAP = None
