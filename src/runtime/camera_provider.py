import os
import cv2

_CAP = None

def _source():
    src = os.environ.get("CAMERA_SRC", "0")
    try:
        return int(src)
    except ValueError:
        return src
    
def get_camera(source=None):
    global _CAP
    if source is None:
        source = _source()
    if _CAP is None:
        if isinstance(source, int):
            _CAP = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            _CAP = cv2.VideoCapture(source)
        if not _CAP.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
    return _CAP

def get_frame_bgr():
    cap = get_camera()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read camera frame")
    return frame

def close_camera():
    global _CAP
    if _CAP is not None:
        _CAP.release()
        _CAP = None
