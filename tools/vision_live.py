import time
import cv2
from imagedetection.detector import Detector
from runtime.camera_provider import get_frame_bgr, close_camera

def main():
    det = Detector(
        board_indices_csv="assets/indices/board_indices.csv",
        board_model_path="assets/models/board_best.pt",
        stones_model_path="assets/models/stones_best.pt",
        stacks_model_path="assets/models/stacks_best.pt",
        conf_min=0.25,
        conf_accept=0.45,
        dist_max_factor=0.7,
        black_cls_id=0,
        white_cls_id=1,
        frame_provider=get_frame_bgr,
    )

    print("q=quit | s=save frame")
    while True:
        frame = get_frame_bgr()
        try:
            _, info = det.get_gamestate(frame)
            vis = info.get("debug_vis", frame)
        except Exception as e:
            vis = frame.copy()
            cv2.putText(vis, str(e), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("vision", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            fn = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, frame)
            print("saved:", fn)

    close_camera()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
