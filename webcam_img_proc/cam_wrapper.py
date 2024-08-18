import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory


def get_file_name(fName):
    return os.path.join(get_package_share_directory('webcam_img_proc'), 'elp210', fName)


class ELP210Wrapper():
    def __init__(self, cam_idx) -> None:
        calib_file_path = get_file_name("cam_calib.npz") 

        # Load the calibration parameters from the file
        calib_data = np.load(calib_file_path)
        K = calib_data["k"]
        dist = calib_data["dist"]
        self.cap = cv2.VideoCapture(cam_idx)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 150)
        image_shape = (1280, 720)  # Assuming width x height
        self.K, self.roi = cv2.getOptimalNewCameraMatrix(K, dist, image_shape, 1, image_shape)
        self.dist = np.zeros_like(dist)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(K, dist, None, self.K, image_shape, 5)
    
    def update(self):
        ret, self.img_raw = self.cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            return None
        self.img_rect = cv2.remap(self.img_raw, self.mapx, self.mapy, cv2.INTER_LINEAR)
        x, y, w, h = self.roi
        self.img_rect = self.img_rect[y:y+h, x:x+w]
    
    def get_raw_img(self):
        return self.img_raw
    
    def get_rect_img(self):
        return self.img_rect

    def __del__(self):
        self.cap.release()