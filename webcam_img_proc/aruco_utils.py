import cv2.aruco as aruco
import cv2
import numpy as np
import os


def rotation_matrix_to_euler_angles(R):
    """
    Converts a rotation matrix to Euler angles (in radians).
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z]) 

 
class ArucoDetector:
    def __init__(self, pose_on=False , expected_mids =  [0, 1, 2, 3, 4, 5]) -> None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        arucoParameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(aruco_dict, arucoParameters)

        self.pose_on = pose_on  
        self.marker_size = 0.165
        self.corners = None
        self.ids = None
        self.aruco_info = {}
        self.camera_matrix = np.zeros((3, 3))
        self.dist_coeffs = np.zeros((5, 1))
        self.expected_mids = expected_mids  

    def detect_bgr(self, bgr_img):
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        self.detect_gray(gray_img)

    def detect_gray(self, gray_img):
        self.aruco_info = {}
        # corners, ids, rejectedImgPoints
        self.corners, self.ids, _ = self.detector.detectMarkers(gray_img)
        if self.ids is None or len(self.ids) == 0:
            return
        # make a map of ids to corners
        for i, marker_idx in enumerate(self.ids):
            mid = marker_idx[0]
            if mid not in self.expected_mids:   
                continue
            corners = np.reshape(self.corners[i], (4, 2))
            self.aruco_info[mid] = {"corners": corners}

        # rotaton and translation estimation
        if (self.pose_on):
            self.estimate_position()

    
    def estimate_position(self):
        if (self.camera_matrix is None) or (self.dist_coeffs is None):
            print("Error: Camera calibration data not found")
            return None, None
        if (self.corners is None) or (self.ids is None):
            #print("Error: No ArUco markers detected")
            return None, None
        self.rvecs, self.tvecs, _ = aruco.estimatePoseSingleMarkers(self.corners, self.marker_size , self.camera_matrix, self.dist_coeffs)
        for i in range(len(self.ids)):
            mid = self.ids[i][0]
            if mid not in self.get_all_ids():
                continue
            R, _ = cv2.Rodrigues(self.rvecs[i])
            # Calculate distance to the marker
            distance = np.linalg.norm(self.tvecs[i])
            eul = rotation_matrix_to_euler_angles(R)
            eul[0] += np.pi
            # (-np.pi to np.pi)
            eul[0] = (eul[0] + np.pi) % (2 * np.pi) - np.pi
            self.aruco_info[mid]["tvec"] = self.tvecs[i][0]
            self.aruco_info[mid]["eul"] = eul
            # self.idTRMap[self.ids[i][0]] = {"tvec": self.tvecs[i], "eul": eul}

    def drawMarkers(self, img):
        if (self.corners is None) or (self.ids is None):
            return img
        img = aruco.drawDetectedMarkers(img, self.corners)
        return img
    
    def draw_pose(self, img):
        if (self.ids is None):
            return img
        for i in range(len(self.ids)):
            cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs, self.rvecs[i], self.tvecs[i], 0.1)
        return img
    
    def update_idXYZ(self, mid, xyz):
        self.aruco_info[mid]["xyz"] = xyz
    
    def get_all_ids(self):
        return self.aruco_info.keys()
    
    def update_center(self):
        for mid in self.get_all_ids():
            center = self.get_center(mid)
            self.aruco_info[mid]["center"] = center
    
    def get_center(self, mid):
        if mid in self.get_all_ids():
            corners = self.aruco_info[mid]["corners"]
            center = np.mean(corners, axis=0)
            return center
        return None