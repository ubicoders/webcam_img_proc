import rclpy
from rclpy.node import Node
import cv2
from .perf_utils import TimerTicTok
from .cam_wrapper import ELP210Wrapper
from .aruco_utils import ArucoDetector
from aruco_interface.msg import ImageMarkers
from .aruco_msgpack import pack_aruco

class WebcamDisplayNode(Node):
    def __init__(self):
        super().__init__('elp_210')
        self.tictok = TimerTicTok()
        

        self.pub_aruco = self.create_publisher(ImageMarkers, '/cam/aruco', 10)
       
        # Initialize webcam (default is camera index 0)
        self.cam = ELP210Wrapper(4)
        self.aruco_det = ArucoDetector(pose_on=True)
        self.aruco_det.camera_matrix = self.cam.K
        self.aruco_det.dist_coeffs = self.cam.dist

        # Create a timer that triggers the callback function at a rate of 30 Hz
        self.timer = self.create_timer(0.001, self.timer_callback)

    def timer_callback(self):
        self.tictok.update_and_pprint()
        self.cam.update()

        img_rect = self.cam.get_rect_img()
        #print(img_rect.shape)

        # Detect ArUco markers in the image
        self.aruco_det.detect_bgr(img_rect)
        img_rect = self.aruco_det.drawMarkers(img_rect)

        # Publish the ArUco markers
        self.pub_aruco.publish(pack_aruco("elp210", self.aruco_det.aruco_info))

        # # Display the image
        # cv2.imshow("Webcam Feed", img_rect)
        # cv2.waitKey(1) 

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()




def main(args=None):
    rclpy.init(args=args)
    node = WebcamDisplayNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user (KeyboardInterrupt).")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
