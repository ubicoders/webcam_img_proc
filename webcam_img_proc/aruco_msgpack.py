from aruco_interface.msg import ImageMarkers, ArucoMarker, Point2D
from std_msgs.msg import Header
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import rclpy


def pack_aruco(serial_number, left_info, right_info={}):
    # Initialize a Header message
    header = Header()

    # Populate the Header fields
    header.stamp = rclpy.clock.Clock().now().to_msg()  # Set the timestamp to the current time
    header.frame_id = "webcam"  # Set the frame_id to some meaningful frame

    # Initialize an ImageMarkers message
    image_markers = ImageMarkers()
    image_markers.header = header
    cam_name_msg = String()
    cam_name_msg.data = f"{serial_number}"
    image_markers.cam_name = cam_name_msg
    image_markers.aruco_markers_0 = pack_aruco_markers(left_info)
    if right_info is not None:
        image_markers.aruco_markers_1 = pack_aruco_markers(right_info)
    return image_markers

def pack_aruco_markers(list_aruco_info):
    keys = list_aruco_info.keys()
    aruco_markers = []
    for key in keys:
        if key not in [0, 1, 2, 3, 4, 5]:
            continue
        marker = pack_each_aruco(key, list_aruco_info[key])
        aruco_markers.append(marker)
    return aruco_markers

def get_value(dict, key_name):
    if key_name in dict:
        return dict[key_name]
    return None

def pack_each_aruco(mid, marker_info):
    
    nd_corners = get_value(marker_info, "corners")
    nd_tvec = get_value(marker_info, "tvec")
    nd_eul = get_value(marker_info, "eul")
    nd_center = get_value(marker_info, "center")
    nd_xyz = get_value(marker_info, "xyz")   
    
    marker = ArucoMarker()
    marker.mid = int(mid)

    # corners
    all_corners = []    
    for i in range(nd_corners.shape[0]):
        point = Point2D()
        point.x = int(nd_corners[i, 0])
        point.y = int(nd_corners[i, 1])
        all_corners.append(point)
    marker.corners = all_corners

    # tvec
    if (nd_tvec is not None):
        vec3 = Vector3()
        vec3.x = nd_tvec[0]
        vec3.y = nd_tvec[1]
        vec3.z = nd_tvec[2]
        marker.tvec = vec3
    
    # eul
    if (nd_eul is not None):
        vec3 = Vector3()
        vec3.x = nd_eul[0]
        vec3.y = nd_eul[1]
        vec3.z = nd_eul[2]
        marker.eul = vec3

    # center
    if (nd_center is not None):
        point = Vector3()
        point.x = nd_center[0]*1.0
        point.y = nd_center[1]*1.0
        marker.center = point

    # xyz
    if (nd_xyz is not None):
        vec3 = Vector3()
        vec3.x = nd_xyz[0]
        vec3.y = nd_xyz[1]
        vec3.z = nd_xyz[2]
        marker.pc_xyz = vec3
    return marker