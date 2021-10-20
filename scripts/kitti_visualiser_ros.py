import os
import pathlib
import fire
import numpy as np
import cv2
import ipdb as pdb



# Ros Includes
import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point
import ros_numpy



rospy.init_node('kitti_visualiser', anonymous=True)
pcl_pub = rospy.Publisher("kitti_velo", PointCloud2, queue_size=10)





def rgb_to_float(color):
    """ Converts an RGB list to the packed float format used by PCL
    
        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"
    
        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]
            
        Returns:
            float_rgb: RGB value packed as a float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb





def publish_cloud(points, timestamp = None):
    npoints = points.shape[0] # Num of points in PointCloud
    points_arr = np.zeros((npoints,), dtype=[
                                        ('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('intensity', np.float32)])
    points = np.transpose(points)
    points_arr['x'] = points[0]
    points_arr['y'] = points[1]
    points_arr['z'] = points[2]
    points_arr['intensity'] = rgb_to_float([int(points[4]), int(points[5]),int(points[6])])


    if timestamp == None:
        timestamp = rospy.Time.now()
    cloud_msg = ros_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id = "/kitti/base_link")
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)






# def np2ros_pub_2(points, pcl_pub, timestamp, color):
#     npoints = points.shape[0] # Num of points in PointCloud
#     points_arr = np.zeros((npoints,), dtype=[
#                                         ('x', np.float32),
#                                         ('y', np.float32),
#                                         ('z', np.float32),
#                                         ('intensity', np.float32)])
#     points = np.transpose(points)
#     points_arr['x'] = points[0]
#     points_arr['y'] = points[1]
#     points_arr['z'] = points[2]

#     # float_rgb = rgb_to_float(color)
#     points_arr['intensity'] = Points[3]
#     # points_arr['g'] = 255
#     # points_arr['b'] = 255

#     if timestamp == None:
#         timestamp = rospy.Time.now()
#     cloud_msg = ros_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id = "/kitti/base_link")
#     # rospy.loginfo("happily publishing sample pointcloud.. !")
#     pcl_pub.publish(cloud_msg)








def KittiDataset(root, set, rgb = False):
    data_path = pathlib.Path(root) / set
    lidar_path = data_path / "velodyne"
    image_path = data_path / "image_2"
    calib_path = data_path / "calib"
    label_path = data_path / "label_2"

    list = os.listdir(image_path) # dir is your directory path

    for ids in range(len(list)):
        lidar_file = lidar_path / ('%06d.bin' % ids)
        image_file = image_path / ('%06d.png' % ids)
        calib_file = calib_path / ('%06d.txt' % ids)
        label_file = label_path / ('%06d.txt' % ids)
        if rgb:
            lidar = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 7)
        else:
            lidar = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        image = cv2.imread(str(image_file))
        publish_cloud(lidar)
        pdb.set_trace()






if __name__ == '__main__':
    fire.Fire(KittiDataset)



