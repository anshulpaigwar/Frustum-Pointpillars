#!/usr/bin/env python

import numpy as np

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

def hv_in_range(x, y, z, fov, fov_type='h'):
	"""
	Extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit

	Args:
	`x`:velodyne points x array
	`y`:velodyne points y array
	`z`:velodyne points z array
	`fov`:a two element list, e.g.[-45,45]
	`fov_type`:the fov type, could be `h` or 'v',defualt in `h`

	Return:
	`cond`:condition of points within fov or not

	Raise:
	`NameError`:"fov type must be set between 'h' and 'v' "
	"""
	d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
	if fov_type == 'h':
		return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi/180), np.arctan2(y, x) < (-fov[0] * np.pi/180))
	elif fov_type == 'v':
		return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), np.arctan2(z, d) > (fov[0] * np.pi / 180))
	else:
		raise NameError("fov type must be set between 'h' and 'v' ")

def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]

def velo_cb(msg):
    global pcl_msg, cond, np_p, np_p_ranged
    pcl_msg = pc2.read_points(msg, skip_nans=False)
    np_p = np.array(list(pcl_msg), dtype=np.float32)
    np_p = np.delete(np_p, -1, 1)

    cond = hv_in_range(x=np_p[:,0],
    y=np_p[:,1],
    z=np_p[:,2],
    fov=[-45,45],
    fov_type='h')
    
    np_p_ranged = np_p[cond]
    
    #  publish to /velodyne_poitns_modified
    publish_test(np_p_ranged, msg.header.frame_id)

#  publishing function for DEBUG
def publish_test(np_p_ranged,frame_id):
	header = Header()
	header.stamp = rospy.Time()
	header.frame_id = frame_id

	x = np_p_ranged[:, 0].reshape(-1)
	y = np_p_ranged[:, 1].reshape(-1)
	z = np_p_ranged[:, 2].reshape(-1)

	# if intensity field exists
	if np_p_ranged.shape[1] == 4:
		i = np_p_ranged[:, 3].reshape(-1)
	else:
		i = np.zeros((np_p_ranged.shape[0], 1)).reshape(-1)

	cloud = np.stack((x, y, z, i))

	# point cloud segments
	# 4 PointFields as channel description
	msg_segment = pc2.create_cloud(header=header,
									fields=_make_point_field(4),
									points=cloud.T)
	#  publish to /velodyne_points_modified
	pub_velo.publish(msg_segment) #  DEBUG

if __name__ == '__main__':

    # save_model_dir = os.path.join(path_model + '/save_model', 'pre_trained_car')

    #  initializing voxelnet
    # voxelnet_init()

    #  code added for using ROS
    rospy.init_node('test_node')

    sub_ = rospy.Subscriber("velodyne_points", PointCloud2, velo_cb, queue_size=1)

    pub_velo = rospy.Publisher("velodyne_points_modified", PointCloud2, queue_size=1)
    
    print("[+] voxelnet_ros_node has started!")
    rospy.spin()
