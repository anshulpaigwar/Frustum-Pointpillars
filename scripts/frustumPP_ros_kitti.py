#!/usr/bin/env python

# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import cv2
import matplotlib.pyplot as plt

import pathlib
import sys
path_model = "/home/anshul/es3cap/codes/pointpillars/second.pytorch/"
sys.path.append(path_model)
#print sys.path

from pathlib import Path
import glob
import os
#print os.getcwd()
import time
import numpy as np
import math
import json

from second.pytorch.inference import TorchInferenceContext
import second.core.box_np_ops as box_np_ops

import ipdb as pdb

bridge = CvBridge()

# plt.ion()
# fig = plt.figure()


def ry_to_rz(ry):
    angle = -ry - np.pi / 2

    if angle >= np.pi:
        angle -= np.pi
    if angle < -np.pi:
        angle = 2*np.pi + angle

    return angle




# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# code from /opt/ros/kinetic/lib/python2.7/dist-packages/tf/transformations.py
def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    # print("ak : {}".format(type(ak)))
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion

def kitti_anno_to_corners(info, annos=None):
    rect = info['calib/R0_rect']
    P2 = info['calib/P2']
    Tr_velo_to_cam = info['calib/Tr_velo_to_cam']
    if annos is None:
        annos = info['annos']
    dims = annos['dimensions']
    loc = annos['location']
    rots = annos['rotation_y']
    scores = None
    if 'score' in annos:
        scores = annos['score']
    boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    boxes_lidar = box_np_ops.box_camera_to_lidar(boxes_camera, rect,
                                                 Tr_velo_to_cam)
    boxes_corners = box_np_ops.center_to_corner_box3d(
        boxes_lidar[:, :3],
        boxes_lidar[:, 3:6],
        boxes_lidar[:, 6],
        origin=[0.5, 0.5, 0],
        axis=2)
    return boxes_corners, scores, boxes_lidar

def remove_low_score(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno['score']) if s >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations


def remove_dontcare(image_anno):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x != "DontCare"
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations



# def draw_rects(img, bboxes, color=(0,0,255), thickness=1, darken=1, show_gauss = True):
#     cmap = plt.cm.get_cmap('hsv', 256)
#     cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

#     bboxes = bboxes.astype(int)
#     # img = image.copy() * darken
#     gauss = np.zeros(img.shape, dtype = img.dtype)
#     for bbox in bboxes:
#         xmin, ymin, xmax, ymax = bbox
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax) , color, thickness, cv2.LINE_AA)

#         if show_gauss:
#             w = xmax-xmin
#             h = ymax-ymin
#             x0 =(xmax+xmin)/2
#             y0 =(ymax+ymin)/2
#             X = np.linspace(xmin, xmax, xmax - xmin, endpoint=False)
#             Y = np.linspace(ymin, ymax, ymax - ymin, endpoint=False)
#             X, Y = np.meshgrid(X, Y)
#             pos = np.vstack([X.ravel(), Y.ravel()]).T
#             prob = np.exp(-((pos[:,0] - x0)**2/(0.3*w**2)) - ((pos[:,1] - y0)**2/(0.3*h**2) ))
#             pos = pos.astype(int)
#             gauss_color = cmap[(255 * prob).astype(int), :] #CHANGED: added clip bcoz pointcloud not filtered
#             gauss[pos[:,1], pos[:,0]] = gauss_color[:,[2,1,0]]
#     added_img = cv2.addWeighted(img,0.6,gauss,0.4,0)
#     return added_img







def draw_rects(img, bboxes, color=(0,0,255), thickness=1, darken=1, show_gauss = True):
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    bboxes = bboxes.astype(int)
    # img = image.copy() * darken
    gauss = np.zeros(img.shape, dtype = img.dtype)
    temp = np.zeros((img.shape[0],img.shape[1]))
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax) , color, thickness, cv2.LINE_AA)

        if show_gauss:
            w = xmax-xmin
            h = ymax-ymin
            x0 =(xmax+xmin)/2
            y0 =(ymax+ymin)/2
            X = np.linspace(xmin, xmax, xmax - xmin, endpoint=False)
            Y = np.linspace(ymin, ymax, ymax - ymin, endpoint=False)
            X, Y = np.meshgrid(X, Y)
            pos = np.vstack([X.ravel(), Y.ravel()]).T
            new_prob = np.exp(-((pos[:,0] - x0)**2/(0.3*w**2)) - ((pos[:,1] - y0)**2/(0.3*h**2) ))   ############ original function
            # new_prob = np.exp(-((((pos[:,0] - x0)/w)**4) + ((pos[:,1] - y0)/h)**4 ))
            pos = pos.astype(int)
            cur_prob = temp[pos[:,1], pos[:,0]]
            prob_mask = new_prob < cur_prob
            new_prob[prob_mask] = cur_prob[prob_mask]
            temp[pos[:,1], pos[:,0]] = new_prob
            # pdb.set_trace()
    # fig.clear()
    # cs = plt.imshow(temp.T, interpolation='nearest')
    # cbar = fig.colorbar(cs)
    # plt.draw()

    # plt.pause(0.01)
    # plt.clf()
    gauss_color = cmap[(255 * temp.ravel()).astype(int), :] #CHANGED: added clip bcoz pointcloud not filtered
    gauss_color[temp.ravel() == 0] = 0
    gauss = gauss_color[:,[2,1,0]].reshape(gauss.shape).astype(img.dtype)
    added_img = cv2.addWeighted(img,0.7,gauss,0.3,0)
    return added_img








class Settings:
    def __init__(self, cfg_path):
        self._cfg_path = cfg_path
        self._settings = {}
        self._setting_defaultvalue = {}
        if not Path(self._cfg_path).exists():
            with open(self._cfg_path, 'w') as f:
                f.write(json.dumps(self._settings, indent=2, sort_keys=True))
        else:
            with open(self._cfg_path, 'r') as f:
                self._settings = json.loads(f.read())

    def set(self, name, value):
        self._settings[name] = value
        with open(self._cfg_path, 'w') as f:
            f.write(json.dumps(self._settings, indent=2, sort_keys=True))

    def get(self, name, default_value=None):
        if name in self._settings:
            return self._settings[name]
        if default_value is None:
            raise ValueError("name not exist")
        return default_value

    def save(self, path):
        with open(path, 'w') as f:
            f.write(json.dumps(self._settings, indent=2, sort_keys=True))

    def load(self, path):
        with open(self._cfg_path, 'r') as f:
            self._settings = json.loads(f.read())

class Processor_ROS:
    def __init__(self, config_path, ckpt_path, label_dir = None):
        self.points = None

        self.json_setting = Settings(str('/home/anshul/es3cap/codes/pointpillars/' + ".kittiviewerrc"))
        # self.config_path = self.json_setting.get("latest_vxnet_cfg_path", "")

        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.label_dir = label_dir
        
        self.image_info = None
        self.inputs = None

        self.inference_ctx = None

    def initialize(self):
        # self.read_calib()
        self.build_vxnet()
        self.load_vxnet()

    def _extend_matrix(self, mat):
        mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
        return mat

    def get_label_anno(self, label_path):
        annotations = {}
        annotations.update({
            'name': [],
            'bbox': []
        })
        with open(label_path, 'r') as f:
            lines = f.readlines()
        # if len(lines) == 0 or len(lines[0]) < 15:
        #     content = []
        # else:
        content = [line.strip().split(' ') for line in lines]
        num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
        annotations['name'] = np.array([x[0] for x in content])
        num_gt = len(annotations['name'])
        annotations['bbox'] = np.array(
            [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)

        if len(content) != 0 and len(content[0]) == 16:  # have score
            annotations['score'] = np.array([float(x[15]) for x in content])
        else:
            annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)
        annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
        return annotations

    def get_info(self, idx, data_path, calib = True, extend_matrix=True):
        label_t = time.time()
        image_info = {'image_idx': idx, 'pointcloud_num_features': 4}
        annotations = None

        if self.label_dir:
            label_path = label_dir / ('%06d.txt' % idx)
        else:
            label_path = data_path / "label_2" / ('%06d.txt' % idx)

        annotations = self.get_label_anno(str(label_path))
        # annotations = remove_low_score(annotations, 0.5)
        annotations = remove_dontcare(annotations)
        label_te = time.time()
        print('read_bbox', label_te - label_t)

        if calib:
            calib_path = data_path / "calib"/ ('%06d.txt' % idx)

            with open(str(calib_path), 'r') as f:
                lines = f.readlines()
            # P0 = np.array(
            #     [float(info) for info in lines[0].split(' ')[1:13]]).reshape(
            #         [3, 4])
            # P1 = np.array(
            #     [float(info) for info in lines[1].split(' ')[1:13]]).reshape(
            #         [3, 4])
            P2 = np.array(
                [float(info) for info in lines[2].split(' ')[1:13]]).reshape(
                    [3, 4])
            # P3 = np.array(
            #     [float(info) for info in lines[3].split(' ')[1:13]]).reshape(
            #         [3, 4])
            if extend_matrix:
                # P0 = self._extend_matrix(P0)
                # P1 = self._extend_matrix(P1)
                P2 = self._extend_matrix(P2)
                # P3 = self._extend_matrix(P3)
            # image_info['calib/P0'] = P0
            # image_info['calib/P1'] = P1
            image_info['calib/P2'] = P2
            # image_info['calib/P3'] = P3
            R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect
            image_info['calib/R0_rect'] = rect_4x4
            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = self._extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = self._extend_matrix(Tr_imu_to_velo)
            image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam

            # add image shape info for lidar point cloud preprocessing
            image_info["img_shape"] = np.array([375, 1242]) # kitti image size: height, width
        calib_te = time.time()
        print('read_calib', calib_te -label_te )
        if annotations is not None:
            image_info['annos'] = annotations
            # self.image_info = image_info
        return image_info

    def build_vxnet(self):
        print("Start build_vxnet...")
        self.inference_ctx = TorchInferenceContext()
        self.inference_ctx.build(self.config_path)
        self.json_setting.set("latest_vxnet_cfg_path", self.config_path)
        print("Build VoxelNet ckpt succeeded.")

    def load_vxnet(self):
        print("Start load_vxnet...")
        self.json_setting.set("latest_vxnet_ckpt_path", self.ckpt_path)
        self.inference_ctx.restore(self.ckpt_path)
        print("Load VoxelNet ckpt succeeded.")



    def run(self, idx, data_path, points):

        start = time.time()

        image_info = self.get_info(idx, data_path)
        self.image_info = image_info
        rect = image_info['calib/R0_rect']
        P2 = image_info['calib/P2']
        Trv2c = image_info['calib/Tr_velo_to_cam']
        image_shape = image_info['img_shape']

        initial = time.time()
        print("initial_time: ", initial - start)

        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2, image_shape)

        self.inputs = self.inference_ctx.get_inference_input_dict_ros_2(image_info, points, frustum_pp = True)

        preprocess = time.time()
        print("preprocess_time: ",  preprocess - initial)
        # self.inputs["points"] = self.process_cloud(self.inputs)

        with self.inference_ctx.ctx():
            [results] = self.inference_ctx.inference(self.inputs)

        inference = time.time()
        print("inference_time: ", inference - preprocess)
        print("total_time: ", inference - start)
        results = remove_low_score(results, 0.5)

        dt_boxes_corners, scores, dt_box_lidar = kitti_anno_to_corners(image_info, results)

        # print("dt_box_lidar: ", dt_box_lidar)
        
        return dt_boxes_corners, scores, dt_box_lidar



def KittiDataset(root, set, class_names):
    global proc
    data_path = pathlib.Path(root) / set
    lidar_path = data_path / "velodyne"
    image_path = data_path / "image_2"
    # calib_path = data_path / "calib"
    # label_path = data_path / "label_2"

    list = os.listdir(lidar_path) # dir is your directory path

    for idx in range(50, len(list)):
        arr_bbox = BoundingBoxArray()

        lidar_file = lidar_path / ('%06d.bin' % idx)
        image_file = image_path / ('%06d.png' % idx)

        cloud = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        image = cv2.imread(str(image_file))

        # start processing

        dt_boxes_corners, scores, dt_box_lidar = proc.run(idx, data_path, cloud)

        processed_cloud  =  proc.inputs['points'][0]
        if len(processed_cloud[0]) > 4:
            processed_cloud = processed_cloud[:,[0,1,2,-1]]
            processed_cloud[:,2] += 1.7
        publish_velo_modified(processed_cloud, "kitti/base_link") # Check by publising original point cloud

        cloud[:,2] += 1.7
        publish_velo(cloud, "kitti/base_link") # Check by publising original point cloud




        annos = proc.image_info['annos']
        ref_boxes = annos["bbox"]
        ref_names = annos["name"]
        ref_boxes_mask = np.array([n in class_names for n in ref_names], dtype=np.bool_)
        image = draw_rects(image, ref_boxes[ref_boxes_mask])

        # Code for gaussian goes here
        img_msg = bridge.cv2_to_imgmsg(image, encoding="passthrough")
        pub_img.publish(img_msg)

        # process results
        if scores.size != 0:
            # print('Number of detections: ', results['name'].size)
            for i in range(scores.size):
                bbox = BoundingBox()

                bbox.header.frame_id = "kitti/base_link"
                # bbox.header.stamp = rospy.Time.now()

                rotz = ry_to_rz(float(dt_box_lidar[i][6]))

                q = quaternion_from_euler(0,0, rotz)
                bbox.pose.orientation.x = q[0]
                bbox.pose.orientation.y = q[1]
                bbox.pose.orientation.z = q[2]
                bbox.pose.orientation.w = q[3]
                bbox.pose.position.x = float(dt_box_lidar[i][0])
                bbox.pose.position.y = float(dt_box_lidar[i][1])
                bbox.pose.position.z = float(dt_box_lidar[i][2]) + 1.7 * 1.5 ## added 1.7 to compensate height of lidar
                bbox.dimensions.x = float(dt_box_lidar[i][4])
                bbox.dimensions.y = float(dt_box_lidar[i][3])
                bbox.dimensions.z = float(dt_box_lidar[i][5])

                arr_bbox.boxes.append(bbox)

        arr_bbox.header.frame_id = "kitti/base_link"
        # arr_bbox.header.stamp = rospy.Time.now()
        # print("arr_bbox.boxes.size() : {} ".format(len(arr_bbox.boxes)))
        if len(arr_bbox.boxes) is not 0:
            # for i in range(0, len(arr_bbox.boxes)):
            #   print("[+] [x,y,z,dx,dy,dz] : {}, {}, {}, {}, {}, {}".\
            #           format(arr_bbox.boxes[i].pose.position.x,arr_bbox.boxes[i].pose.position.y,arr_bbox.boxes[i].pose.position.z,\
            #           arr_bbox.boxes[i].dimensions.x,arr_bbox.boxes[i].dimensions.y,arr_bbox.boxes[i].dimensions.z))
            #  publish to /voxelnet_arr_bbox
            pub_arr_bbox.publish(arr_bbox)
            #arr_bbox.boxes.clear()
            arr_bbox.boxes = []

        pdb.set_trace()









#  publishing function for DEBUG
def publish_velo_modified(cloud, frame_id):
    header = Header()
    header.stamp = rospy.Time()
    header.frame_id = frame_id

    # point cloud segments
    msg_segment = pc2.create_cloud(header=header,
                                    fields=_make_point_field(4), # 4 PointFields as channel description
                                    points=cloud)

    #  publish to /velodyne_points_modified
    pub_velo.publish(msg_segment) #  DEBUG








#  publishing function for DEBUG
def publish_velo(cloud, frame_id):
	header = Header()
	header.stamp = rospy.Time()
	header.frame_id = frame_id

	# point cloud segments
	msg_segment = pc2.create_cloud(header=header,
									fields=_make_point_field(4), # 4 PointFields as channel description
									points=cloud)

	#  publish to /velodyne_points_modified
	pub_velo_ori.publish(msg_segment) #  DEBUG

#  code from SqueezeSeg (inspired from Durant35)
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

if __name__ == '__main__':
    global proc

    # initializing Pointpillars
    # config_path = '/home/anshul/es3cap/my_codes/frustum_pp/second.pytorch/second/configs/pointpillars/ped_cycle/xyres_16.proto'
    # ckpt_path = '/home/anshul/es3cap/my_codes/frustum_pp/second.pytorch/second/ckpt/old/fpp_gauss_ped_cycle_trainval/best/voxelnet-334080.tckpt'

    config_path = '/home/anshul/es3cap/my_codes/frustum_pp/second.pytorch/second/configs/pointpillars/car/xyres_16.proto'
    ckpt_path = '/home/anshul/es3cap/my_codes/frustum_pp/second.pytorch/second/ckpt/old/fpp_gauss_car_trainval/best/voxelnet-348196.tckpt'
    # class_names = ['Pedestrian', 'Cyclist']
    class_names = ['Car']

    # label_dir = Path('/home/anshul/es3cap/kitti_data/testing/ecp_results/data')
    label_dir = None

    proc = Processor_ROS(config_path, ckpt_path, label_dir)
    proc.initialize()

    # code added for using ROS
    rospy.init_node('second_ros_node')

    # publisher
    pub_velo = rospy.Publisher("velodyne_points_modified", PointCloud2, queue_size=1)
    pub_velo_ori = rospy.Publisher("velodyne_points", PointCloud2, queue_size=1)
    pub_img = rospy.Publisher("kitti_img", Image, queue_size=1)
    pub_arr_bbox = rospy.Publisher("second_arr_bbox", BoundingBoxArray, queue_size=10)

    KittiDataset(root = "/home/anshul/es3cap/kitti_data/", set = "training", class_names = class_names)
