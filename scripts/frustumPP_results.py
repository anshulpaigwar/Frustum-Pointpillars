#!/usr/bin/env python

# -*- coding: utf-8 -*-
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
from second.utils.progress_bar import ProgressBar
import ipdb as pdb


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
    def __init__(self, config_path, ckpt_path, result_path, class_names):
        self.points = None

        self.json_setting = Settings(str('/home/anshul/es3cap/codes/pointpillars/' + ".kittiviewerrc"))
        # self.config_path = self.json_setting.get("latest_vxnet_cfg_path", "")

        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.result_path = result_path
        
        self.image_info = None
        self.inputs = None

        self.inference_ctx = None
        self.class_names = class_names

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

    def get_info(self, idx, data_path, label_info =True, calib = True, extend_matrix=True):

        image_info = {'image_idx': idx, 'pointcloud_num_features': 4}
        annotations = None

        if label_info:
            label_path = data_path / "label_2" / ('%06d.txt' % idx)
            annotations = self.get_label_anno(str(label_path))
            # annotations = remove_low_score(annotations, 0.5)
            annotations = remove_dontcare(annotations)

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
        self.inference_ctx.result_path = self.result_path

    def load_vxnet(self):
        print("Start load_vxnet...")
        self.json_setting.set("latest_vxnet_ckpt_path", self.ckpt_path)
        self.inference_ctx.restore(self.ckpt_path)
        print("Load VoxelNet ckpt succeeded.")



    def run(self, idx, data_path, points):

        image_info = self.get_info(idx, data_path)
        rect = image_info['calib/R0_rect']
        P2 = image_info['calib/P2']
        Trv2c = image_info['calib/Tr_velo_to_cam']
        image_shape = image_info['img_shape']

        annos = image_info['annos']
        ref_names = annos["name"]
        ref_boxes_mask = np.array([n in self.class_names for n in ref_names], dtype=np.bool_)
        if ref_boxes_mask.any() is not None:

            points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2, image_shape)

            self.inputs = self.inference_ctx.get_inference_input_dict_ros_2(image_info, points, frustum_pp = True, add_points_to_example = False)

            with self.inference_ctx.ctx():
                self.inference_ctx.inference(self.inputs)
        else:
            print('creating empty file %06d.txt'% idx)
            file_name = self.result_path + '/' + '%06d.txt' % idx
            f = open(file_name, 'a+')  # open file in append mode
            f.close()



def KittiDataset(root, set):
    global proc
    data_path = pathlib.Path(root) / set
    lidar_path = data_path / "velodyne"
    # image_path = data_path / "image_2"
    # calib_path = data_path / "calib"
    # label_path = data_path / "label_2"

    list = os.listdir(lidar_path) # dir is your directory path
    prog_bar = ProgressBar()
    prog_bar.start(len(list))
    for idx in range(len(list)):

        lidar_file = lidar_path / ('%06d.bin' % idx)
        # image_file = image_path / ('%06d.png' % ids)
        # calib_file = calib_path / ('%06d.txt' % ids)
        # label_file = label_path / ('%06d.txt' % ids)
        cloud = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        # image = cv2.imread(str(image_file))

        # start processing
        proc.run(idx, data_path, cloud)
        prog_bar.print_bar()









if __name__ == '__main__':
    global proc

    # initializing Pointpillars
    config_path = '/home/anshul/es3cap/my_codes/frustum_pp/second.pytorch/second/configs/pointpillars/ped_cycle/xyres_16.proto'
    ckpt_path = '/home/anshul/es3cap/my_codes/frustum_pp/second.pytorch/second/ckpt/frustum_pp_ped/voxelnet-261559.tckpt'
    result_path = "/home/anshul/results"
    class_names = ['Pedestrian']
    # config_path = '/home/anshul/es3cap/my_codes/frustum_pp/second.pytorch/second/configs/pointpillars/car/xyres_16.proto'
    # ckpt_path = '/home/anshul/es3cap/my_codes/frustum_pp/second.pytorch/second/ckpt/frustum_pp_car/voxelnet-271305.tckpt'
    proc = Processor_ROS(config_path, ckpt_path, result_path, class_names)
    proc.initialize()

    KittiDataset(root = "/home/anshul/es3cap/kitti_data/", set = "training")
