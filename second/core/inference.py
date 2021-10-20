import abc
import contextlib

import numpy as np
from google.protobuf import text_format

from second.data.preprocess import merge_second_batch, prep_pointcloud
from second.protos import pipeline_pb2


class InferenceContext:
    def __init__(self):
        self.config = None
        self.root_path = None
        self.target_assigner = None
        self.voxel_generator = None
        self.anchor_cache = None
        self.built = False
        self.result_path = None

    def get_inference_input_dict(self, info, points):
        assert self.anchor_cache is not None
        assert self.target_assigner is not None
        assert self.voxel_generator is not None
        assert self.config is not None
        assert self.built is True
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']
        input_cfg = self.config.eval_input_reader
        model_cfg = self.config.model.second

        input_dict = {
            'points': points,
            'rect': rect,
            'Trv2c': Trv2c,
            'P2': P2,
            'image_shape': np.array(info["img_shape"], dtype=np.int32),
            'image_idx': info['image_idx'],
            'image_path': info['img_path'],
            # 'pointcloud_num_features': num_point_features,
        }
        out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]
        example = prep_pointcloud(
            input_dict=input_dict,
            root_path=str(self.root_path),
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            max_voxels=input_cfg.max_number_of_voxels,
            class_names=list(input_cfg.class_names),
            training=False,
            create_targets=False,
            shuffle_points=input_cfg.shuffle_points,
            generate_bev=False,
            without_reflectivity=model_cfg.without_reflectivity,
            num_point_features=model_cfg.num_point_features,
            anchor_area_threshold=input_cfg.anchor_area_threshold,
            anchor_cache=self.anchor_cache,
            out_size_factor=out_size_factor,
            out_dtype=np.float32)
        example["image_idx"] = info['image_idx']
        example["image_shape"] = input_dict["image_shape"]
        example["points"] = points
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        #############
        # convert example to batched example
        #############
        example = merge_second_batch([example])
        return example


    def get_inference_input_dict_ros(self, info, points):
        assert self.anchor_cache is not None
        assert self.target_assigner is not None
        assert self.voxel_generator is not None
        assert self.config is not None
        assert self.built is True
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']
        input_cfg = self.config.eval_input_reader
        model_cfg = self.config.model.second

        input_dict = {
            'points': points,
            'rect': rect,
            'Trv2c': Trv2c,
            'P2': P2,
            'image_shape': np.array(info["img_shape"], dtype=np.int32),
            # 'image_idx': info['image_idx'],
            # 'image_path': info['img_path'],
            # 'pointcloud_num_features': num_point_features,
        }
        out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]
        example = prep_pointcloud(
            input_dict=input_dict,
            root_path=str(self.root_path),
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            max_voxels=input_cfg.max_number_of_voxels,
            class_names=list(input_cfg.class_names),
            training=False,
            create_targets=False,
            shuffle_points=input_cfg.shuffle_points,
            generate_bev=False,
            without_reflectivity=model_cfg.without_reflectivity,
            num_point_features=model_cfg.num_point_features,
            anchor_area_threshold=input_cfg.anchor_area_threshold,
            anchor_cache=self.anchor_cache,
            out_size_factor=out_size_factor,
            out_dtype=np.float32)
        # example["image_idx"] = info['image_idx']
        example["image_shape"] = input_dict["image_shape"]
        example["points"] = points
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        #############
        # convert example to batched example
        #############
        example = merge_second_batch([example])
        return example


    def get_inference_input_dict_ros_2(self, info, points, frustum_pp = True, add_points_to_example = True):
        assert self.anchor_cache is not None
        assert self.target_assigner is not None
        assert self.voxel_generator is not None
        assert self.config is not None
        assert self.built is True
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']
        input_cfg = self.config.eval_input_reader
        model_cfg = self.config.model.second

        input_dict = {
            'points': points,
            'rect': rect,
            'Trv2c': Trv2c,
            'P2': P2,
            'image_shape': np.array(info["img_shape"], dtype=np.int32),
            'image_idx': info['image_idx'],
            # 'image_path': info['img_path'],
            # 'pointcloud_num_features': num_point_features,
        }
        if 'annos' in info:
            annos = info['annos']
            # annos = kitti.remove_dontcare(annos)
            gt_names = annos["name"]
            bboxes = annos["bbox"]
            input_dict.update({
                'gt_names': gt_names,
                'bboxes': bboxes,
            })
        out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]
        example = prep_pointcloud(
            input_dict=input_dict,
            root_path=str(self.root_path),
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            max_voxels=input_cfg.max_number_of_voxels,
            class_names=list(input_cfg.class_names),
            training=False,
            create_targets=False,
            shuffle_points=input_cfg.shuffle_points,
            generate_bev=False,
            without_reflectivity=model_cfg.without_reflectivity,
            num_point_features=model_cfg.num_point_features,
            anchor_area_threshold=input_cfg.anchor_area_threshold,
            anchor_cache=self.anchor_cache,
            out_size_factor=out_size_factor,
            out_dtype=np.float32,
            frustum_pp = frustum_pp,
            add_points_to_example=add_points_to_example)
        example["image_idx"] = info['image_idx']
        example["image_shape"] = input_dict["image_shape"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        #############
        # convert example to batched example
        #############
        example = merge_second_batch([example])
        return example



    def get_config(self, path):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        return config

    @abc.abstractclassmethod
    def _build(self):
        raise NotImplementedError()

    def build(self, config_path):
        self.config = self.get_config(config_path)
        ret = self._build()
        self.built = True
        return ret

    @abc.abstractclassmethod
    def _inference(self, example):
        raise NotImplementedError()

    def inference(self, example):
        return self._inference(example)

    @abc.abstractclassmethod
    def _restore(self, ckpt_path):
        raise NotImplementedError()

    def restore(self, ckpt_path):
        return self._restore(ckpt_path)

    @abc.abstractclassmethod
    def _ctx(self):
        raise NotImplementedError()

    @contextlib.contextmanager
    def ctx(self):
        yield self._ctx()
