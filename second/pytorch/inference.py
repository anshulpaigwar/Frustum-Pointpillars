from pathlib import Path

import numpy as np
import torch

import torchplus
from second.core import box_np_ops
from second.core.inference import InferenceContext
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import box_coder_builder, second_builder
from second.pytorch.models.voxelnet import VoxelNet
from second.pytorch.train import predict_kitti_to_anno, example_convert_to_torch, _predict_kitti_to_file


class TorchInferenceContext(InferenceContext):
    def __init__(self):
        super().__init__()
        self.net = None
        self.anchor_cache = None

    def _build(self):
        config = self.config
        input_cfg = config.eval_input_reader
        model_cfg = config.model.second
        train_cfg = config.train_config
        batch_size = 1
        voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        grid_size = voxel_generator.grid_size
        self.voxel_generator = voxel_generator
        vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)

        box_coder = box_coder_builder.build(model_cfg.box_coder)
        target_assigner_cfg = model_cfg.target_assigner
        target_assigner = target_assigner_builder.build(
            target_assigner_cfg, bv_range, box_coder)
        self.target_assigner = target_assigner
        out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]
        self.net = second_builder.build(model_cfg, voxel_generator,
                                          target_assigner)
        self.net.cuda().eval()
        if train_cfg.enable_mixed_precision:
            self.net.half()
            self.net.metrics_to_float()
            self.net.convert_norm_to_float(self.net)
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        self.anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }

    def _restore(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        assert ckpt_path.suffix == ".tckpt"
        torchplus.train.restore(str(ckpt_path), self.net)

    def _inference(self, example):
        train_cfg = self.config.train_config
        input_cfg = self.config.eval_input_reader
        model_cfg = self.config.model.second
        example_torch = example_convert_to_torch(example)
        if train_cfg.enable_mixed_precision:
            float_dtype = torch.float16
        else:
            float_dtype = torch.float32

        if self.result_path:
            _predict_kitti_to_file(self.net, example_torch, self.result_path, list(
                    input_cfg.class_names), model_cfg.post_center_limit_range, model_cfg.lidar_input)
            result_annos = None

        else:
            result_annos = predict_kitti_to_anno(
                self.net, example_torch, list(input_cfg.class_names),
                model_cfg.post_center_limit_range, model_cfg.lidar_input)


        return result_annos

    def _ctx(self):
        return None
