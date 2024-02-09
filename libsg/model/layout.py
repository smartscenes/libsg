"""
layout.py
---
Classes to define unified wrapper interface for each layout model.

TODO: This could eventually be split out into multiple files if we add more than 2 model architectures.
"""

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from libsg.model.atiss import atiss_network
from libsg.model.diffuscene import build_network
from libsg.scene_types import SceneLayoutSpec
from libsg.model.utils import descale


class LayoutBase:
    """
    Implements wrapper for scene layout generators.
    """

    def generate(self, layout_spec: SceneLayoutSpec, **kwargs) -> list[dict[str, Any]]:
        """
        Generates scene layout based on specification.

        :param layout_spec: specification for layout architecture
        :return: dictionary of the form
            {
                'class_labels': tensor of scores for classes for each object,
                'dimensions': tensor of dimensions for each object,
                'position': tensor of positions for each object,
                'orientation': tensor of angles to apply to each object,
            }
        """
        raise NotImplementedError


class ATISS(LayoutBase):
    """ATISS model wrapper"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.classes = list(config.data.classes)
        self.bounds = config.data.bounds
        self.room_layout_size = int(config.data.room_layout_size.split(",")[0])
        self.room_dims = config.data.room_dims
        self.up_axis = config.data.up_axis
        self.export_room_mask = config.export_room_mask
        self.model = atiss_network(config=config)
        self.model.eval()

    def prepare_inputs(self, layout_spec: SceneLayoutSpec) -> torch.FloatTensor:
        """
        Create room layout input which can be ingested by model.

        :param layout_spec: specification for layout architecture
        :return: tensor of mask of room, with 1 representing valid placement and 0 otherwise
        """
        room_mask = layout_spec.arch.get_room_mask(layout_size=self.room_layout_size, room_dims=self.room_dims, device="cpu")

        # write room_mask to image for debugging
        if self.export_room_mask:
            image = Image.fromarray((room_mask[0, 0].numpy() * 255).astype(np.uint8), "L")
            image.save("room_mask.png")

        return room_mask

    def generate(self, layout_spec: SceneLayoutSpec, **kwargs) -> list[dict[str, Any]]:
        room_mask = self.prepare_inputs(layout_spec)
        bbox_params = self.model.generate_boxes(room_mask=room_mask)
        boxes = self.descale_bbox_params(bbox_params)
        return self.prepare_outputs(boxes)

    def descale_bbox_params(self, s: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply descaling to bounding box predictions of model to original scale.

        The following operations are performed on each of the parameters:
        * room_layout - as is
        * class_labels - as is
        * objfeats - as is
        * everything else - scaled by bounds[0] and bounds[1]

        :param s: dictionary of outputs from model, pre-scaling
        :return: scaled outputs from model in the same form as the input
        """
        sample_params = {}
        for k, v in s.items():
            if k == "room_layout" or k == "class_labels" or k == "objfeats":
                sample_params[k] = v
            else:
                sample_params[k] = descale(v, np.asarray(self.bounds[k][0]), np.asarray(self.bounds[k][1]))
        return sample_params

    def prepare_outputs(self, boxes: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        """
        Format outputs into standardized form.

        :param boxes: dictionary of the form
            {
                'class_labels': tensor of scores for classes for each object,
                'dimensions': tensor of dimensions for each object,
                'position': tensor of positions for each object,
                'orientation': tensor of angles to apply to each object,
            }
        :return: list of objects of the form
            {
                'wnsynsetkey': name of object class,
                'dimensions': dimensions of object as (x, y, z),
                'position': position of object in scene as (x, y, z),
                'orientation': rotation angle to apply to each object in radians,
            }
        """
        objects = []

        # flip axes if needed
        if self.up_axis == "y":  # assumes LHR, y-axis is vertical, and axes are rotated 180 deg
            axis_order = (0, 2, 1)
            boxes["translations"][0, :, :] = boxes["translations"][0, :, axis_order]
            boxes["translations"][0, :, 0] = -boxes["translations"][0, :, 0]
            boxes["angles"][0, :, :] -= torch.pi
            boxes["angles"][0, :, :] = torch.where(
                boxes["angles"][0, :, :] > -torch.pi, boxes["angles"][0, :, :], boxes["angles"][0, :, :] + 2 * torch.pi
            )
            # NOTE: we do not change the size of the boxes, because assets are mostly stored as vertical y-axis
        else:  # self.up_axis == "z"
            pass

        # reshape outputs
        for i in range(1, boxes["class_labels"].shape[1] - 1):
            objects.append(
                {
                    "wnsynsetkey": self.classes[boxes["class_labels"][0, i].argmax(-1)],
                    "dimensions": boxes["sizes"][0, i, :].tolist(),
                    "position": boxes["translations"][0, i, :].tolist(),
                    "orientation": float(boxes["angles"][0, i, 0]),
                }
            )
        return objects


class DiffuScene(LayoutBase):
    """Implements wrapper for DiffuScene layout generator."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        weight_file = config.get("weight_file", None)
        self.device = config.get("device", "cpu")
        self.classes = list(config.data.classes)
        num_classes = len(config["data"]["classes"])
        self.model = build_network(num_classes, config, weight_file, device=self.device)
        self.model.eval()
        self.sample_num_points = config.network.sample_num_points
        self.point_dim = config.network.point_dim
        self.clip_denoised = config.clip_denoised
        self.up_axis = config.data.up_axis
        self.export_room_mask = config.export_room_mask
        self.room_layout_size = int(config.data.room_layout_size.split(",")[0])
        self.room_dims = config.data.room_dims
        self.bounds = config.data.bounds

    def prepare_inputs(self, layout_spec: SceneLayoutSpec) -> torch.FloatTensor:
        """
        Create room layout input which can be ingested by model.

        :param layout_spec: specification for layout architecture
        :return: tensor of mask of room, with 1 representing valid placement and 0 otherwise
        """
        room_mask = layout_spec.arch.get_room_mask(layout_size=self.room_layout_size, room_dims=self.room_dims, device="cpu")

        # write room_mask to image for debugging
        if self.export_room_mask:
            image = Image.fromarray((room_mask[0, 0].numpy() * 255).astype(np.uint8), "L")
            image.save("room_mask.png")

        return room_mask

    def generate(self, layout_spec: SceneLayoutSpec, **kwargs) -> list[dict[str, Any]]:
        room_mask = self.prepare_inputs(layout_spec)
        bbox_params = self.model.generate_layout(
            room_mask=room_mask,
            num_points=self.sample_num_points,
            point_dim=self.point_dim,
            text=layout_spec.raw,
            device=self.device,
            clip_denoised=self.clip_denoised,
        )

        boxes = self.descale_bbox_params(bbox_params)
        return self.prepare_outputs(boxes)

    def descale_bbox_params(self, s: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply descaling to bounding box predictions of model to original scale.

        The following operations are performed on each of the parameters:
        * room_layout - as is
        * class_labels - as is
        * relations - as is
        * description - as is
        * desc_emb - as is
        * angles - scaled using atan, treating tensor output axes as (batch, object, [x, y])
        * objfeats_32 (object shape codes) - scaled by bounds[1] and bounds[2]
        * objfeats - ignored (no bounds recorded for these)
        * everything else - scaled by bounds[0] and bounds[1]

        :param s: dictionary of outputs from model, pre-scaling
        :return: scaled outputs from model in the same form as the input
        """
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "room_layout" or k == "class_labels" or k == "relations" or k == "description" or k == "desc_emb":
                sample_params[k] = v
                
            elif k == "angles":
                # theta = arctan sin/cos y/x
                sample_params[k] = np.arctan2(v[:, :, 1:2], v[:, :, 0:1])
                
            elif k == "objfeats_32":
                sample_params[k] = descale(
                    v, np.asarray(bounds[k][1]), np.asarray(bounds[k][2])
                )
            
            elif k == "objfeats":
                print("[WARNING] objfeats does not have a corresponding normalization vector in dataset_stats.txt.")
                sample_params[k] = v
                
            else:
                sample_params[k] = descale(v, np.asarray(bounds[k][0]), np.asarray(bounds[k][1]))
        
        return sample_params

    def prepare_outputs(self, boxes: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        """
        Format outputs into standardized form.

        :param boxes: dictionary of the form
            {
                'class_labels': tensor of scores for classes for each object,
                'dimensions': tensor of dimensions for each object,
                'position': tensor of positions for each object,
                'orientation': tensor of angles to apply to each object,
            }
        :return: list of objects of the form
            {
                'wnsynsetkey': name of object class,
                'dimensions': dimensions of object as (x, y, z),
                'position': position of object in scene as (x, y, z),
                'orientation': rotation angle to apply to each object in radians,
            }
        """
        # flip axes if needed
        if self.up_axis == "y":  # assumes LHR, y-axis is vertical, and axes are rotated 180 deg
            axis_order = (0, 2, 1)
            boxes["translations"][0, :, :] = boxes["translations"][0, :, axis_order]
            boxes["translations"][0, :, 0] = -boxes["translations"][0, :, 0]
            boxes["angles"][0, :, :] -= torch.pi
            boxes["angles"][0, :, :] = torch.where(
                boxes["angles"][0, :, :] > -torch.pi, boxes["angles"][0, :, :], boxes["angles"][0, :, :] + 2 * torch.pi
            )
            # NOTE: we do not change the size of the boxes, because assets are mostly stored as vertical y-axis
        else:  # self.up_axis == "z"
            pass

        objects = []
        for i in range(boxes["class_labels"].shape[1]):  # start and end tokens not included
            class_idx = boxes["class_labels"][0, i].argmax(-1)
            objects.append(
                {
                    "wnsynsetkey": self.classes[class_idx],
                    "dimensions": boxes["sizes"][0, i, :].tolist(),
                    "position": boxes["translations"][0, i, :].tolist(),
                    "orientation": float(boxes["angles"][0, i, -1]),
                }
            )
        return objects
