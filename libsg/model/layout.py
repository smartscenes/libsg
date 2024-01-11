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

from libsg.model.atiss import atiss_network, descale_bbox_params
from libsg.model.diffuscene import build_network
from libsg.model.utils import square_room_mask
from libsg.scene_types import SceneLayoutSpec


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
        room_mask = layout_spec.arch.get_room_mask(room_dim=self.room_layout_size)

        # write room_mask to image for debugging
        if self.export_room_mask:
            image = Image.fromarray((room_mask[0, 0].numpy() * 255).astype(np.uint8), "L")
            image.save("room_mask.png")

        return room_mask

    def generate(self, layout_spec: SceneLayoutSpec, **kwargs) -> list[dict[str, Any]]:
        room_mask = self.prepare_inputs(layout_spec)
        bbox_params = self.model.generate_boxes(room_mask=room_mask)
        boxes = descale_bbox_params(self.bounds, bbox_params)
        return self.prepare_outputs(boxes)

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
        if self.up_axis == "y":
            axis_order = (0, 2, 1)
            boxes["sizes"][0, :, :] = boxes["sizes"][0, :, axis_order]
            boxes["translations"][0, :, :] = boxes["translations"][0, :, axis_order]
        elif self.up_axis == "x":
            axis_order = (2, 1, 0)
            boxes["sizes"][0, :, :] = boxes["sizes"][0, :, axis_order]
            boxes["translations"][0, :, :] = boxes["translations"][0, :, axis_order]
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
        self.bounds = config.data.bounds

    def prepare_inputs(self, layout_spec: SceneLayoutSpec) -> torch.FloatTensor:
        """
        Create room layout input which can be ingested by model.

        :param layout_spec: specification for layout architecture
        :return: tensor of mask of room, with 1 representing valid placement and 0 otherwise
        """
        room_mask = layout_spec.arch.get_room_mask(room_dim=self.room_layout_size)

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
            text=None,
            device=self.device,
            clip_denoised=self.clip_denoised,
        )

        # the original code includes a dataset.post_process(bbox_params), but it appears to just be the identity
        # function
        boxes = descale_bbox_params(self.bounds, bbox_params)
        return self.prepare_outputs(boxes)

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
        if self.up_axis == "y":
            axis_order = (0, 2, 1)
            boxes["sizes"][0, :, :] = boxes["sizes"][0, :, axis_order]
            boxes["translations"][0, :, :] = boxes["translations"][0, :, axis_order]
        elif self.up_axis == "x":
            axis_order = (2, 1, 0)
            boxes["sizes"][0, :, :] = boxes["sizes"][0, :, axis_order]
            boxes["translations"][0, :, :] = boxes["translations"][0, :, axis_order]
        else:  # self.up_axis == "z"
            pass

        objects = []
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
