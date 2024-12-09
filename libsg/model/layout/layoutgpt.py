"""
layoutgpt.py
---
Layout implementation for LayoutGPT.
"""

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from libsg.model.layoutgpt.model import LayoutGPT as LayoutGPTModel
from libsg.scene_types import SceneLayoutSpec
from libsg.model.utils import descale
from .base import BaseLayout


class LayoutGPT(BaseLayout):
    """ATISS model wrapper"""

    SHIFT_BY_SCENE_CENTROID = True

    def __init__(self, config: DictConfig):
        super().__init__()
        self.classes = list(config.data.classes)
        self.raw_classes = list(config.data.raw_classes)
        self.object_types = list(config.data.object_types)
        self.bounds = config.data.bounds
        self.room_layout_size = int(config.data.room_layout_size.split(",")[0])
        self.room_dims = config.data.room_dims
        self.up_axis = config.data.up_axis
        self.export_room_mask = config.export_room_mask
        self.model = LayoutGPTModel(**config.network)
        self.device = "cpu"

    def prepare_inputs(self, layout_spec: SceneLayoutSpec) -> torch.FloatTensor:
        """
        Create room layout input which can be ingested by model.

        :param layout_spec: specification for layout architecture
        :return: tensor of mask of room, with 1 representing valid placement and 0 otherwise
        """
        room_mask = layout_spec.arch.get_room_mask(
            layout_size=self.room_layout_size, room_dims=self.room_dims, device="cpu"
        )

        # write room_mask to image for debugging
        if self.export_room_mask:
            image = Image.fromarray((room_mask[0, 0].cpu().numpy() * 255).astype(np.uint8), "L")
            image.save("room_mask.png")

        return room_mask

    def generate(self, layout_spec: SceneLayoutSpec, **kwargs) -> list[dict[str, Any]]:
        room_mask = self.prepare_inputs(layout_spec)
        bbox_params = self.model.generate_boxes(layout_spec.raw, room_mask=room_mask)
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

        :param s: dictionary of outputs from model, pre-scaling, with the following keys:
        {
            'class_labels': tensor of size [batch_size, num_objects, num_classes] representing a one-hot encoding of object classes,
            'translations': tensor of size [batch_size, num_objects, 3] representing translation vector for each object,
            'sizes': tensor of size [batch_size, num_objects, 3] representing the sizes of each object,
            'angles': tensor of size [batch_size, num_objects, 1] representing the angles of each object,
        }
        :return: scaled outputs from model in the same form as the input
        """
        sample_params = {}
        for k, v in s.items():
            if k == "room_layout" or k == "class_labels" or k == "objfeats":
                sample_params[k] = v
            elif k == "angles":
                sample_params[k] = v * np.pi / 180
            elif k == "sizes":
                sample_params[k] = v * self.room_dims[0] / self.room_layout_size
            elif k == "translations":
                sample_params[k] = v * self.room_dims[0] / self.room_layout_size - torch.tensor(
                    [[[self.room_dims[0] / 2, self.room_dims[1] / 2, 0]]]
                )
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

        # reshape outputs
        for i in range(len(boxes["class_labels"][0])):
            try:
                class_idx = self.object_types.index(boxes["class_labels"][0][i])
                wnsynsetkey = self.classes[class_idx]
                raw_class = self.raw_classes[class_idx]
            except ValueError:
                wnsynsetkey = None
                raw_class = None
            objects.append(
                {
                    "wnsynsetkey": wnsynsetkey,
                    "class": raw_class,
                    "dimensions": boxes["sizes"][0, i, :].tolist(),
                    "position": boxes["translations"][0, i, :].tolist(),
                    "orientation": float(boxes["angles"][0, i]),
                }
            )
        return objects
