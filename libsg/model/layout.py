"""
layout.py
---
Classes to define unified wrapper interface for each layout model.

TODO: This could eventually be split out into multiple files if we add more than 2 model architectures.
"""

import pickle
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from diffusers.training_utils import EMAModel

from libsg.model.atiss import atiss_network
from libsg.model.diffuscene import build_network
from libsg.model.instructscene import ObjectFeatureVQVAE, load_checkpoints, Sg2ScDiffusion
from libsg.model.instructscene.clip_encoders import OpenShapeTextEncoder
from libsg.scene_types import SceneLayoutSpec
from libsg.model.utils import descale, ObjectClassNotFoundError, EdgeRelationshipNotFoundError


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
        self.raw_classes = list(config.data.raw_classes)
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
        room_mask = layout_spec.arch.get_room_mask(
            layout_size=self.room_layout_size, room_dims=self.room_dims, device="cpu"
        )

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
                boxes["angles"][0, :, :] > -torch.pi,
                boxes["angles"][0, :, :],
                boxes["angles"][0, :, :] + 2 * torch.pi,
            )
            # NOTE: we do not change the size of the boxes, because assets are mostly stored as vertical y-axis
        else:  # self.up_axis == "z"
            pass

        # reshape outputs
        for i in range(1, boxes["class_labels"].shape[1] - 1):
            class_idx = boxes["class_labels"][0, i].argmax(-1)
            objects.append(
                {
                    "wnsynsetkey": self.classes[class_idx],
                    "class": self.raw_classes[class_idx],
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
        self.raw_classes = list(config.data.raw_classes)
        num_classes = len(config.data.classes)
        self.model = build_network(num_classes, config, weight_file, device=self.device)
        self.model.eval()
        self.sample_num_points = config.network.sample_num_points
        self.point_dim = config.network.point_dim
        self.clip_denoised = config.clip_denoised
        self.up_axis = config.data.up_axis
        self.export_room_mask = config.export_room_mask
        self.export_embeddings = config.export_embeddings
        self.room_layout_size = int(config.data.room_layout_size.split(",")[0])
        self.room_dims = config.data.room_dims
        self.bounds = config.data.bounds

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

            elif k in {"objfeats", "objfeats_32"}:
                # the dataset_stats.txt file only has bounds for objfeats_32, but the original dataset_stats.txt
                # files generated by DiffuScene have the same bounds for both.
                sample_params[k] = descale(
                    v,
                    np.asarray(bounds["objfeats_32"][1]),
                    np.asarray(bounds["objfeats_32"][2]),
                )

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
                boxes["angles"][0, :, :] > -torch.pi,
                boxes["angles"][0, :, :],
                boxes["angles"][0, :, :] + 2 * torch.pi,
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
                    "class": self.raw_classes[class_idx],
                    "dimensions": boxes["sizes"][0, i, :].tolist(),
                    "position": boxes["translations"][0, i, :].tolist(),
                    "orientation": float(boxes["angles"][0, i, -1]),
                    "embedding": boxes["objfeats"][0, i, :].cpu().numpy() if self.export_embeddings else None,
                }
            )
        return objects


class InstructScene(LayoutBase):
    """InstructScene model wrapper"""

    # reverse index lookup for InstructScene object classes
    OBJECT_LOOKUP = {
        "bedroom": {
            "armchair": 0,
            "bookcase": 1,
            "cabinet": 2,
            "ceiling lamp": 3,
            "chair": 4,
            "children cabinet": 5,
            "coffee table": 6,
            "desk": 7,
            "double bed": 8,
            "bed": 8,
            "dressing chair": 9,
            "dressing table": 10,
            "kids bed": 11,
            "nightstand": 12,
            "corner/side table": 12,
            "round end table": 12,
            "pendant lamp": 13,
            "shelf": 14,
            "single bed": 15,
            "sofa": 16,
            "multi-seat sofa": 16,
            "stool": 17,
            "table": 18,
            "tv stand": 19,
            "wardrobe": 20,
        },
        "diningroom": {
            "armchair": 0,
            "bookcase": 1,
            "cabinet": 2,
            "ceiling lamp": 3,
            "chaise longue sofa": 4,
            "chinese chair": 5,
            "coffee table": 6,
            "console table": 7,
            "corner/side table": 8,
            "desk": 9,
            "dining chair": 10,
            "dining table": 11,
            "l-shaped sofa": 12,
            "lazy sofa": 13,
            "lounge chair": 14,
            "loveseat sofa": 15,
            "multi-seat sofa": 16,
            "pendant lamp": 17,
            "round end table": 18,
            "shelf": 19,
            "stool": 20,
            "tv stand": 21,
            "wardrobe": 22,
            "wine cabinet": 23,
        },
        "livingroom": {
            "armchair": 0,
            "bookcase": 1,
            "cabinet": 2,
            "ceiling lamp": 3,
            "chaise longue sofa": 4,
            "chinese chair": 5,
            "coffee table": 6,
            "console table": 7,
            "corner/side table": 8,
            "desk": 9,
            "dining chair": 10,
            "dining table": 11,
            "l-shaped sofa": 12,
            "lazy sofa": 13,
            "lounge chair": 14,
            "loveseat sofa": 15,
            "multi-seat sofa": 16,
            "pendant lamp": 17,
            "round end table": 18,
            "shelf": 19,
            "stool": 20,
            "tv stand": 21,
            "wardrobe": 22,
            "wine cabinet": 23,
        },
    }

    def __init__(self, config: DictConfig):
        super().__init__()
        self.classes = list(config.data.classes)
        self.raw_classes = list(config.data.raw_classes)
        self.bounds = config.data.bounds
        self.room_layout_size = int(config.data.room_layout_size.split(",")[0])
        self.room_dims = config.data.room_dims
        self.up_axis = config.data.up_axis
        self.export_room_mask = config.export_room_mask
        self.device = config.get("device", "cpu")
        self.cfg_scale = config.network.cfg_scale
        self.max_length = config.data.max_length
        self.predicate_types = config.data.predicate_types
        self.reverse_predicates = config.data.reverse_predicates

        # instantiate model
        print("Load pretrained VQ-VAE")
        with open(config.network.objfeat_bounds, "rb") as f:
            kwargs = pickle.load(f)
        self.vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
        self.vqvae_model.load_state_dict(torch.load(config.network.vqvae_checkpoint, map_location="cpu")["model"])
        self.vqvae_model = self.vqvae_model.to(self.device)
        self.vqvae_model.eval()

        # Initialize the model
        self.text_encoder = OpenShapeTextEncoder(config.network.text_encoder, device=self.device)
        self.sg_to_scene_model = self._load_sg_to_scene_model(config)

    def _load_sg_to_scene_model(self, config):
        # Initialize the model
        model = Sg2ScDiffusion(
            len(self.raw_classes),
            len(self.predicate_types),
            use_objfeat="objfeat" in config.network.name,
        ).to(self.device)

        # Create EMA for the model
        ema_config = config["training"]["ema"]
        if ema_config["use_ema"]:
            ema_states = EMAModel(model.parameters())
            ema_states.to(self.device)
        else:
            ema_states: EMAModel = None

        # Load the weights from a checkpoint
        load_epoch = load_checkpoints(
            model,
            config.network.ckpt_dir,
            ema_states,
            epoch=config.network.ckpt_epoch,
            device=self.device,
        )

        # Evaluate with the EMA parameters if specified
        if ema_states is not None:
            print(f"Copy EMA parameters to the model")
            ema_states.copy_to(model.parameters())
        model.eval()
        return model

    def prepare_inputs(self, layout_spec: SceneLayoutSpec) -> torch.FloatTensor:
        """
        Create scene layout specification input which can be ingested by model.

        :param layout_spec: specification for layout architecture
        :return: tensor of mask of room, with 1 representing valid placement and 0 otherwise
        """
        scene_graph = layout_spec.graph

        # parse objects from scene graph
        try:
            objs = [InstructScene.OBJECT_LOOKUP[scene_graph["room_type"]][ob["name"]] for ob in scene_graph["objects"]]
        except KeyError as e:
            raise ObjectClassNotFoundError(f"Tried to retrieve class index for object but could not parse: {e}")
        while len(objs) < self.max_length:  # pad to max length
            objs.append(len(self.raw_classes))
        objs = torch.tensor(objs)  # (max_length,)
        objs = objs.unsqueeze(0)  # (bs, max_length)
        obj_mask = (objs != len(self.raw_classes)).long()  # (bs, max_length)

        # parse edges/relationships from scene graph
        edges = len(self.predicate_types) * torch.ones((self.max_length, self.max_length), dtype=torch.int64)  # (n, n)
        for rel in scene_graph["relationships"]:
            subject_id = rel["subject_id"] - 1  # originally 1-indexed
            target_id = rel["target_id"] - 1

            try:
                edges[subject_id, target_id] = self.predicate_types.index(rel["type"])
                edges[target_id, subject_id] = self.predicate_types.index(self.reverse_predicates[rel["type"]])
            except ValueError as e:
                raise EdgeRelationshipNotFoundError(f"Could not parse edge relationship type: {rel['type']}")

        # quantized objfeats
        # the original InstructScene uses a diffusion-based model to generate the scene graph. Here, we use an LLM
        # instead and apply CLIP to each attr+object to generate the objfeat features.
        features = [
            (i, obj["feature"]) for i, obj in enumerate(scene_graph["objects"]) if obj.get("feature") is not None
        ]
        if features:
            objfeat_vq_indices = torch.randint(0, 64, size=(self.max_length, features[0][1].size(0)))
            for idx, feat in features:
                objfeat_vq_indices[idx] = feat
        else:
            objfeat_vq_indices = None

        missing_indices = []
        texts = []
        for i, obj in enumerate(scene_graph["objects"]):
            if obj.get("feature") is None:
                missing_indices.append(i)
                if obj.get("attributes", []):
                    texts.append(", ".join(obj["attributes"]) + " " + obj["name"])
                else:
                    texts.append(obj["name"])

        # if raw text is provided and obj feature is not already, encode text using CLIP model and then quantize
        if texts:
            objfeats = self.text_encoder(texts)
            new_objfeats_vq = self.vqvae_model.quantize_to_indices(objfeats)
            if objfeat_vq_indices is None:
                objfeat_vq_indices = torch.randint(0, 64, size=(self.max_length, new_objfeats_vq.size(1)))
            for idx in range(len(texts)):
                objfeat_vq_indices[missing_indices[idx]] = new_objfeats_vq[idx]

        objfeat_vq_indices = objfeat_vq_indices.unsqueeze(0)  # (bs, n, d)

        return objs, edges, objfeat_vq_indices, obj_mask

    def generate(self, layout_spec: SceneLayoutSpec, **kwargs) -> list[dict[str, Any]]:
        # Unpack the batch parameters
        objs, edges, objfeat_vq_indices, obj_masks = self.prepare_inputs(layout_spec)
        # Generate the box parameters
        with torch.no_grad():
            boxes_pred = self.sg_to_scene_model.generate_samples(
                objs,
                edges,
                objfeat_vq_indices,
                obj_masks,
                self.vqvae_model,
                cfg_scale=self.cfg_scale,
            )

        objs = objs.cpu()
        obj_masks = obj_masks.cpu()
        boxes_pred = boxes_pred.cpu()  # (B, N, 8)

        bbox_params = {
            "class_labels": F.one_hot(
                objs, num_classes=len(self.raw_classes) + 1
            ).float(),  # +1 for empty node (not really used)
            "translations": boxes_pred[..., :3],
            "sizes": boxes_pred[..., 3:6],
            "angles": boxes_pred[..., 6:],
        }

        boxes = self.descale_bbox_params(bbox_params)
        return self.prepare_outputs(boxes, objfeat_vq_indices, obj_masks)

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
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "room_layout" or k == "class_labels" or k == "relations" or k == "description" or k == "desc_emb":
                sample_params[k] = v

            elif k == "angles":
                # theta = arctan sin/cos y/x
                sample_params[k] = np.arctan2(v[:, :, 1:2], v[:, :, 0:1])

            else:
                sample_params[k] = descale(v, np.asarray(bounds[k][0]), np.asarray(bounds[k][1]))

        return sample_params

    def prepare_outputs(self, boxes: dict[str, torch.Tensor], objfeat_vq_indices, obj_masks) -> list[dict[str, Any]]:
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
                boxes["angles"][0, :, :] > -torch.pi,
                boxes["angles"][0, :, :],
                boxes["angles"][0, :, :] + 2 * torch.pi,
            )
            # NOTE: we do not change the size of the boxes, because assets are mostly stored as vertical y-axis
        else:  # self.up_axis == "z"
            pass

        # reconstruct objfeat embeddings
        B, N = objfeat_vq_indices.shape[:2]
        objfeats = self.vqvae_model.reconstruct_from_indices(objfeat_vq_indices.reshape(B * N, -1)).reshape(B, N, -1)
        objfeats = (objfeats * obj_masks[..., None].float()).cpu().numpy()

        objects = []
        for i in range(boxes["class_labels"].shape[1]):  # start and end tokens not included
            class_idx = boxes["class_labels"][0, i].argmax(-1)
            if class_idx >= len(self.raw_classes):  # empty object
                continue
            objects.append(
                {
                    "wnsynsetkey": self.classes[class_idx],
                    "class": self.raw_classes[class_idx],
                    "dimensions": boxes["sizes"][0, i, :].tolist(),
                    "position": boxes["translations"][0, i, :].tolist(),
                    "orientation": float(boxes["angles"][0, i, -1]),
                    "embedding": objfeats[0, i, :],
                }
            )
        return objects
