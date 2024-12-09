import csv
import glob
import json
import os
import random
import sys
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional

import numpy as np
import pandas as pd
import pysolr
import requests
from easydict import EasyDict
from omegaconf import DictConfig

from libsg.scene_types import JSONDict, ObjectSpec


class AssetGroup:
    def __init__(self, id, metadata=None):
        self.id = id
        self.align_to = None
        self.scale_to = 1
        self.center_to = None
        if metadata:
            reader = csv.DictReader(open(metadata, "r"))
            self._metadata = {}
            for row in reader:
                id = row["id"]
                for k in row:
                    if k in ["maxX", "maxY", "maxZ", "minX", "minY", "minZ", "dimsX", "dimsY", "dimsZ"]:
                        row[k] = float(row[k])
                self._metadata[id] = row

    def set_align_to(self, up, front):
        self.align_to = {"up": up, "front": front}

    def model_info(self, model_id):
        if "." in model_id:
            tokens = model_id.split(".")
            source = tokens[0]
            id = tokens[1]
        else:
            source = self.id
            id = model_id
        model_info = {"source": source, "id": id}
        if self._metadata and id in self._metadata:
            model_info = self._metadata[id]
        return model_info

    def has_transforms(self) -> bool:
        return self.align_to or (self.scale_to is not None and self.scale_to != 1) or self.center_to

    def to_json(self, obj=None) -> JSONDict:
        obj = obj if obj else {}
        obj[self.id] = {"alignTo": self.align_to, "scaleTo": self.scale_to, "centerTo": self.center_to}
        return obj


class AssetDb:
    MAX_IDS = 1024

    def __init__(self, name, cfg, solr_url=None):
        self.name = name
        metadata_path = cfg.get("metadata")
        path = cfg.get("path")
        self.assets = {}
        if metadata_path:
            metadata = pd.read_csv(metadata_path)
            scene_ids = metadata["sceneId"]  # FIXME: this is not generalizable

            for s_id in scene_ids:
                self.assets[s_id] = path.format(scene_id=s_id)
        elif path:
            if not os.path.isdir(path):
                # TODO: use logger
                print(f"Warning: invalid path for assets {self.name}: {path}", file=sys.stderr)
            # TODO: this is specific for scenestates
            scenestates = glob.glob(path + "/*.json")
            for s in scenestates:
                base = os.path.basename(s)
                uuid = base.split(".")[0]
                self.assets[uuid] = s
        self.defaults = cfg.get("defaults", {})
        self.config = cfg
        # print(self.assets)
        self.__solr = pysolr.Solr(solr_url) if solr_url else None

    def get(self, id: Optional[str] = None):
        if id is not None:
            return self.assets.get(id)
        else:
            return random.choice(list(self.assets.values()))

    def search(self, *args, **kwargs):
        return self.__solr.search(*args, **kwargs)

    @property
    def default_up(self):
        return self.defaults.get("up")

    @property
    def default_front(self):
        return self.defaults.get("front")

    def _to_fullids(self, source: str, ids: list[str]):
        return [id if "." in id else f"{source}.{id}" for id in ids]

    def get_query_for_ids(self, ids: list[str]):
        # fullids = self._to_fullids(None, ids)
        return f'fullId:({" OR ".join(ids)})'

    @staticmethod
    def _to_floats(s, default=None):
        return [float(x) for x in s.split(",")] if s else default

    @staticmethod
    def _to_scaled_floats(s, scale, default=None):
        return [float(x) * scale for x in s.split(",")] if s else default

    @staticmethod
    def _to_lower(s, default=None):
        return [x.lower() for x in s] if s else default

    def _query_metadata(self, ids: list[str], fields: list[str] = None, **kwargs):
        if fields is None:
            postprocess = True
            fields = "fullId,wnsynsetkey,up,front,aligned.dims,category0"
        else:
            postprocess = False
            fields = ",".join(fields)

        query = self.get_query_for_ids(ids[:1024])
        results = self.__solr.search(query, fl=fields, **kwargs)
        converted = []
        if postprocess:
            if len(results):
                for result in results:
                    up = result.get("up")
                    up = AssetDb._to_floats(up, self.default_up)
                    front = result.get("front")
                    front = AssetDb._to_floats(front, self.default_front)
                    raw_dims = result.get("aligned.dims")
                    raw_dims = AssetDb._to_floats(raw_dims, None)
                    aligned_dims = result.get("aligned.dims")
                    aligned_dims = AssetDb._to_scaled_floats(aligned_dims, 1 / 100, None)
                    category0 = result.get("category0")
                    category0 = AssetDb._to_lower(category0, None)

                    converted.append(
                        EasyDict(
                            {
                                "fullId": result["fullId"],
                                "wnsynsetkey": result.get("wnsynsetkey", None),
                                "up": up,
                                "front": front,
                                "raw_dims": raw_dims,
                                "dims": aligned_dims,
                                "category0": category0,
                            }
                        )
                    )
            return converted
        else:
            return [dict(r) for r in results]

    def get_metadata(self, id: str):
        results = self._query_metadata([id])
        if len(results) == 1:
            return results[0]
        return results

    def get_metadata_for_ids(self, ids: list[str], fields: list[str] = None):
        response = []
        i = 0
        while i < len(ids):
            batch_ids = ids[i : i + self.MAX_IDS]
            response.extend(self._query_metadata(batch_ids, fields, rows=len(batch_ids)))
            i += self.MAX_IDS
        return response

    # sort assets by closeness to specified dimensions
    def sort_by_dim(self, ids: list[str], dims: list[float]):
        metadata = self.get_metadata_for_ids(ids)
        for m in metadata:
            if m["dims"] is None:
                m["dim_se"] = 0.0  # FIXME: some objects don't have a provided raw dimension
            else:
                m["dim_se"] = np.sum((np.asarray(dims) - np.asarray(m["dims"])) ** 2)
        metadata = sorted(metadata, key=lambda m: m["dim_se"])
        return metadata


@dataclass
class ThreedFutureAsset:
    """Implementation based on the ThreedFutureModel class of the DiffuScene codebase."""

    model_uid: str
    model_jid: str
    model_info: dict[str, Any]
    position: list[float]
    rotation: list[float]
    scale: list[float]
    path_to_models: str
    up: list[float] = field(default_factory=lambda: [0, 1, 0])
    front: list[float] = field(default_factory=lambda: [0, 0, 1])

    @property
    def raw_model_path(self):
        return os.path.join(self.path_to_models, self.model_jid, "raw_model.obj")

    # add normalized point cloud of raw_model
    @property
    def raw_model_norm_pc_path(self):
        return os.path.join(self.path_to_models, self.model_jid, "raw_model_norm_pc.npz")

    @property
    def raw_model_norm_pc_lat_path(self):
        return os.path.join(self.path_to_models, self.model_jid, "raw_model_norm_pc_lat32.npz")

    @property
    def raw_model_norm_pc_lat32_path(self):
        return os.path.join(self.path_to_models, self.model_jid, "raw_model_norm_pc_lat32.npz")

    @property
    def texture_image_path(self):
        return os.path.join(self.path_to_models, self.model_jid, "texture.png")

    @property
    def path_to_bbox_vertices(self):
        return os.path.join(self.path_to_models, self.model_jid, "bbox_vertices.npy")

    # add normalized point cloud of raw_model
    def raw_model_norm_pc(self):
        points = np.load(self.raw_model_norm_pc_path)["points"].astype(np.float32)
        return points

    def raw_model_norm_pc_lat(self):
        latent = np.load(self.raw_model_norm_pc_lat_path)["latent"].astype(np.float32)
        return latent

    def raw_model_norm_pc_lat32(self):
        latent = np.load(self.raw_model_norm_pc_lat32_path)["latent"].astype(np.float32)
        return latent

    def centroid(self, offset=[[0, 0, 0]]):
        return self.corners(offset).mean(axis=0)

    @cached_property
    def size(self):
        corners = self.corners()
        return np.array(
            [
                np.sqrt(np.sum((corners[4] - corners[0]) ** 2)) / 2,
                np.sqrt(np.sum((corners[2] - corners[0]) ** 2)) / 2,
                np.sqrt(np.sum((corners[1] - corners[0]) ** 2)) / 2,
            ]
        )

    def bottom_center(self, offset=[[0, 0, 0]]):
        centroid = self.centroid(offset)
        size = self.size
        return np.array([centroid[0], centroid[1] - size[1], centroid[2]])

    @cached_property
    def bottom_size(self):
        return self.size * [1, 2, 1]

    @cached_property
    def z_angle(self):
        # See BaseThreedFutureModel._transform for the origin of the following
        # code.
        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:])) * 2

        if np.sum(axis) == 0 or np.isnan(theta):
            return 0

        assert np.dot(axis, [1, 0, 1]) == 0
        assert 0 <= theta <= 2 * np.pi

        if theta >= np.pi:
            theta = theta - 2 * np.pi

        return np.sign(axis[1]) * theta

    @cached_property
    def label(self):
        return self.model_info["category"]

    def corners(self, offset=[[0, 0, 0]]):
        try:
            bbox_vertices = np.load(self.path_to_bbox_vertices, mmap_mode="r")
        except:
            bbox_vertices = np.array(self.raw_model().bounding_box.vertices)
            np.save(self.path_to_bbox_vertices, bbox_vertices)
        c = self._transform(bbox_vertices)
        return c + offset

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def int_label(self, all_labels):
        return all_labels.index(self.label)


class ThreedFutureAssetDB:
    def __init__(self, cfg: DictConfig):
        self.use_object_class = cfg.get("use_object_class", False)
        self.use_object_size = cfg.get("use_object_size", False)
        self.datasets = {}
        for dataset_name, dataset_path in cfg.datasets.items():
            with open(dataset_path, "r") as f:
                dataset = json.load(f)

            self.datasets[dataset_name] = {}
            for obj in dataset:
                self.datasets[dataset_name][obj["model_jid"]] = ThreedFutureAsset(**obj)

    def get(self, id: str, *, dataset_name: str):
        return self.datasets[dataset_name][id]

    def retrieve_by_embedding(
        self,
        object_spec: ObjectSpec,
        *,
        dataset_name: str,
    ) -> ThreedFutureAsset:
        # TODO: we could use object class as well theoretically
        model_ids = []
        mses_feat = []
        mses_size = []
        for model_id, obj in self.datasets[dataset_name].items():
            if not self.use_object_class or object_spec.description in obj.label:
                model_ids.append(model_id)  # FIXME: pretty hacky whoops
                if object_spec.embedding.shape[0] == 32:  # use objfeats_32
                    mses_feat.append(np.sum((obj.raw_model_norm_pc_lat32() - object_spec.embedding) ** 2, axis=-1))
                else:  # use objfeats
                    mses_feat.append(np.sum((obj.raw_model_norm_pc_lat() - object_spec.embedding) ** 2, axis=-1))

                if self.use_object_size and object_spec.dimensions is not None:
                    mses_size.append(np.sum((obj.size - object_spec.dimensions) ** 2, axis=-1))

        if not model_ids:
            raise ValueError(f"No models found for the given object class: {object_spec.description}")

        if mses_size:
            ind = np.lexsort((mses_feat, mses_size))
        else:
            ind = np.argsort(mses_feat)
        return self.datasets[dataset_name][model_ids[ind[0]]]  # TODO: add some randomness here, maybe within closest 5
