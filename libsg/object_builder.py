"""
object_builder.py
---
This module is responsible for generating and retrieving objects for scenes.
"""

import copy
import logging
import os
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Optional, Union

import numpy as np
import torch
from filelock import FileLock
from omegaconf import DictConfig
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
from shap_e.models.transmitter.base import Transmitter, VectorDecoder
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.rendering.torch_mesh import TorchMesh
from shap_e.util.collections import AttrDict

from libsg.assets import AssetDb
from libsg.model.instructscene.clip_encoders import CLIPTextEncoderWrapper
from libsg.scene import ModelInstance
from libsg.scene_types import ObjectSpec, PlacementSpec, Point3D


@dataclass
class Placement:
    """Dataclass for object and its placement in scene"""

    position: Point3D = None
    up: Point3D = None
    front: Point3D = None
    ref_object: Union[ModelInstance, str] = None
    spec: PlacementSpec = None


class ObjectBuilder:
    """
    Build or retrieve objects for scenes.
    """

    DEFAULT_UP = [0, 0, 1]
    DEFAULT_FRONT = [0, 1, 0]
    DEFAULT_ROWS = 25000
    LAST_ARCH = None

    def __init__(self, model_db: AssetDb, cfg: DictConfig, **kwargs):
        self.cfg = cfg
        self.__model_db = model_db

        # object retrieval parameters
        self.size_threshold = cfg.get("size_threshold", 0.5)

        # object generation parameters
        if torch.cuda.is_available():
            logging.debug("Using CUDA")
            self.device = torch.device("cuda")
        else:
            logging.debug("Using CPU")
            self.device = torch.device("cpu")

        self.batch_size = cfg.generation.get("batch_size", 1)
        self.guidance_scale = cfg.generation.get("guidance_scale", 15.0)
        self.gen_output_dir = cfg.generation.get("output_dir")
        self.gen_metadata = cfg.generation.get("metadata_file")

        self.ret_embedding_field = cfg.retrieval.embedding_field
        self.ret_top_k = cfg.retrieval.top_k

        if self.gen_output_dir:
            os.makedirs(self.gen_output_dir, exist_ok=True)

        self._text_encoder = None

        sources_raw = kwargs.get("sceneInference.assetSources")
        if sources_raw:
            self.sources = sources_raw.split(",")
        else:
            self.sources = None
        use_wnsynset = kwargs.get("sceneInference.useCategory", "false").lower()
        if use_wnsynset not in {"true", "false"}:
            raise ValueError(f"Invalid value for sceneInference.useCategory: {use_wnsynset}")
        self.use_wnsynset = use_wnsynset == "true"

    @cached_property
    def generation_text_model(self):
        return load_model("text300M", device=self.device)

    @cached_property
    def generation_xm_model(self):
        return load_model("transmitter", device=self.device)

    @cached_property
    def generation_diffusion_model(self):
        return diffusion_from_config(load_config("diffusion"))

    @property
    def text_encoder(self):
        return CLIPTextEncoderWrapper.get_model(name=self.cfg.text_encoder, device=self.device)

    @staticmethod
    def get_source(gen_method, retrieve_type):
        if gen_method == "generate":
            return "t2sModel"
        else:
            return "fpModel"

    def generate(self, object_spec: ObjectSpec, placement_spec: PlacementSpec, **kwargs) -> ModelInstance:
        """
        Generate a new 3D object based on the given specifications.

        Args:
            object_spec (ObjectSpec): Specification of the object to generate.
            placement_spec (PlacementSpec): Specification of where to place the object.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelInstance: The generated 3D object.

        Raises:
            AssertionError: If the generation output directory is not set.
        """
        assert self.gen_output_dir is not None, "Generation output directory must be set to generate objects"

        # TODO: move this out into a separate model file for generation
        latents = sample_latents(
            batch_size=self.batch_size,
            model=self.generation_text_model,
            diffusion=self.generation_diffusion_model,
            guidance_scale=self.guidance_scale,
            model_kwargs=dict(texts=[object_spec.description] * self.batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        assert len(latents) >= 1
        for i, latent in enumerate(latents):
            model_id = str(uuid.uuid4())
            t = self._decode_latent_mesh(self.generation_xm_model, latent).tri_mesh()

            # move origin to bottom of model
            t.verts[:, 2] -= np.min(t.verts[:, 2])

            # # rescale object to size
            # if object_spec.dimensions:
            #     # normalize to [0, 1]
            #     verts_norm = (t.verts - np.min(t.verts, axis=0)) / (np.max(t.verts, axis=0) - np.min(t.verts, axis=0))

            #     # scale to object dimensions
            #     t.verts = verts_norm * np.expand_dims(np.array(object_spec.dimensions), axis=0)

            #     # recenter object
            #     t.verts[:, :2] -= np.mean(t.verts[:, :2], axis=0)

            os.makedirs(os.path.join(self.gen_output_dir, model_id), exist_ok=True)
            with open(os.path.join(self.gen_output_dir, model_id, "raw_model.obj"), "w") as f:
                t.write_obj(f)

            print(f"Writing new object {model_id} metadata ('{object_spec.description}') to {self.gen_metadata}...")
            lock = FileLock(f"{self.gen_metadata}.lock")
            with lock:
                with open(self.gen_metadata, "a") as f:
                    f.write(f'{model_id},"{object_spec.description}","{datetime.now()}"\n')

        # FIXME: there is probably a better way to add the prefix source
        return ModelInstance(model_id=f"t2sModel.{model_id}", up=[0, 0, 1], front=[0, 1, 0])

    def _create_pan_cameras(self, size: int, device: torch.device) -> DifferentiableCameraBatch:
        origins = []
        xs = []
        ys = []
        zs = []
        for theta in np.linspace(0, 2 * np.pi, num=20):
            z = np.array([np.sin(theta), np.cos(theta), -0.5])
            z /= np.sqrt(np.sum(z**2))
            origin = -z * 4
            x = np.array([np.cos(theta), -np.sin(theta), 0.0])
            y = np.cross(z, x)
            origins.append(origin)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return DifferentiableCameraBatch(
            shape=(1, len(xs)),
            flat_camera=DifferentiableProjectiveCamera(
                origin=torch.from_numpy(np.stack(origins, axis=0)).float().to(device),
                x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
                y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
                z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
                width=size,
                height=size,
                x_fov=0.7,
                y_fov=0.7,
            ),
        )

    @torch.no_grad()
    def _decode_latent_mesh(self, xm: Transmitter | VectorDecoder, latent: torch.Tensor) -> TorchMesh:
        decoded = xm.renderer.render_views(
            AttrDict(cameras=self._create_pan_cameras(2, latent.device)),  # lowest resolution possible
            params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(latent[None]),
            options=AttrDict(rendering_mode="stf", render_with_direction=False),
        )
        return decoded.raw_meshes[0]

    def modify(self, arch: ModelInstance, modify_spec: ObjectSpec) -> ModelInstance:
        pass

    def retrieve(
        self,
        object_spec: ObjectSpec,
        placement_spec: Optional[PlacementSpec] = None,
        *,
        dataset_name: Optional[str] = None,  # originally for 3D-FUTURE
        max_retrieve: int = 0,
        constraints: str = "",
        filter_parts: bool = True,
        **kwargs,
    ) -> ModelInstance:
        """
        Retrieve an object from the database based on the given specifications.

        Args:
            object_spec (ObjectSpec): Specification of the object to retrieve.
            placement_spec (Optional[PlacementSpec]): Specification for object placement.
            dataset_name (Optional[str]): Name of the dataset to retrieve from.
            max_retrieve (int): Maximum number of objects to retrieve.
            constraints (str): Additional constraints for retrieval.
            filter_parts (bool): Whether to filter out object parts.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelInstance: The retrieved object.

        Raises:
            ValueError: If no matching object is found.

        FIXME: in practice would prefer not to pass the dataset_name, but needed for diffuscene 3D-FUTURE assets for
        now.
        """

        def parse_response(selected_metadata):
            if max_retrieve > 0:
                results = []
                for entry in selected_metadata:
                    model_id = entry["fullId"]
                    up = entry["up"]
                    front = entry["front"]
                    results.append(ModelInstance(model_id=model_id, up=up, front=front))
                return results
            else:
                model_id = selected_metadata["fullId"]
                up = selected_metadata["up"]
                front = selected_metadata["front"]
                return ModelInstance(model_id=model_id, up=up, front=front)

        match object_spec.type:
            case "model_id":
                model_id = object_spec.description
                return ModelInstance(model_id=model_id)

            case "category":  # use solr to search based on wnsynsetkey
                query = object_spec.description if object_spec.description else "*"
                if constraints:  # TODO: check if this is correct
                    query = f"{query} AND {constraints}"
                fq = self._build_object_query(object_spec, placement_spec, constraints=constraints)
                results = self.__model_db.search(query, fl="fullId", fq=fq, rows=ObjectBuilder.DEFAULT_ROWS)

                # ignore parts
                if filter_parts:
                    results = filter(lambda r: "_part_" not in r["fullId"], results)

                model_ids = list([result["fullId"] for result in results])
                if len(model_ids) == 0:
                    raise ValueError(f'Cannot find object matching query "{query}" with filter "{fq}"')

                selected_metadata = self._filter_by_size(model_ids, object_spec, max_retrieve=max_retrieve)
                return parse_response(selected_metadata)

            case "embedding":
                if object_spec.embedding is None:
                    desc = object_spec.description
                    object_spec.embedding = self.get_text_embedding(desc)
                if constraints:
                    constraints = " ".join([f"preFilter={c}" for c in constraints.split("AND")])
                else:
                    constraints = 'preFilter=""'
                query = f"{{!knn f={self.ret_embedding_field} topK={max(max_retrieve, self.ret_top_k)} {constraints}}}"
                fq = self._build_object_query(object_spec, placement_spec)
                embedding = object_spec.embedding.tolist()
                results = self.__model_db.search(
                    query + str(embedding),
                    fl="fullId",
                    fq=fq,
                    rows=ObjectBuilder.DEFAULT_ROWS,
                )

                # ignore parts
                if filter_parts:
                    results = filter(lambda r: "_part_" not in r["fullId"], results)

                model_ids = list([result["fullId"] for result in results])
                if len(model_ids) == 0:
                    raise ValueError(f'Cannot find object matching query "{query}" with filter "{fq}"')

                selected_metadata = self._filter_by_size(model_ids, object_spec, max_retrieve=max_retrieve)
                return parse_response(selected_metadata)

                # deprecated code to use internal 3D-FUTURE assets
                # model = self.__threed_future_db.retrieve_by_embedding(object_spec, dataset_name=dataset_name)
                # model_instance = ModelInstance(model_id=f"3dfModel.{model.model_jid}", up=model.up, front=model.front)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        return self.text_encoder(text)[0]

    # construct object solr query constraints (source, synset, support etc.)
    def _build_object_query(
        self, object_spec: ObjectSpec, placement_spec: Optional[PlacementSpec] = None, constraints: str = ""
    ):
        """Construct query constraints for solr query into object database"""

        fq = ""
        if self.sources:
            fq += "+(" + " OR ".join([f"source:{source}" for source in self.sources]) + ")"
        if self.use_wnsynset and object_spec.wnsynsetkey:
            fq += f" +wnhypersynsetkeys:{object_spec.wnsynsetkey}"
        if not placement_spec:
            return fq
        reference_object = PlacementSpec.get_placement_reference_object(placement_spec)
        if reference_object:
            match reference_object.description.lower():
                case "wall":
                    fq = fq + " +support:vertical"
                case "ceiling":
                    fq = fq + " +support:top"
                case _:
                    fq = fq + " -support:top -support:vertical"
        if constraints:
            fq += f" {constraints}"
        return fq

    def _filter_by_size(self, model_ids, object_spec: ObjectSpec, max_retrieve: int = 0):
        # filter by size
        if object_spec.dimensions:
            dim_sorted_models = self.__model_db.sort_by_dim(model_ids, object_spec.dimensions)
            total = sum(object_spec.dimensions) ** 2
            trunc_sorted = [m for m in dim_sorted_models if m["dim_se"] < self.size_threshold * total]
            if len(trunc_sorted) > 0:
                if max_retrieve > 0:
                    selected_metadata = np.random.choice(
                        trunc_sorted, size=min(max_retrieve, len(trunc_sorted)), replace=False
                    )
                else:
                    selected_metadata = random.choice(trunc_sorted)
            else:
                logging.warning(
                    f"No models found within size threshold for object category {object_spec.wnsynsetkey} with size {object_spec.dimensions}",
                )
                if max_retrieve > 0:
                    selected_metadata = dim_sorted_models[:max_retrieve]
                else:
                    selected_metadata = dim_sorted_models[0]
        else:
            if max_retrieve > 0:
                model_ids = np.random.choice(model_ids, size=min(max_retrieve, len(model_ids)), replace=False)
                selected_metadata = self.__model_db.get_metadata_for_ids(model_ids)
            else:
                model_ids = [random.choice(model_ids)]
                selected_metadata = self.__model_db.get_metadata_for_ids(model_ids)[0]

        return selected_metadata
