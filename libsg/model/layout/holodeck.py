import multiprocessing as mp
import os
import pprint
from omegaconf import DictConfig
from typing import Any

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from libsg.scene_types import JSONDict, SceneLayoutSpec
from libsg.model.holodeck.constants import OBJATHOR_ASSETS_DIR
from libsg.model.holodeck.generation.ceiling_objects import CeilingObjectGenerator
from libsg.model.holodeck.generation.floor_objects import FloorObjectGenerator
from libsg.model.holodeck.generation.objaverse_retriever import ObjathorRetriever
from libsg.model.holodeck.generation.object_selector import ObjectSelector
from libsg.model.holodeck.generation.small_objects import SmallObjectGenerator, ControllerProcess
from libsg.model.holodeck.generation.wall_objects import WallObjectGenerator
from libsg.model.instructscene.clip_encoders import CLIPTextEncoderWrapper
from .base import BaseLayout


class Holodeck(BaseLayout):
    SHIFT_BY_SCENE_CENTROID = False  # Holodeck generates layout based on provided arch coordinates, so no shift needed

    def __init__(
        self,
        llm_model_name: str,
        clip_params: DictConfig,
        retrieval_threshold: int,
        random_selection: bool = False,
        use_constraint: bool = True,
        use_milp: bool = False,
        add_ceiling: bool = False,
        objaverse_asset_dir: str = OBJATHOR_ASSETS_DIR,
        **kwargs,
    ):
        # initialize llm
        try:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise EnvironmentError(
                "Expected an OpenAI API key in order to use the LLMSceneParser. Please set OPENAI_API_KEY and "
                "try again."
            )
        self.llm = ChatOpenAI(
            model_name=llm_model_name,
            max_tokens=2048,
            openai_api_key=openai_api_key,
        )

        # initialize CLIP
        (
            self.clip_model,
            _,
            self.clip_preprocess,
        ) = open_clip.create_model_and_transforms(clip_params.model_name, pretrained=clip_params.pretrained)
        self.clip_tokenizer = open_clip.get_tokenizer(clip_params.model_name)

        # initialize sentence transformer
        self.sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

        self.retrieval_threshold = retrieval_threshold
        self.object_retriever = ObjathorRetriever(
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,
            clip_tokenizer=self.clip_tokenizer,
            sbert_model=self.sbert_model,
            retrieval_threshold=self.retrieval_threshold,
        )
        self.object_selector = ObjectSelector(object_retriever=self.object_retriever, llm=self.llm)
        self.floor_object_generator = FloorObjectGenerator(object_retriever=self.object_retriever, llm=self.llm)
        self.wall_object_generator = WallObjectGenerator(object_retriever=self.object_retriever, llm=self.llm)
        self.ceiling_generator = CeilingObjectGenerator(object_retriever=self.object_retriever, llm=self.llm)
        self.small_object_generator = SmallObjectGenerator(object_retriever=self.object_retriever, llm=self.llm)

        self.used_assets = []  # currently not used, but specifies assets to exclude

        self.random_selection = random_selection
        self.use_constraint = use_constraint
        self.use_milp = use_milp
        self.add_ceiling = add_ceiling
        self.additional_requirements_object = "N/A"
        self.additional_requirements_ceiling = "N/A"

        self.objaverse_asset_dir = objaverse_asset_dir

        self.up_axis = "y"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_encoder = CLIPTextEncoderWrapper.get_model(name="ViT-bigG-14", device=device)

        # self.materials_db = MaterialsDB(materials_dir)

    def select_objects(self, scene, additional_requirements_object, used_assets=[]):
        self.object_selector.used_assets = used_assets
        object_selection_plan, selected_objects = self.object_selector.select_objects(
            scene, additional_requirements_object
        )
        scene["object_selection_plan"] = object_selection_plan
        scene["selected_objects"] = selected_objects
        return scene

    def generate_ceiling_objects(self, scene, additional_requirements_ceiling="N/A"):
        (
            raw_ceiling_plan,
            ceiling_objects,
        ) = self.ceiling_generator.generate_ceiling_objects(scene, additional_requirements_ceiling)
        scene["ceiling_objects"] = ceiling_objects
        scene["raw_ceiling_plan"] = raw_ceiling_plan
        return scene

    def generate_small_objects(self, scene, used_assets=[]):
        self.small_object_generator.used_assets = used_assets
        submit_queue = mp.Queue()
        return_queue = mp.Queue()
        controller_process = ControllerProcess(submit_queue, return_queue, self.objaverse_asset_dir, scene)
        controller_process.start()
        event = return_queue.get(timeout=30)  # wait for controller to start
        receptacle_ids = [
            obj["objectId"] for obj in event.metadata["objects"] if obj["receptacle"] and "___" not in obj["objectId"]
        ]
        if "Floor" in receptacle_ids:
            receptacle_ids.remove("Floor")

        try:
            (
                small_objects,
                receptacle2small_objects,
            ) = self.small_object_generator.generate_small_objects(scene, controller_process, receptacle_ids)
            scene["small_objects"] = small_objects
            scene["receptacle2small_objects"] = receptacle2small_objects
        except:
            import traceback

            traceback.print_exc()
            scene["small_objects"] = []
            print("Failed to generate small objects")

        submit_queue.put(None)  # terminate
        controller_process.join()

        return scene

    def generate_clip_embeddings(self, object_names: list[str]):
        with torch.no_grad():
            query_feature_clip = self.text_encoder(object_names)
            return F.normalize(query_feature_clip, p=2, dim=-1)

    def generate(self, layout_spec: SceneLayoutSpec, **kwargs) -> list[dict[str, Any]]:
        """Generate an architecture based on the given specification.

        :param scene_spec: unstructured scene specification
        :raises ValueError: scene spec type not supported for generating architecture
        :return:
        """
        # scene = layout_spec.arch.to_holodeck()
        scene = layout_spec.arch.raw
        if scene is None:
            raise ValueError(
                "The Holodeck layout module was not able to extract a raw ai2thor-format scene architecture. Please confirm that you used Holodeck as the arch method "
            )

        # select objects
        self.object_selector.random_selection = self.random_selection
        scene = self.select_objects(
            scene,
            additional_requirements_object=self.additional_requirements_object,
            used_assets=self.used_assets,
        )

        # generate floor objects
        self.floor_object_generator.use_milp = self.use_milp
        scene["floor_objects"] = self.floor_object_generator.generate_objects(scene, use_constraint=self.use_constraint)

        # generate wall objects
        scene["wall_objects"] = self.wall_object_generator.generate_wall_objects(
            scene, use_constraint=self.use_constraint
        )

        # combine floor and wall objects
        scene["objects"] = scene["floor_objects"] + scene["wall_objects"]

        # generate small objects
        scene = self.generate_small_objects(scene, used_assets=self.used_assets)
        scene["objects"] += scene["small_objects"]

        # generate ceiling objects
        if self.add_ceiling:
            scene = self.generate_ceiling_objects(
                scene,
                additional_requirements_ceiling=self.additional_requirements_ceiling,
            )
            scene["objects"] += scene["ceiling_objects"]

        # TODO: change format
        # TODO: fix axis alignment

        # extract boxes
        objects = self.prepare_outputs(scene)
        print("OBJECTS:")
        pprint.pprint(objects)

        return objects

    def prepare_outputs(self, scene: JSONDict) -> list[dict[str, Any]]:
        CM_TO_M = 100   # convert cm to m for mesh scale
        
        def transform_position(position, axis_order: list[int]):
            reordered = np.array(position)[axis_order]
            # reordered[0] = -reordered[0] # Different from default (ATISS/DiffuScene)
            return reordered.tolist()

        def transform_angle(angle):
            # Mirror the angle by negating it
            angle = -angle # Different from default (ATISS/DiffuScene)
            angle -= np.pi
            if angle <= -np.pi:
                angle += 2 * np.pi
            return angle

        # extract boxes
        objects = []
        for obj in scene["objects"]:
            object_id = f"objaverse.{obj['assetId']}"
            object_name = obj["object_name"].rpartition("-")[0]  # remove last index on object name
            object_name = object_name.replace("_", " ")

            dim = obj["dimension"]
            dim = [dim["x"]/CM_TO_M, dim["y"]/CM_TO_M, dim["z"]/CM_TO_M]
            pos = obj["position"]
            # pos = [pos["x"], pos["y"] - dim[1] / 2, pos["z"]] #for HSSD asset
            pos = [pos["x"], pos["y"], pos["z"]]
            orientation = obj["rotation"]["y"] * np.pi / 180

            # flip axes if needed
            if self.up_axis == "y":  # assumes LHR, y-axis is vertical, and axes are rotated 180 deg
                axis_order = [0, 2, 1]
                dim = [dim[ax] for ax in axis_order]
                pos = transform_position(pos, axis_order)
                orientation = transform_angle(orientation)

            objects.append(
                {
                    "id": object_id,
                    "wnsynsetkey": None,
                    "class": object_name,
                    "dimensions": dim,
                    "position": pos,
                    "orientation": orientation,
                }
            )

        object_names = [obj["class"] for obj in objects]
        embeddings = self.generate_clip_embeddings(object_names)
        for idx, obj in enumerate(objects):
            obj["embedding"] = embeddings[idx].cpu().numpy()

        return objects
