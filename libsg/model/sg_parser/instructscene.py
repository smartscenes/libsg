from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from diffusers.training_utils import EMAModel

from libsg.model.instructscene import model_from_config, load_checkpoints, SgObjfeatVQDiffusion
from libsg.model.instructscene.sg_diffusion_vq_objfeat import scatter_trilist_to_matrix
from libsg.model.instructscene.clip_encoders import CLIPTextEncoder
from libsg.scene_types import SceneSpec, SceneType
from .base import BaseSceneParser
from .room_type import RoomTypeParser


class InstructSceneParser(BaseSceneParser):
    """
    Parse an unstructured scene description into a structured scene specification which can be used by downstream
    modules using the InstructScene method.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.raw_classes = list(config.data.raw_classes)
        self.device = config.get("device", "cpu")
        self.cfg_scale = config.network.cfg_scale
        self.max_length = config.data.max_length
        self.predicate_types = config.data.predicate_types
        self.condition_type = config.get("condition_type", "text")

        # Initialize the model
        self.text_encoder = CLIPTextEncoder(config.network.text_encoder, device=self.device)
        self.text_to_sg_model = self._load_text_to_sg_model(config, text_emb_dim=self.text_encoder.text_emb_dim)

        self.room_type_parser = RoomTypeParser(OmegaConf.create({}))

    def _load_text_to_sg_model(self, config: DictConfig, text_emb_dim: int) -> SgObjfeatVQDiffusion:
        """Initialize model for parsing scene graph from text"""
        # Initialize the model
        text_to_sg_model = model_from_config(
            config.network, len(self.raw_classes), len(self.predicate_types), text_emb_dim=text_emb_dim
        ).to(self.device)

        # Create EMA for the model
        ema_config = config["training"]["ema"]
        if ema_config["use_ema"]:
            ema_states = EMAModel(text_to_sg_model.parameters())
            ema_states.to(self.device)
        else:
            ema_states: EMAModel = None

        # Load the weights from a checkpoint
        load_checkpoints(
            text_to_sg_model, config.network.ckpt_dir, ema_states, epoch=config.network.ckpt_epoch, device=self.device
        )

        # Evaluate with the EMA parameters if specified
        if ema_states is not None:
            print(f"Copy EMA parameters to the model\n")
            ema_states.copy_to(text_to_sg_model.parameters())
        text_to_sg_model.eval()
        return text_to_sg_model

    def parse(self, scene_spec: SceneSpec) -> SceneSpec:
        # parse room type
        room_type = self.room_type_parser.parse(scene_spec).input

        if self.condition_type == "none":
            text_last_hidden_state = torch.zeros_like(text_last_hidden_state)
            text_embeds = torch.zeros_like(text_embeds)
        text_last_hidden_state, text_embeds = self.text_encoder(scene_spec.input)

        objs, edges, objfeat_vq_indices = self.text_to_sg_model.generate_samples(
            1, self.max_length, text_last_hidden_state, text_embeds, cfg_scale=self.cfg_scale
        )  # batch_size = 1
        object_names, edges, objfeat_vq_indices = self._process_outputs(objs, edges, objfeat_vq_indices)

        # build scene graph
        scene_graph = self._create_scene_graph(room_type, object_names, edges, objfeat_vq_indices)

        return SceneSpec(
            type=SceneType.category,
            input=room_type,
            format=scene_spec.format,
            raw=scene_spec.input,
            scene_graph=scene_graph,
            room_type=room_type,
        )

    def _process_outputs(
        self, objs: torch.Tensor, edges: torch.Tensor, objfeat_vq_indices: Optional[torch.Tensor]
    ) -> tuple[list[str], torch.Tensor, Optional[torch.Tensor]]:
        if objfeat_vq_indices is not None:
            # Replace the empty token with a random token within the vocabulary
            objfeat_vq_indices_rand = torch.randint_like(objfeat_vq_indices, 0, 64)
            objfeat_vq_indices[objfeat_vq_indices == 64] = objfeat_vq_indices_rand[objfeat_vq_indices == 64]

        # convert to scene graph format
        # parse objects
        objs = objs.argmax(dim=-1)  # (bs, n)
        obj_masks = (objs != len(self.raw_classes)).long()  # (bs, n)
        object_names = [self.raw_classes[idx] for idx in objs[0] if idx < len(self.raw_classes)]

        # Mask and symmetrize edges
        edges = edges.argmax(dim=-1)  # (bs, n*(n-1)/2)
        edges = F.one_hot(edges, num_classes=len(self.predicate_types) + 1).float()
        edges = scatter_trilist_to_matrix(edges, objs.shape[-1])  # (bs, n, n, n_pred_types+1)
        e_mask1 = obj_masks.unsqueeze(1).unsqueeze(-1)  # (bs, 1, n, 1)
        e_mask2 = obj_masks.unsqueeze(2).unsqueeze(-1)  # (bs, n, 1, 1)
        edges = edges * e_mask1 * e_mask2  # mask out edges to non-existent objects
        edges_negative = edges[
            ...,
            [*range(len(self.predicate_types) // 2, len(self.predicate_types))]
            + [*range(0, len(self.predicate_types) // 2)]
            + [*range(len(self.predicate_types), edges.shape[-1])],
        ]  # (bs, n, n, n_pred_types+1)
        edges = edges + edges_negative.permute(0, 2, 1, 3)
        edge_mask = torch.eye(objs.shape[-1], device=self.device).bool().unsqueeze(0).unsqueeze(-1)  # (1, n, n, 1)
        edge_mask = ((~edge_mask).float() * e_mask1 * e_mask2).squeeze(-1)  # (bs, n, n)
        assert torch.all(
            edges.sum(dim=-1) == edge_mask
        )  # every edge is one-hot encoded, except for the diagonal and empty nodes
        edges_empty = edges[edges.sum(dim=-1) == 0]
        edges_empty[..., -1] = 1.0
        edges[edges.sum(dim=-1) == 0] = edges_empty  # set the empty edges to the last class
        edges = torch.argmax(edges, dim=-1)  # (bs, n, n)

        return object_names, edges, objfeat_vq_indices

    def _create_scene_graph(
        self, room_type: str, object_names: list[str], edges: torch.Tensor, objfeat_vq_indices: Optional[torch.Tensor]
    ) -> dict[str, Any]:
        scene_graph = {
            "room_type": room_type,
            "objects": [],
            "relationships": [],
        }

        # iterate trhough objects
        for i, obj_name in enumerate(object_names):
            scene_graph["objects"].append(
                {
                    "id": i + 1,  # 1 indexed
                    "name": obj_name,
                    "attributes": [],
                    "feature": objfeat_vq_indices[0, i] if objfeat_vq_indices is not None else None,
                }
            )

        # iterate through edges
        for s in range(edges.shape[1]):
            for t in range(edges.shape[2]):
                if edges[0, s, t] < len(self.predicate_types):
                    scene_graph["relationships"].append(
                        {
                            "type": self.predicate_types[edges[0, s, t].item()],
                            "subject_id": s,
                            "target_id": t,
                        }
                    )

        return scene_graph
