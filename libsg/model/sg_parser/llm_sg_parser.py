import json
import os

from omegaconf import DictConfig
from openai import OpenAI

from libsg.scene_types import SceneSpec, SceneType
from .base import BaseSceneParser


OV_SCENE_GRAPH_PROMPT = """Please follow the examples in the Visual Genome dataset and generate a scene graph that best describes the following text:
"{}"
Return the output in a JSON format according to the following format:
{{
  "room_type": type of room, such as bedroom,
  "objects": [
    {{
      "id": id of object,
      "name": name of object as string,
      "attributes": array of string,
    }}
  ],
  "relationships": [
    {{
        "type": type of relationship as string,
        "subject_id": id of object which is the subject of the relationship,
        "target_id": id of object which is the target of the relationship
    }}
  ]
}}

The object and relationship IDs should start with 0 and increment. Every subject_id and target_id in relationships should correspond to an existing object ID.

If a number of objects are specified, please include each object in the count as a separate node. For example, if the text specifies "two chairs", include two separate nodes for the chairs.
"""

THREEDFRONT_SCENE_GRAPH_PROMPT = """Please follow the examples in the Visual Genome dataset and generate a scene graph that best describes the following text:
"{}"
Return the output in a JSON format according to the following format:
{{
  "room_type": bedroom, diningroom, or livingroom,
  "objects": [
    {{
      "id": id of object,
      "name": name of object as string,
      "attributes": array of string,
    }}
  ],
  "relationships": [
    {{
        "id": id of relationship,
        "type": type of relationship as string. It must be one of "above", "left of", "in front of", "closely left of", "closely in front of", "below", "right of", "behind", "closely right of", or "closely behind"
        "subject_id": id of object which is the subject of the relationship,
        "target_id": id of object which is the target of the relationship
    }}
  ]
}}

The object and relationship IDs should start with 0 and increment. Every subject_id and target_id in relationships should correspond to an existing object ID.

If the room type is bedroom, the object name must be one of "armchair", "bookshelf", "cabinet", "ceiling lamp", "chair", "children cabinet", "coffee table", "desk", "double bed", "dressing chair", "dressing table", "kids bed", "nightstand", "pendant lamp", "shelf", "single bed", "sofa", "stool", "table", "tv stand", or "wardrobe".

If the room type is diningroom or livingroom, the object name must be one of "armchair", "bookcase", "cabinet", "ceiling lamp", "chaise longue sofa", "chinese chair", "coffee table", "console table", "corner/side table", "desk", "dining chair", "dining table", "l-shaped sofa", "lazy sofa", "lounge chair", "loveseat sofa", "multi-seat sofa", "pendant lamp", "round end table", "shelf", "stool", "tv stand", "wardrobe", or "wine cabinet".

If a number of objects are specified, please include each object in the count as a separate node. For example, if the text specifies "two chairs", include two separate nodes for the chairs.
"""


class LLMSceneParser(BaseSceneParser):
    """
    Parse an unstructured scene description into a structured scene specification using an LLM-based method.

    Use of this module requires an OPENAI_API_KEY environment variable.
    """

    prompts = {
        "open_vocabulary": OV_SCENE_GRAPH_PROMPT,
        "3dfront": THREEDFRONT_SCENE_GRAPH_PROMPT,
    }

    def __init__(self, cfg: DictConfig):
        self.prompt = LLMSceneParser.prompts[cfg.prompt_type]

    def parse(self, scene_spec: SceneSpec) -> SceneSpec:
        print(f"Parsing scene graph from text input: {scene_spec.input}")
        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise EnvironmentError(
                "Expected an OpenAI API key in order to use the LLMSceneParser. Please set OPENAI_API_KEY and "
                "try again."
            )

        client = OpenAI(api_key=api_key)
        inp = self.prompt.format(scene_spec.input)

        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant helping a user generate a semantic scene graph from a text description of a scene.",
                },
                {"role": "user", "content": inp},
            ],
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        try:
            output_json = json.loads(raw_output)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse scene graph response from OpenAI as JSON:\n{raw_output}")
        return SceneSpec(
            type=SceneType.category,
            input=output_json["room_type"],
            format=scene_spec.format,
            raw=scene_spec.input,
            scene_graph=output_json,
            room_type=output_json["room_type"],
        )
