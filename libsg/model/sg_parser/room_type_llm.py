import json
import os

from omegaconf import DictConfig
from openai import OpenAI

from libsg.scene_types import SceneSpec, SceneType
from .base import BaseSceneParser

MODEL = "gpt-4o-mini"
ROOM_TYPE_PROMPT = """Identify the most likely type of room that the scene is describing in the following prompt:
{}

The room type should be one of 'bedroom', 'diningroom', or 'livingroom', returned in JSON format according to the following syntax:
{{
  "room_type": type of room
}}
"""


class RoomTypeLLMParser(BaseSceneParser):
    """
    Parse an unstructured scene description into a structured scene specification using an LLM-based method.

    Use of this module requires an OPENAI_API_KEY environment variable.
    """

    def __init__(self, cfg: DictConfig, **kwargs):
        self.prompt = ROOM_TYPE_PROMPT
        self.model = kwargs.get("sceneInference.parserLLM", "gpt-4o")

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
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant helping a user figure out the scene type from a text description of a scene.",
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
            room_type=output_json["room_type"],
            arch_spec=scene_spec.arch_spec,
        )
