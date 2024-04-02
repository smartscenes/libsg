"""
app.py
------
The external facing API for the text-to-scene application. This API will assume calls are being made from an
external party, e.g. the frontend application, and thus handle any pre- or post-processing to convert the 
inputs/outputs into a form that the internal API can handle.
"""

import os
import traceback

import flask
from flask import jsonify, request
from flask_restx import Api, Resource, fields
from werkzeug.exceptions import HTTPException

from libsg import __version__
from libsg.api import (
    object_add as object_add_api,
    object_remove as object_remove_api,
    scene_generate as scene_generate_api,
    # scene_retrieve as scene_retrieve_api,
)
from libsg.scene_types import (
    JSONDict,
    SceneState,
    SceneSpec,
    SceneType,
)


app = flask.Flask(__name__)
api = Api(
    app,
    version=__version__,
    title="Text-to-scene API",
    description="API for Text-to-scene generation application.",
)

# define API specs
# define main model specs
ObjectRetrieveSpecModel = api.model(
    "ObjectRetrieveSpec",
    {
        "type": fields.String(
            required=True,
            description="type of input specification",
            enum=["model_id", "object_id", "category"],
            example="category",
        ),
        "object": fields.String(required=True, description="input specification for object", example="chair"),
        "wnsynsetkey": fields.String(
            required=False, description="[type=category] WordNet synset search", example="armchair.n.01"
        ),
        "dimensions": fields.List(
            fields.Float,
            required=False,
            min_items=3,
            max_items=3,
            description="desired physical dimensions as a list of the form [x, y, z]",
            example=[0.5, 1.0, 0.5],
        ),
    },
)

ObjectGenerateSpecModel = api.model(
    "ObjectGenerateSpec",
    {
        "type": fields.String(
            required=True,
            description="type of input specification",
            enum=["text"],
            example="text",
        ),
        "object": fields.String(required=True, description="input specification for object", example="wooden chair"),
        "dimensions": fields.List(
            fields.Float,
            required=False,
            min_items=3,
            max_items=3,
            description="desired physical dimensions as a list of the form [x, y, z]",
            example=[0.5, 1.0, 0.5],
        ),
    },
)

PlacementSpecModel = api.model(
    "PlacementSpec",
    {
        "type": fields.String(
            required=True,
            description="specify type of placement for object",
            enum=["placement_point", "placement_relation"],
        ),
        "position": fields.List(
            fields.Float,
            required=False,
            description="[type=placement_point] specify (x, y, z) coordinates at which to place object",
            min_items=3,
            max_items=3,
        ),
        "orientation": fields.Float(
            required=False, description="[type=placement_point] angle in radians for CCW rotation around up axis"
        ),
        "allow_collisions": fields.Boolean(
            required=False,
            description="[type=placement_point] if true, allow collisions between objects and object-scene",
            default=False,
        ),
        "relation": fields.String(
            required=False,
            description="[type=placement_relation] relation of object placement to reference object",
            enum=["on", "next"],
        ),
        "reference": fields.Nested(ObjectRetrieveSpecModel, required=False),
    },
)


def scenestate_json_wrapper(scenestate) -> JSONDict:
    """Generate JSON wrapper around scene state input"""
    return {"format": "sceneState", "scene": scenestate}


@app.route("/scene/retrieve/", defaults={"id": "102344115"}, methods=["GET", "POST"])
@app.route("/scene/retrieve/<id>", methods=["GET", "POST"])
def scene_retrieve(id) -> SceneState:
    """Retrieve scene by ID.

    :param id: scene ID
    :return: scene state corresponding to ID
    """
    format_ = request.json.get("format", "STK")
    config = request.json.get("config", {})
    scene_spec = SceneSpec(type="id", input=id, format=format_)
    return scene_retrieve_api(scene_spec, **config)


@app.route("/scene/generate", methods=["GET", "POST"])
@api.doc(
    description="Generates a scene based on the provided prompt or category.",
    body=api.model(
        "SceneGenerateApi",
        {
            "type": fields.String(
                required=True,
                description="type of input specification",
                enum=["category", "text"],
                example="text",
            ),
            "input": fields.String(
                required=True, description="input specification for scene", example="Generate a scene of a bedroom."
            ),
            "format": fields.String(
                required=True, description="format of output scene state", enum=["STK"], example="STK"
            ),
        },
    ),
)
def generate_scene():
    """
    Generate a scene based on a scene description prompt.

    Arguments should include the following parameters:
    {
        "type": text | category,
        "input": free-form input description (text) or scene type (category),
        "format": STK (default) | HAB,
    }

    As of now, the input primarily supports text prompts which include a reference to the name of the scene type.
    """
    type_ = request.json["type"]
    input_ = request.json.get("input", "bedroom")
    format_ = request.json.get("format", "STK")
    config = request.json.get("config", {})
    scene_spec = SceneSpec(type=SceneType(type_), input=input_, format=format_)
    return scene_generate_api(scene_spec, **config)


@app.route("/object/add", methods=["POST"])
def add_object() -> JSONDict:
    """Add an object to the existing scene.

    TODO: This API has not been fully tested yet.

    :return: New scene state with object added
    """

    add_json = request.json.get_json()
    scene_state = add_json["scene_state"]
    if "modifications" in scene_state:  # clear any previous modifications
        scene_state["modifications"] = []

    if "specs" in add_json:  # multiple additions
        new_scene_state = scene_state
        for spec in add_json["specs"]:
            new_scene_state = object_add_api(new_scene_state, spec["object_spec"], spec["placement_spec"])
    else:  # single addition
        object_spec = add_json["object_spec"]
        placement_spec = add_json["placement_spec"]
        new_scene_state = object_add_api(scene_state, object_spec, placement_spec)
    return scenestate_json_wrapper(new_scene_state)


@app.route("/object/remove", methods=["POST"])
def object_remove() -> JSONDict:
    """Remove an object from the existing scene.

    TODO: This API has not been fully tested yet.

    :return: New scene state with object removed
    """

    remove_json = request.get_json()
    scene_state = remove_json["scene_state"]
    object_spec = remove_json["object_spec"]
    new_scene_state = object_remove_api(scene_state, object_spec)
    return scenestate_json_wrapper(new_scene_state)


# TODO: Make this be more informative
@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    response = jsonify(error=str(e))
    return response, 500
