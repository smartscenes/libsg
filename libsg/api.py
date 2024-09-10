"""
api.py
------
The internal facing API for the text-to-scene application. This API will assume calls are being made internally using
the class specifications defined for the backend rather than for HTTP requests. The internal API may also expose
some additional parameters which are not made available to external users for the sake of simplifying the user-facing
interface.

Examples of JSON payloads to below endpoints

# SceneSpec
{  
   "type": "category"           # room type
   "input": "bedroom"           # actual input
   "format": "STK"              # STK json scene-state (other formats such as HAB is also supported)
}

{  
   "type": "text"                                                # free-form text
   "input": "a bedroom with a bed and two nightstands"           # actual input
   "format": "STK"              # STK json scene-state (other formats such as HAB is also supported)
}

# ObjectSpec
{
    "type": "model_id",  # a model asset id
    "object": "fpModel.afc88dcb16a8a7e93e905bde56a83b783876fbc4"
}
{
    "type": "object_id",  # a specific object instance id
    "object": "0_0_10"
}
{
    "type": "category",              # a category search (will return instance from category)
    "object": "chair",               # raw text query (matches against any field, default is wildcard)
    "wnsynsetkey": "armchair.n.01",  # WordNet synset search (optional)
    "dimensions": [0.5, 1.0, 0.5]    # desired physical dimensions (optional)
}

# PlacementSpec
{
    "type": "placement_point",  # place at specific point and orientation in scene coordinates
    "position": [x, y, z],      # point at which to place
    "orientation": theta,       # angle in radians for CCW rotation around up axis (optional)
    "allow_collisions": false   # whether to allow collisions between objects and object-scene (optional)
}
{
    "type": "placement_relation",  # a placement defined by relation to reference object
    "relation": "on|next",         # TODO: expand/refine relation classes
    "reference": ObjectSpec
}

# AddObjectSpec
{
    "type": "add_object"
    "scene_state": SceneState,
    "object_spec": ObjectSpec,
    "placement_spec": PlacementSpec
}

# RemoveObjectSpec
{
    "type": "remove_object"
    "scene_state": SceneState,
    "object_spec": ObjectSpec,
    "remove_children": True  # whether supported objects (e.g., dishes on table) should be removed
    # TODO: expand into more refined strategies for handling adjustments to scene post-removal
}
"""

from typing import Any, Optional

from libsg.assets import AssetDb
from libsg.object_builder import ObjectBuilder
from libsg.scene import ModelInstance
from libsg.scene_types import (
    JSONDict,
    SceneState,
    SceneContext,
    ObjectSpec,
    PlacementSpec,
    MoveSpec,
    SceneSpec,
    SceneLayoutSpec,
    SceneLayout,
    SceneModifySpec,
)
from libsg.scene_builder import SceneBuilder
from libsg.scene_parser import SceneParser
from libsg.config import config as cfg


"""
API for manipulating single object.
Can extend to handle object groups (ObjectSpec -> ObjectGroupSpec)
"""


# @app.route('/object/add')
def object_add(scene_state: SceneState, object_spec: ObjectSpec, placement_spec: PlacementSpec) -> SceneState:
    raise NotImplementedError("Not tested")

    scene_builder = SceneBuilder(cfg.scene_builder, cfg.arch, cfg.layout)
    return scene_builder.object_add(scene_state, object_spec, placement_spec)


# @app.route('/object/remove')
def object_remove(scene_state: SceneState, object_spec: ObjectSpec) -> SceneState:
    # TODO: may want to add back RemoveSpec argument
    # TODO match object_spec to object in scene, remove and return scene
    raise NotImplementedError("Not tested")

    scene_builder = SceneBuilder(cfg.scene_builder, cfg.arch, cfg.layout)
    new_scene_state = scene_builder.object_remove(scene_state, object_spec)
    return new_scene_state


# @app.route('/object/replace')
def object_replace(scene_state: SceneState, object_spec: ObjectSpec, new_object_spec: ObjectSpec) -> SceneState:
    # remove object matching object_spec, replace with new_object_spec
    raise NotImplementedError("Not tested")

    scene_no_obj = object_remove(scene_state, object_spec)
    return object_add(scene_no_obj, new_object_spec)


# @app.route('/object/move')
def object_move(scene_state: SceneState, object_spec: ObjectSpec, move_spec: MoveSpec) -> SceneState:
    raise NotImplementedError


# @app.route('/object/retrieve')
def object_retrieve(object_spec: ObjectSpec, max_retrieve: int = 1, constraints: str = "") -> list[ModelInstance]:
    base_solr_url = cfg.scene_builder.solr_url
    model_db = AssetDb("model", cfg.scene_builder.get("model_db"), solr_url=f"{base_solr_url}/models3d")
    object_builder = ObjectBuilder(model_db, cfg.scene_builder.model_db)
    return object_builder.retrieve(object_spec, max_retrieve=max_retrieve, constraints=constraints)


# @app.route('/object/retrieve')
def embedding_retrieve(object_spec: ObjectSpec) -> list[float]:
    object_builder = ObjectBuilder(None, cfg.scene_builder.model_db)
    embedding_tensor = object_builder.get_text_embedding(object_spec.description)
    return embedding_tensor.cpu().tolist()


# @app.route('/object/generate')
def object_generate(object_spec: ObjectSpec) -> SceneState:
    raise NotImplementedError


# @app.route('/object/suggest')
def object_suggest(object_spec: ObjectSpec, context: SceneContext) -> SceneState:
    raise NotImplementedError


"""
Operators at the architecture level.
"""


# @app.route('/arch/generate')
def arch_generate():
    raise NotImplementedError


# @app.route('/arch/modify')
def arch_modify():
    raise NotImplementedError


# @app.route('/arch/retrieve')
def arch_retrieve():
    raise NotImplementedError


"""
Operators at the scene level.
"""


def scene_generate_layout(layout_spec: SceneLayoutSpec, model_name: Optional[str] = None, **kwargs) -> SceneLayout:
    """Generate course scene layout given architecture and layout parameters.

    :param layout_spec: specification for layout architecture
    :param model_name: name of model to use. If not specified, model specified in config will be used.
    :return: list of objects of the form
            {
                'wnsynsetkey': name of object class,
                'dimensions': dimensions of object as (x, y, z),
                'position': position of object in scene as (x, y, z),
                'orientation': rotation angle to apply to each object in radians,
            }
    """
    scene_builder = SceneBuilder(cfg.scene_builder, cfg.arch, cfg.layout, **kwargs)
    layout = scene_builder.generate_layout(layout_spec, model_name)
    return layout


# @app.route('/scene/generate')
def scene_generate(scene_spec: SceneSpec, **kwargs) -> JSONDict:
    """Generate a scene with architecture and objects given prompt specification.

    :param scene_spec: specification of scene to generate
    :return: scene state object specifying architecture and objects to use in scene, and their locations
    """
    parsed_scene_spec = SceneParser(cfg.parser, **kwargs).parse(scene_spec)
    scene_builder = SceneBuilder(cfg.scene_builder, cfg.arch, cfg.layout, **kwargs)
    scene_state = scene_builder.generate(parsed_scene_spec)
    return scene_state


# @app.route('/scene/modify')
def scene_modify(scene_state: SceneState, modify_spec: SceneModifySpec) -> SceneState:
    raise NotImplementedError


# @app.route('/scene/retrieve')
def scene_retrieve(scene_spec: SceneSpec, **kwargs) -> SceneState:
    """Retrieve scene by ID.

    TODO: API not tested yet

    :param scene_spec: specification of scene to retrieve
    :return: scene state object specifying architecture and objects to use in scene, and their locations
    """
    scene_builder = SceneBuilder(cfg.scene_builder, cfg.arch, cfg.layout, **kwargs)
    scene_state = scene_builder.retrieve(scene_spec)
    return scene_state


"""
Free form text-based API.
A type field indicates input specification type and model/strategy to follow
Common arguments: spec_type (text/language code, template), strategy (how to generate/modify/retrieve),
"""


# @app.route('/generate')
def generate():
    raise NotImplementedError


# @app.route('/modify')
def modify():
    # /scene/modify: initial_scene_state, modification: text
    raise NotImplementedError


# /scene/retrieve
# @app.route('/retrieve')
def retrieve():
    raise NotImplementedError
