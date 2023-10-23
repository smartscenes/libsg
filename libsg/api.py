from libsg.scene_types import (SceneState, SceneContext, ObjectInstance,
                               ObjectSpec, PlacementSpec, MoveSpec, RemoveSpec,
                               SceneSpec, SceneModifySpec)


"""
API for manipulating single object.
Can extend to handle object groups (ObjectSpec -> ObjectGroupSpec)
"""

# @app.route('/object/add')
def object_add(scene_state: SceneState,
               object_spec: ObjectSpec,
               placement_spec: PlacementSpec) -> SceneState:
    #TODO add ObjectInstance matching object_spec to scene, return scene
    return SceneState()

# @app.route('/object/remove')
def object_remove(scene_state: SceneState,
                  object_spec: ObjectSpec,
                  remove_spec: RemoveSpec) -> SceneState:
    #TODO match object_spec to object in scene, remove and return scene
    return SceneState()

# @app.route('/object/replace')
def object_replace(scene_state: SceneState,
                   object_spec: ObjectSpec,
                   new_object_spec: ObjectSpec) -> SceneState:
    # remove object matching object_spec, replace with new_object_spec
    scene_no_obj = object_remove(scene_state, object_spec)
    return object_add(scene_no_obj, new_object_spec)

# @app.route('/object/move')
def object_move(scene_state: SceneState,
                object_spec: ObjectSpec,
                move_spec: MoveSpec) -> SceneState:
    return ObjectInstance()

# @app.route('/object/retrieve')
def object_retrieve(object_spec: ObjectSpec) -> SceneState:
    return ObjectInstance()

# @app.route('/object/suggest')
def object_suggest(object_spec: ObjectSpec,
                   context: SceneContext) -> SceneState:
    return ObjectInstance()


"""
Operators at the architecture level.
"""

# @app.route('/arch/generate')
def arch_generate():
    pass

# @app.route('/arch/modify')
def arch_modify():
    pass

# @app.route('/arch/retrieve')
def arch_retrieve():
    pass


"""
Operators at the scene level.
"""

# @app.route('/scene/generate')
def scene_generate(scene_spec: SceneSpec) -> SceneState:
    #TODO generate SceneState corresponding to scene_spec
    pass

# @app.route('/scene/modify')
def scene_modify(scene_state: SceneState, modify_spec: SceneModifySpec) -> SceneState:
    pass

# @app.route('/scene/retrieve')
def scene_retrieve(scene_spec: SceneSpec) -> SceneState:
    #TODO retrieve SceneState corresponding to scene_spec
    return SceneState()


"""
Free form text-based API.
A type field indicates input specification type and model/strategy to follow
Common arguments: spec_type (text/language code, template), strategy (how to generate/modify/retrieve),
"""

# @app.route('/generate')
def generate():
    pass

# @app.route('/modify')
def modify():
# /scene/modify: initial_scene_state, modification: text
    pass

# /scene/retrieve
# @app.route('/retrieve')
def retrieve():
    pass

"""
Examples of JSON payloads to above endpoints

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
    "relation": "on|next",  # TODO: expand/refine relation classes
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