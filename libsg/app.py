import flask
import os
import traceback

from flask import jsonify, request
from libsg.config import config as cfg
from werkzeug.exceptions import HTTPException
from libsg.scene_types import (SceneState, SceneContext, ObjectInstance,
                               ObjectSpec, PlacementSpec, MoveSpec, RemoveSpec,
                               SceneSpec, SceneModifySpec)
from libsg.scene_builder import SceneBuilder


app = flask.Flask(__name__)
print(cfg)
scene_builder = SceneBuilder(cfg.scene_builder)


@app.route("/")
def hello_world():
    return "<p>Hello, SceneBuilder!</p>"


@app.route('/scene/retrieve/', defaults={'id': '102344115'})
@app.route('/scene/retrieve/<id>')
def scene_retrieve(id):
    scene_spec = SceneSpec(type='id', input=id, format="HAB")
    scene_state = scene_builder.retrieve(scene_spec)
    return scene_state


@app.route('/scene/generate')
def scene_generate():
    stage_id = request.args.get('stage_id', '102344115')
    format = request.args.get('format', "HAB")
    scene_spec = SceneSpec(type='id', input=stage_id, format=format)
    scene_state = scene_builder.generate(scene_spec)
    return scene_state


def scenestate_json_wrapper(scenestate):
    return { 'format': 'sceneState', 'scene': scenestate }


@app.route('/object/remove', methods=['POST'])
def object_remove():
    remove_json = request.get_json()
    scene_state = remove_json['scene_state']
    object_spec = remove_json['object_spec']
    new_scene_state = scene_builder.object_remove(scene_state, object_spec)
    return scenestate_json_wrapper(new_scene_state)


@app.route('/object/add', methods=['POST'])
def object_add():
    add_json = request.get_json()
    scene_state = add_json['scene_state']
    if 'modifications' in scene_state:  # clear any previous modifications
        scene_state['modifications'] = []
    if 'specs' in add_json:  # multiple additions
        new_scene_state = scene_builder.object_add_multiple(scene_state, add_json['specs'])
    else:  # single addition
        object_spec = add_json['object_spec']
        placement_spec = add_json['placement_spec']
        new_scene_state = scene_builder.object_add(scene_state, object_spec, placement_spec)
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
