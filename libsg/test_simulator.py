#!/usr/bin/env python
import argparse
import json

from libsg.simulator import Simulator
from libsg.scene import Scene

from libsg.simscene import SimScene


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test simulator')
    parser.add_argument('--input', '-i', required=True, help='Input scenestate JSON')
    parser.add_argument('--object_dir', '-d', required=True, help='Directory with object collision meshes')
    parser.add_argument('--gui', action='store_true', help='Whether to enable GUI')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')
    parser.add_argument('--use_y_up', '-y', action='store_true', help='Whether to use y-up (default is z-up)')
    args = parser.parse_args()

    sim_mode = 'gui' if args.gui else 'direct'
    sim = Simulator(mode=sim_mode, verbose=args.verbose, use_y_up=args.use_y_up)
    scene_state = json.load(open(args.input))['scene']
    scene = Scene.from_json(scene_state)
    sim_scene = SimScene(sim, scene, { 'collision_mesh_dir': args.object_dir }, include_ground=True)

    sim.run()
