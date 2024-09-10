from collections import defaultdict

import numpy as np

from .base import EvaluationBase
from libsg.scene import Scene
from libsg.scene_types import SceneGraph


class ObjectCategoryDistribution(EvaluationBase):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.reset()

    def reset(self):
        self.frequencies = defaultdict(int)
        self.num_scenes = 0

    def __call__(self, inp, scene_graph: SceneGraph, scene: Scene):
        if scene_graph is None:
            return

        for obj_class in set(obj.name for obj in scene_graph.objects):
            self.frequencies[obj_class] += 1

        self.num_scenes += 1

    def log(self):
        print("Object Category Distribution:")
        aggregate = {}
        if self.frequencies:
            for obj_class, freq in self.frequencies.items():
                print(f"  {obj_class}: {freq / self.num_scenes:.2f}")
                aggregate[obj_class] = freq / self.num_scenes
        else:
            print("  <No objects found>")
        return aggregate


class ObjectCountDistribution(EvaluationBase):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.reset()

    def reset(self):
        self.frequencies = defaultdict(list)
        self.num_scenes = 0

    def __call__(self, inp, scene_graph: SceneGraph, scene: Scene):
        if scene_graph is None:
            return

        frequencies = defaultdict(int)
        for obj in scene_graph.objects:
            frequencies[obj.name] += 1

        # update counts
        for obj_class, freq in frequencies.items():
            self.frequencies[obj_class].append(freq)

        self.num_scenes += 1

    def log(self):
        print("Object Count Distribution:")
        aggregate = {}
        if self.frequencies:
            for obj_class, freqs in self.frequencies.items():
                freq_arr = np.zeros((self.num_scenes,))
                freq_arr[: len(freqs)] = freqs
                mean = np.mean(freq_arr)
                std = np.std(freq_arr)
                print(f"  {obj_class}: {mean:.2f} ({std:.2f})")
                aggregate[obj_class] = {"mean": mean, "std": std}
        else:
            print("  <No objects found>")
        return aggregate


class SceneGraphDistribution(EvaluationBase):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.reset()

    def reset(self):
        self.frequencies = defaultdict(int)
        self.num_scenes = 0

    def __call__(self, inp, scene_graph: SceneGraph, scene: Scene):
        if scene_graph is None:
            return

        for relationship in set((rel.subject.name, rel.type, rel.target.name) for rel in scene_graph.relationships):
            self.frequencies[relationship] += 1

        self.num_scenes += 1

    def log(self):
        print("Scene Graph Relationship Distribution:")
        aggregate = {}
        if self.frequencies:
            for rel, freq in self.frequencies.items():
                print(f"  {rel}: {freq / self.num_scenes:.2f}")
                aggregate[str(rel)] = freq / self.num_scenes
        else:
            print("  <No relationships found>")
        return aggregate
