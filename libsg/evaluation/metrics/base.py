from libsg.scene import Scene


class EvaluationBase:
    def __call__(self, inp, scene_graph, scene: Scene, **kwargs):
        raise NotImplementedError
