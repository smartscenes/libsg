from libsg.scene import Scene


class EvaluationBase:
    def __call__(self, inp, scene: Scene):
        raise NotImplementedError
