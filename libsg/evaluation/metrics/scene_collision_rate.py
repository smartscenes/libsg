from .base import EvaluationBase
from libsg.scene import Scene
from libsg.simulator import Simulator
from libsg.simscene import SimScene


class SceneCollisionRate(EvaluationBase):
    def __call__(self, inp, scene: Scene):
        collision_count = 0
        with Simulator(mode="direct", verbose=False, use_y_up=False) as sim:
            sim_scene = SimScene(sim, scene, self.__model_db.config)
            sim.step()
            for mi in scene.model_instances:
                contacts = sim.get_contacts(obj_id_a=mi.id, include_collision_with_static=True)
                if contacts:
                    collision_count += 1
                    break

        return collision_count
