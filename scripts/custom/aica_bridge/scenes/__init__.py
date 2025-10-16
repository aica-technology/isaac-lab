from .compliant_scene import CompliantScene
from .lift_scene import LiftScene
from .force_limit_scene import ForceLimitScene
from .basic_scene import BasicScene

scenes = {"compliant_scene": CompliantScene, "lift_scene": LiftScene, "force_limit_scene": ForceLimitScene, "basic_scene": BasicScene}
