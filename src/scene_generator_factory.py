from .scene import *
from .scene_generator import BaseSceneGenerator
from typing import Tuple, Dict

class SceneGeneratorFactory:
    """Factory for creating different types of scene generators"""
    
    @staticmethod
    def create_generator(generator_type: str, model_paths: Dict[str, str], image_size: Tuple[int, int]=(800, 600)) -> BaseSceneGenerator:
        generators = {
            'random_clutter': RandomClutterSceneGenerator,
            # temporarily removed other generators
            # 'partially_occluded': PartiallyOccludedSceneGenerator,
            # 'multi_level': MultiLevelSceneGenerator
        }
        
        generator_class = generators.get(generator_type)
        if generator_class is None:
            raise ValueError(f"Unknown generator type: {generator_type}")
            
        return generator_class(model_paths, image_size)