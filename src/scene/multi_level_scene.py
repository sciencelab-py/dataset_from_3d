import pyrender
import random
import numpy as np
from typing import Dict, Any
from ..scene_generator import BaseSceneGenerator

# Currently, the factory only supports the RandomClutterSceneGenerator
class MultiLevelSceneGenerator(BaseSceneGenerator):
    """Generates scenes with objects at different height levels"""
    def __init__(self, model_paths, image_size = ...):
        super().__init__(model_paths, image_size)
        self.level_gap = 0.05  # Height difference between levels
        self.max_levels = 3  # Maximum number of levels
        self.center_z = 0.05  # Center level height
        self.look_at_z = 0.05  # Look at point height

        self.plane_range = 0.5  # Range for X and Y coordinates
        self.height_variation = 0.1  # Allowed vertical variation within a level
    
    def generate_scene(self, min_objects=2, max_objects=5):
        scene = pyrender.Scene(bg_color=np.random.uniform(low=0.0, high=1.0, size=3))
        objects_metadata = []
        
        num_objects = random.randint(min_objects, max_objects)
        levels = self._generate_height_levels(num_objects)
        
        for i, level in enumerate(levels):
            object_type = random.choice(list(self.meshes.keys()))
            mesh_copy = self.meshes[object_type].copy()

            translation_range = {
                "min": [-self.plane_range, -self.plane_range, level['height']],
                "max": [self.plane_range, self.plane_range, level['height'] + self.height_variation]
            }
            
            mesh_copy, transform_metadata = self.apply_random_transformations(
                mesh_copy, translation_range=translation_range
            )
            
            obb = mesh_copy.bounding_box_oriented
            mesh = pyrender.Mesh.from_trimesh(mesh_copy)
            node = scene.add(mesh)
            
            objects_metadata.append({
                'object_id': i,
                'object_type': object_type,
                'transformations': transform_metadata,
                'level': level['level'],
                'obb': {
                    'center': obb.centroid.tolist(),
                    'extents': obb.extents.tolist(),
                    'transform': obb.transform.tolist(),
                    'vertices': obb.vertices.tolist()
                }
            })
        
        camera = self._setup_camera(scene)
        self._add_lights(scene)
        
        return scene, objects_metadata, camera['pose'], camera['projection']
    
    def _generate_height_levels(self, num_objects):
        """Generate different height levels for objects"""
        levels = []
        max_levels = min(self.max_levels, num_objects)  # Maximum 3 levels
        objects_per_level = num_objects // max_levels
        
        for level in range(max_levels):
            height = level * self.level_gap  # 1.0 unit height difference between levels
            for _ in range(objects_per_level):
                levels.append({
                    'level': level,
                    'height': height
                })
        
        # Add remaining objects to random levels
        remaining = num_objects - len(levels)
        for _ in range(remaining):
            level = random.randint(0, max_levels - 1)
            levels.append({
                'level': level,
                'height': level * self.level_gap
            })
        
        random.shuffle(levels)
        return levels

    def _get_camera_parameters(self, center: float, scene_size: float) -> Dict[str, Any]:
        """Camera parameters for multi-level scenes"""
        camera_config = self.config.camera
        
        # Calculate phi (vertical angle)
        phi_min = np.radians(15)  # Lower angle to see stacking clearly
        phi_max = np.radians(45)  # Higher angle to see top objects
        phi = np.random.uniform(phi_min, phi_max)
        
        # Raise center point and look_at point for better vertical perspective
        center = np.array([0, 0, scene_size * self.center_z])
        look_at_point = np.array([0, 0, scene_size * self.look_at_z])
        
        return {
            'distance': scene_size * camera_config["distance_weight"],
            'phi': phi,
            'theta': np.random.uniform(0, 2 * np.pi),
            'center': center,
            'look_at_point': look_at_point
        }