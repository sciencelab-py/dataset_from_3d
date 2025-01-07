import pyrender
import random
import numpy as np
from typing import Dict, Any
from ..scene_generator import BaseSceneGenerator

# Currently, the factory only supports the RandomClutterSceneGenerator
class PartiallyOccludedSceneGenerator(BaseSceneGenerator):
    """Generates scenes where objects partially occlude each other"""
    def __init__(self, model_paths, image_size = ...):
        super().__init__(model_paths, image_size)

        self.min_radius = 0.1
        self.max_radius = 0.3

        self.postion_offset = 0.1
        self.min_z = -0.25
        self.max_z = 0.25

        self.phi_min = np.radians(0)
        self.phi_max = np.radians(0)
    
    def generate_scene(self, min_objects=2, max_objects=5):
        scene = pyrender.Scene(bg_color=np.random.uniform(low=0.0, high=1.0, size=3))
        objects_metadata = []
        
        num_objects = random.randint(min_objects, max_objects)
        base_positions = self._generate_overlapping_positions(num_objects)
        
        for i, position in enumerate(base_positions):
            object_type = random.choice(list(self.meshes.keys()))
            mesh_copy = self.meshes[object_type].copy()
            
            # Define translation range centered around the base position
            translation_range = {
                "min": [position[0] - self.postion_offset, position[1] - self.postion_offset, self.min_z],
                "max": [position[0] + self.postion_offset, position[1] + self.postion_offset, self.max_z]
            }
            
            mesh_copy, transform_metadata = self.apply_random_transformations(
                mesh_copy, translation_range=translation_range
            )
            
            obb = mesh_copy.bounding_box_oriented
            mesh = pyrender.Mesh.from_trimesh(mesh_copy)
            scene.add(mesh)
            
            objects_metadata.append({
                'object_id': i,
                'object_type': object_type,
                'transformations': transform_metadata,
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
    
    def _generate_overlapping_positions(self, num_objects):
        """Generate positions that ensure partial overlap between objects"""
        positions = []
        
        for i in range(num_objects):
            if i == 0:
                # First object at scene center
                positions.append([0, 0])
            else:
                # Generate position with partial overlap with at least one existing object
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(self.min_radius, self.max_radius)  # Ensure some overlap
                position = [
                    distance * np.cos(angle),
                    distance * np.sin(angle)
                ]
                positions.append(position)
        
        return positions
    
    def _get_camera_parameters(self, center: float, scene_size: float) -> Dict[str, Any]:
        """Camera parameters for partially occluded scenes"""
        camera_config = self.config.camera
        
        # Calculate phi (vertical angle)
        phi = np.random.uniform(self.phi_min, self.phi_max)
        
        # Add slight random offset to center
        center_offset = np.random.uniform(-self.postion_offset, self.postion_offset, size=3)   

        return {
            'distance': scene_size * camera_config["distance_weight"],
            'phi': phi,
            'theta': np.random.uniform(0, 2 * np.pi),
            'center': center_offset,
            'look_at_point': np.zeros(3)  # Look at the scene center
        }