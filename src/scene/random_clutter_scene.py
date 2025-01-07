import pyrender
import random
import numpy as np
from typing import Dict, Any
from ..scene_generator import BaseSceneGenerator

class RandomClutterSceneGenerator(BaseSceneGenerator):
    """Generates scenes with randomly cluttered objects, simulating messy real-world environments"""

    def __init__(self, model_paths, image_size = ...):
        super().__init__(model_paths, image_size)

        # Define base plane for objects
        self.plane_size = 0.5  # Base plane size
        self.base_height = 0
        self.max_attempts = 50  # Maximum attempts to place each object
 
    def generate_scene(self, min_objects=2, max_objects=5):
        scene = pyrender.Scene(bg_color=np.random.uniform(low=0.0, high=1.0, size=3))
        objects_metadata = []
        placed_objects = []
        
        # Generate random number of objects (more objects for cluttered scene)
        num_objects = random.randint(min_objects, max_objects)

        for i in range(num_objects):
            object_type = random.choice(list(self.meshes.keys()))
            
            # Try different positions until no intersection
            for attempt in range(self.max_attempts):
                mesh_copy = self.meshes[object_type].copy()
                
                # Define random position on the base plane
                translation_range = {
                    "min": [-self.plane_size/2, -self.plane_size/2, self.base_height],
                    "max": [self.plane_size/2, self.plane_size/2, self.base_height + 0.5]
                }
                
                mesh_copy, transform_metadata = self.apply_random_transformations(
                    mesh_copy, translation_range=translation_range
                )
                
                # Check intersection with previously placed objects
                intersects = False
                for placed_obj in placed_objects:
                    if mesh_copy.bounds[0][0] <= placed_obj.bounds[1][0] and \
                       mesh_copy.bounds[1][0] >= placed_obj.bounds[0][0] and \
                       mesh_copy.bounds[0][1] <= placed_obj.bounds[1][1] and \
                       mesh_copy.bounds[1][1] >= placed_obj.bounds[0][1] and \
                       mesh_copy.bounds[0][2] <= placed_obj.bounds[1][2] and \
                       mesh_copy.bounds[1][2] >= placed_obj.bounds[0][2]:
                        intersects = True
                        break
                
                if not intersects or attempt == self.max_attempts - 1:
                    placed_objects.append(mesh_copy)
                    break
            
            # Calculate oriented bounding box
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
        
        # Setup camera and lights
        camera = self._setup_camera(scene)
        self._add_lights(scene)
        
        pyrender.Viewer(scene)
        
        return scene, objects_metadata, camera['pose'], camera['projection']
    
    def _get_camera_parameters(self, center: float, scene_size: float) -> Dict[str, Any]:
        """Camera parameters for cluttered scenes"""
        camera_config = self.config.camera
        
        # Giới hạn góc phi theo config
        phi_min = np.radians(camera_config["phi_range"]["min"])
        phi_max = np.radians(camera_config["phi_range"]["max"])
        phi = np.random.uniform(phi_min, phi_max)
        
        # Calculate distance and other parameters
        return {
            'distance': scene_size * camera_config["distance_weight"],
            'phi': phi,
            'theta': np.random.uniform(0, 2 * np.pi),
            'center': center,  # Use scene center
            'look_at_point': center  # Look at scene center
        }