import numpy as np
import random
import pyrender
import trimesh
from .camera_positioner import CameraPositioner
from .config import Config
from scipy.spatial.transform import Rotation
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

class BaseSceneGenerator(ABC):
    def __init__(self, model_paths: Dict[str, str], image_size: Tuple[int, int]=(800, 600)):
        self.model_paths = model_paths
        self.image_size = image_size
        self.meshes = self._load_meshes()
        self.config = Config.get_instance()
        self.camera_positioner = CameraPositioner()

    @abstractmethod
    def generate_scene(self, min_objects=2, max_objects=5) -> Tuple[pyrender.Scene, List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """Abstract method that each specific generator must implement"""
        raise NotImplementedError
    
    @abstractmethod
    def _get_camera_parameters(self, center: float, scene_size: float) -> Dict[str, Any]:
        """Each generator should implement its own camera parameters"""
        pass

    def calculate_optimal_camera_position(self, scene):
        """Base implementation of camera positioning"""
        bounds = self.calculate_scene_bounds(scene)
        center = (bounds[0] + bounds[1]) / 2
        size = np.linalg.norm(bounds[1] - bounds[0])
        
        # Get base camera parameters
        params = self._get_camera_parameters(center, size)
        
        # Calculate camera position
        camera_position = self.camera_positioner.calculate_camera_position(
            center=params['center'],
            distance=params['distance'],
            phi=params['phi'],
            theta=params['theta']
        )
        
        # Calculate camera rotation
        rotation = self.camera_positioner.calculate_camera_rotation(
            camera_position, 
            params['look_at_point']
        )
        
        # Create final camera pose
        return self.camera_positioner.create_camera_pose(camera_position, rotation)
      
    def _load_meshes(self):
        """Load tất cả models"""
        meshes = {}
        for name, path in self.model_paths.items():
            mesh = trimesh.load(path)
            if isinstance(mesh, trimesh.Scene):
                # Merge tất cả meshes trong scene thành một mesh duy nhất
                merged_mesh = trimesh.util.concatenate(
                    [geometry for geometry in mesh.geometry.values()]
                )
                meshes[name] = merged_mesh
                continue
            meshes[name] = mesh
        return meshes
    
    def apply_random_transformations(self, mesh, translation_range=None):
        """Áp dụng các biến đổi ngẫu nhiên cho mesh"""     
        mesh = mesh.copy()
        scene_config = self.config.scene_generation

        # Scale ngẫu nhiên
        scale = random.uniform(
            scene_config["scale_range"]["min"], 
            scene_config["scale_range"]["max"])
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= scale
        mesh.apply_transform(scale_matrix)
        
        # Rotation ngẫu nhiên
        rotation = Rotation.random()
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation.as_matrix()
        mesh.apply_transform(rotation_matrix)
        
        # Translation ngẫu nhiên
        if translation_range is None:
            translation_range = {
                "min": scene_config["translation_range"]["min"],
                "max": scene_config["translation_range"]["max"]
            }
        
        translation = np.random.uniform(
            translation_range["min"], 
            translation_range["max"], 
            size=3
        )
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation
        mesh.apply_transform(translation_matrix)
        
        return mesh, {
            'scale': scale,
            'rotation': rotation.as_quat().tolist(),
            'translation': translation.tolist()
        }

    def calculate_scene_bounds(self, scene):
        """Tính toán bounds của toàn bộ scene"""
        all_points = []
        
        # Lấy tất cả các điểm từ tất cả các nodes trong scene
        for node in scene.nodes:
            if node.mesh is not None:
                # Lấy transform của node
                transform = node.matrix
                
                # Lấy tất cả vertices từ tất cả primitives của mesh
                for primitive in node.mesh.primitives:
                    mesh_points = primitive.positions
                    # Áp dụng transform
                    transformed_points = np.dot(transform, 
                                            np.hstack((mesh_points, 
                                                    np.ones((mesh_points.shape[0], 1)))).T).T[:, :3]
                    all_points.extend(transformed_points)
        
        if not all_points:
            return np.array([[-1, -1, -1], [1, 1, 1]])
            
        all_points = np.array(all_points)
        
        # Thêm margin để đảm bảo objects không bị cắt
        bounds_min = np.min(all_points, axis=0)
        bounds_max = np.max(all_points, axis=0)
        margin = (bounds_max - bounds_min) * 0.1  # 10% margin
        
        return np.array([bounds_min - margin, bounds_max + margin])
    
    def _setup_camera(self, scene):
        camera_config = self.config.camera
        camera = pyrender.PerspectiveCamera(
            yfov=np.radians(camera_config["fov"]), 
            aspectRatio=self.image_size[0] / self.image_size[1]
        )
        camera_pose = self.calculate_optimal_camera_position(scene)
        scene.add(camera, pose=camera_pose)
        return {
            'pose': camera_pose,
            'projection': camera.get_projection_matrix()
        }
    
    def _add_lights(self, scene):
        """Thêm ánh sáng vào scene"""
        light_config = self.config.lighting
        min_lights = light_config['num_lights']["min"]
        max_lights = light_config['num_lights']["max"]

        min_light_color = light_config['light_color']["min"]
        max_light_color = light_config['light_color']["max"]

        min_light_intensity = light_config['light_intensity']["min"]
        max_light_intensity = light_config['light_intensity']["max"]

        light = pyrender.DirectionalLight(
            color=np.random.uniform(min_light_color, max_light_color, size=3),
            intensity=random.uniform(min_light_intensity, max_light_intensity)
        )
        
        for _ in range(random.randint(min_lights, max_lights)):
            light_pose = np.eye(4)
            light_position_range = light_config['light_position_range']
            light_pose[:3, 3] = np.random.uniform(
                light_position_range["min"], 
                light_position_range["max"], 
                size=3
            )
            scene.add(light, pose=light_pose)