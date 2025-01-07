import numpy as np
import random
import pyrender
import trimesh
from .config import Config
from scipy.spatial.transform import Rotation

class SceneGenerator:
    def __init__(self, model_paths, image_size=(800, 600)):
        self.model_paths = model_paths
        self.image_size = image_size
        self.meshes = self._load_meshes()
        self.config = Config.get_instance()
        
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
    
    def apply_random_transformations(self, mesh):
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
        translation = np.random.uniform(
            scene_config["translation_range"]["min"], 
            scene_config["translation_range"]["max"], 
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

    def calculate_optimal_camera_position(self, scene):
        """Tính toán vị trí camera tối ưu để hiển thị tất cả objects"""
        bounds = self.calculate_scene_bounds(scene)
        center = (bounds[0] + bounds[1]) / 2
        size = np.linalg.norm(bounds[1] - bounds[0])
        
        # Get camera config
        camera_config = self.config.camera
        
        # Khoảng cách camera để đảm bảo tất cả objects nằm vừa trong viewport
        distance = size * camera_config["distance_weight"]
        
        # Tạo vị trí camera ngẫu nhiên nhưng giữ khoảng cách không đổi
        theta = np.random.uniform(0, 2 * np.pi)
        # Giới hạn góc phi theo config
        phi_min = np.radians(camera_config["phi_range"]["min"])
        phi_max = np.radians(camera_config["phi_range"]["max"])
        phi = np.random.uniform(phi_min, phi_max)
        
        camera_position = center + distance * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        # Tạo ma trận look_at
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_position
        
        # Hướng camera về tâm scene
        direction = center - camera_position
        direction = direction / np.linalg.norm(direction)
        
        # Tính vector up (không song song với direction)
        up = np.array([0, 0, 1])
        if np.abs(np.dot(up, direction)) > 0.9:
            up = np.array([0, 1, 0])
            
        # Tạo ma trận rotation cho camera
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction)
        up = up / np.linalg.norm(up)
        
        rotation = np.eye(3)
        rotation[:, 0] = right
        rotation[:, 1] = up
        rotation[:, 2] = -direction
        
        camera_pose[:3, :3] = rotation
        
        return camera_pose

    def generate_scene(self, min_objects=2, max_objects=5):
        """Tạo một scene mới"""
        # Set random background color with values between 0 and 1
        scene = pyrender.Scene(bg_color=np.random.uniform(low=0.0, high=1.0, size=3))
        objects_metadata = []
        
        num_objects = random.randint(min_objects, max_objects)
        
        # Thêm objects
        for i in range(num_objects):
            # Chọn ngẫu nhiên một model
            object_type = random.choice(list(self.meshes.keys()))
            mesh_copy = self.meshes[object_type].copy()
            # Áp dụng transforms
            mesh_copy, transform_metadata = self.apply_random_transformations(mesh_copy)

            # Tính oriented bounding box
            obb = mesh_copy.bounding_box_oriented

            mesh = pyrender.Mesh.from_trimesh(mesh_copy)
            node = scene.add(mesh)
            
            # Lưu metadata của object
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

        # Get camera config
        camera_config = self.config.camera
        camera_fov = np.radians(camera_config["fov"])
        
        # Thiết lập camera
        camera = pyrender.PerspectiveCamera(yfov=camera_fov, aspectRatio=self.image_size[0] / self.image_size[1])
        camera_pose = self.calculate_optimal_camera_position(scene)
        camera_node = scene.add(camera, pose=camera_pose)
        
        # Thêm ánh sáng
        self._add_lights(scene)
        
        return scene, objects_metadata, camera_pose, camera.get_projection_matrix()
    
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