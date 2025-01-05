import numpy as np
from scipy.spatial.transform import Rotation
import random
import cv2
import os
from PIL import Image
import json
import pyrender
import trimesh
import datetime
import matplotlib.pyplot as plt

class DatasetGenerator:
    def __init__(self, 
                 model_paths,  # Dict với key là tên model, value là đường dẫn
                 output_dir,
                 image_size=(640, 480),
                 min_angle=30):
        self.model_paths = model_paths
        self.output_dir = output_dir
        self.image_size = image_size
        self.min_angle = min_angle
        self.metadata = []
        
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tất cả models
        self.meshes = {}
        for name, path in model_paths.items():
            loaded_obj = trimesh.load(path)
            if isinstance(loaded_obj, trimesh.Scene):
                # Nếu là scene, lưu trữ scene
                self.meshes[name] = {
                    'type': 'scene',
                    'data': loaded_obj
                }
            else:
                # Nếu là mesh đơn lẻ
                self.meshes[name] = {
                    'type': 'mesh',
                    'data': loaded_obj
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
        
        # Khoảng cách camera để đảm bảo tất cả objects nằm vừa trong viewport
        distance = size * 0.75
        
        # Tạo vị trí camera ngẫu nhiên nhưng giữ khoảng cách không đổi
        theta = np.random.uniform(0, 2 * np.pi)
        # Giới hạn góc phi để không nhìn quá cao hoặc quá thấp
        phi = np.random.uniform(np.pi/6, np.pi/3)
        
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

    def convert_to_pyrender(self, obj):
        """Chuyển đổi object sang pyrender Mesh hoặc Scene"""
        if isinstance(obj, trimesh.Scene):
            # Chuyển đổi scene sang pyrender Scene
            py_scene = pyrender.Scene()
            
            # Duyệt qua tất cả edges trong graph
            for geometry in obj.geometry.values():
                try:
                    # Chuyển mesh sang pyrender
                    py_mesh = pyrender.Mesh.from_trimesh(geometry)
                    # Thêm vào scene
                    py_scene.add(py_mesh)
                except Exception as e:
                    print(f"Warning: Không thể chuyển đổi mesh: {str(e)}")
                    continue

            return py_scene
        else:
            # Chuyển đổi mesh đơn lẻ
            try:
                return pyrender.Mesh.from_trimesh(obj)
            except Exception as e:
                print(f"Warning: Không thể chuyển đổi mesh: {str(e)}")
                return None

    def create_scene_variations(self, min_objects=2, max_objects=5):
        """Tạo scene với nhiều loại objects và scenes khác nhau"""
        scene = pyrender.Scene(bg_color=np.random.uniform(0, 1, 3))
        objects_metadata = []
        
        # Số lượng objects/scenes ngẫu nhiên
        num_objects = random.randint(min_objects, max_objects)
        
        for i in range(num_objects):
            try:
                # Chọn ngẫu nhiên một model
                object_type = random.choice(list(self.meshes.keys()))
                original_obj = self.meshes[object_type]['data']
                
                # Áp dụng transforms
                transformed_obj, transform_metadata = self.apply_random_transformations(original_obj)
                
                # Chuyển đổi sang pyrender và thêm vào scene
                if self.meshes[object_type]['type'] == 'scene':
                    py_obj = self.convert_to_pyrender(transformed_obj)
                    if py_obj is not None:
                        # Thêm tất cả nodes từ converted scene
                        for node in py_obj.nodes:
                            if node.mesh is not None:
                                scene.add(node.mesh, pose=node.matrix)
                else:
                    py_mesh = self.convert_to_pyrender(transformed_obj)
                    if py_mesh is not None:
                        scene.add(py_mesh)
                
                # Lưu metadata
                objects_metadata.append({
                    'object_id': i,
                    'object_type': object_type,
                    'is_scene': self.meshes[object_type]['type'] == 'scene',
                    'transformations': transform_metadata
                })
            except Exception as e:
                print(f"Warning: Không thể xử lý object {object_type}: {str(e)}")
                continue
        
        # Kiểm tra nếu scene trống
        if len(scene.nodes) == 0:
            raise ValueError("Không thể tạo scene hợp lệ - không có objects nào được thêm thành công")
        
        # Thêm camera với góc nhìn phù hợp
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3)
        camera_pose = self.calculate_optimal_camera_position(scene)
        scene.add(camera, pose=camera_pose)
        
        # Thêm ánh sáng từ nhiều hướng
        light_positions = [
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1]
        ]
        
        for light_pos in light_positions:
            light = pyrender.DirectionalLight(
                color=np.random.uniform(0.8, 1.0, 3),
                intensity=random.uniform(1.0, 2.0)
            )
            light_pose = np.eye(4)
            light_pose[:3, 3] = np.array(light_pos) * 3  # Nhân với 3 để đặt đèn xa hơn
            scene.add(light, pose=light_pose)
        
        return scene, objects_metadata, camera_pose

    def apply_random_transformations(self, obj):
        """Áp dụng các biến đổi ngẫu nhiên cho mesh hoặc scene"""
        if isinstance(obj, trimesh.Scene):
            # Tạo bản sao của scene
            new_scene = obj.copy()
            
            # Tạo transform ngẫu nhiên
            scale = random.uniform(0.8, 1.2)
            rotation = Rotation.random()
            translation = np.random.uniform(-0.5, 0.5, 3)
            
            # Tạo ma trận transform tổng hợp
            transform = np.eye(4)
            transform[:3, :3] = rotation.as_matrix() * scale
            transform[:3, 3] = translation
            
            try:
                # Áp dụng transform cho scene
                new_scene.apply_transform(transform)
            except Exception as e:
                print(f"Warning: Không thể áp dụng transform cho scene: {str(e)}")
            
            return new_scene, {
                'scale': scale,
                'rotation': rotation.as_quat().tolist(),
                'translation': translation.tolist()
            }
        else:
            # Xử lý mesh đơn lẻ
            try:
                mesh = obj.copy()
                
                # Scale
                scale = random.uniform(0.8, 1.2)
                scale_matrix = np.eye(4)
                scale_matrix[:3, :3] *= scale
                mesh.apply_transform(scale_matrix)
                
                # Rotation
                rotation = Rotation.random()
                rotation_matrix = np.eye(4)
                rotation_matrix[:3, :3] = rotation.as_matrix()
                mesh.apply_transform(rotation_matrix)
                
                # Translation
                translation = np.random.uniform(-0.5, 0.5, 3)
                translation_matrix = np.eye(4)
                translation_matrix[:3, 3] = translation
                mesh.apply_transform(translation_matrix)
                
                return mesh, {
                    'scale': scale,
                    'rotation': rotation.as_quat().tolist(),
                    'translation': translation.tolist()
                }
            except Exception as e:
                print(f"Warning: Không thể áp dụng transform cho mesh: {str(e)}")
                return obj.copy(), {
                    'scale': 1.0,
                    'rotation': [0, 0, 0, 1],
                    'translation': [0, 0, 0]
                }
    
    def add_noise_and_augmentation(self, image):
        """Thêm nhiễu và augmentation cho ảnh"""
        # Gaussian noise
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        # Điều chỉnh độ sáng
        brightness = random.uniform(0.7, 1.3)
        image = np.clip(image * brightness, 0, 1)
        
        # Điều chỉnh độ tương phản
        contrast = random.uniform(0.8, 1.2)
        image = np.clip((image - 0.5) * contrast + 0.5, 0, 1)
        
        # Mô phỏng thiếu sáng
        if random.random() < 0.3:
            image = np.power(image, random.uniform(1.0, 2.0))
        
        return image

    def generate_dataset(self, num_samples):
        """Tạo dataset"""
        for i in range(num_samples):
            # Tạo scene với nhiều objects
            scene, objects_metadata, camera_pose = self.create_scene_variations()
            
            # Render scene
            r = pyrender.OffscreenRenderer(*self.image_size)
            color, depth = r.render(scene)
            
            # Chuyển đổi và xử lý ảnh
            image = color.astype(float) / 255.0
            image = self.add_noise_and_augmentation(image)
            
            # Lưu ảnh RGB
            image_path = os.path.join(self.output_dir, f'sample_{i:06d}.png')
            Image.fromarray((image * 255).astype(np.uint8)).save(image_path)
            
            # Normalize và lưu depth map
            depth_path = os.path.join(self.output_dir, f'depth_{i:06d}.png')
            
            if depth is not None:
                # Normalize depth vào range [0, 255]
                depth_normalized = ((depth - depth.min()) * 255 / 
                                (depth.max() - depth.min() + 1e-8)).astype(np.uint8)
                
                # Lưu dưới dạng ảnh grayscale
                cv2.imwrite(depth_path, depth_normalized)
                
                # Tùy chọn: lưu bản có màu sử dụng matplotlib
                depth_color_path = os.path.join(self.output_dir, f'depth_color_{i:06d}.png')
                plt.figure(figsize=(8, 6))
                plt.imshow(depth, cmap='viridis')
                plt.colorbar()
                plt.savefig(depth_color_path)
                plt.close()
            
            # Lưu metadata
            metadata = {
                'image_path': image_path,
                'depth_path': depth_path,
                'timestamp': datetime.datetime.now().isoformat(),
                'objects': objects_metadata,
                'camera': {
                    'pose': camera_pose.tolist(),
                    'fov': np.pi/3,
                    'aspect_ratio': self.image_size[0] / self.image_size[1]
                },
                'scene_info': {
                    'num_objects': len(objects_metadata),
                    'image_size': self.image_size,
                    'depth_stats': {
                        'min': float(depth.min()) if depth is not None else None,
                        'max': float(depth.max()) if depth is not None else None,
                        'mean': float(depth.mean()) if depth is not None else None
                    }
                }
            }
            self.metadata.append(metadata)
            
            if (i + 1) % 10 == 0:
                print(f"Đã tạo {i + 1}/{num_samples} mẫu")
                self._save_metadata()
        
        self._save_metadata()

    def _save_metadata(self):
        """Lưu metadata vào file JSON"""
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

# Sử dụng
if __name__ == "__main__":
    # Dictionary chứa paths tới các models khác nhau
    model_paths = {
        'blue_sample': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\am-5401_blue\\am-5401_blue.obj",
        'yellow_sample': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\am-5401_yellow\\am-5401_yellow.obj",
        'red_sample': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\am-5401_red\\am-5401_red.obj",
        'blue_specimen': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\blue_specimen\\blue_specimen.obj",
        'red_specimen': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\red_specimen\\red_specimen.obj",
    }
    
    output_dir = "D:\\Projects\\ScienceLab\\ftc25-test\\dataset"
    
    generator = DatasetGenerator(
        model_paths=model_paths,
        output_dir=output_dir,
    )
    
    generator.generate_dataset(num_samples=3)