import os
import numpy as np
import cv2
import pyrender
import random
from .config import Config
from numpy.linalg import inv
from abc import ABC, abstractmethod

class DatasetGenerator(ABC):
    def __init__(self, output_dir, image_size=(800, 600)):
        self.output_dir = output_dir
        self.image_size = image_size
        self.config = Config.get_instance()
        os.makedirs(output_dir, exist_ok=True)
        
    @abstractmethod
    def process_scene(self, scene_data, visualize=False):
        """Process một scene và tạo data"""
        pass
    
    @abstractmethod
    def save_metadata(self):
        """Lưu metadata của dataset"""
        pass
    
    def add_noise_and_augmentation(self, image):
        """Thêm nhiễu và augmentation cho ảnh"""
        augmentation_config = self.config.augmentation

        # Gaussian noise
        noise = np.random.normal(0, augmentation_config["noise_sigma"], image.shape)
        image = np.clip(image + noise, 0, 1)
        
        # Điều chỉnh độ sáng
        min_brightness = augmentation_config["brightness_range"]["min"]
        max_brightness = augmentation_config["brightness_range"]["max"]
        brightness = random.uniform(min_brightness, max_brightness)
        MIN_PIXEL_VALUE = 0
        MAX_PIXEL_VALUE = 1
        image = np.clip(image * brightness, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
        
        # Constants for contrast adjustment
        MIDDLE_INTENSITY = 0.5
        MIN_PIXEL_VALUE = 0
        MAX_PIXEL_VALUE = 1

        # Điều chỉnh độ tương phản
        min_contrast = augmentation_config["contrast_range"]["min"]
        max_contrast = augmentation_config["contrast_range"]["max"]
        contrast = random.uniform(min_contrast, max_contrast)
        image = np.clip((image - MIDDLE_INTENSITY) * contrast + MIDDLE_INTENSITY, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
        
        # Mô phỏng thiếu sáng với gamma correction
        if random.random() < augmentation_config["gamma_probability"]:
            min_gamma = augmentation_config["gamma_range"]["min"]
            max_gamma = augmentation_config["gamma_range"]["max"]
            gamma = random.uniform(min_gamma, max_gamma)
            image = np.power(image, gamma)
        
        return image
    
    def render_scene(self, scene):
        """Render một scene"""
        r = pyrender.OffscreenRenderer(*self.image_size)
        color, depth = r.render(scene)
        return color, depth

    def visualize_bounding_boxes(self, image, objects_metadata, camera_pose, projection):
        """Vẽ bounding boxes lên ảnh và tạo annotations"""
        vis_image = image.copy()
        
        for obj in objects_metadata:
            obb = obj['obb']
            # Tạo các điểm của box trong local space
            corners = np.array(obb['vertices'])
                        
            # Project sang 2D
            points_2d = self._project_points(corners, camera_pose, projection)
            if points_2d is None:
                continue
            
            bbox_2d = self._calculate_2d_obb(points_2d)
            # Vẽ box và label
            self._draw_box(vis_image, bbox_2d, points_2d, obj['object_type'])
        
        return vis_image
    
    def _project_points(self, points, view, projection):
        """Project các điểm 3D thành 2D"""
        # Convert points to homogeneous coordinates
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Apply view and projection transforms
        transform = projection @ inv(view)
        p = transform @ points_h.T
        
        # Perspective division
        p = p / p[3]
        
        # Convert to screen coordinates
        w, h = self.image_size
        screen_x = (w/2 * p[0] + w/2)  # Map [-1,1] to [0,width]
        screen_y = h - (h/2 * p[1] + h/2)  # Map [-1,1] to [0,height] with top-left origin
        
        return np.vstack((screen_x, screen_y)).T

    def _calculate_2d_obb(self, points_2d):
        # Chuyển points thành numpy array để tối ưu tính toán
        points_array = np.array(points_2d)
        
        # Tính ma trận hiệp phương sai
        mean = np.mean(points_array, axis=0)
        centered_points = points_array - mean
        cov_matrix = np.cov(centered_points.T)
        
        # Tính vector riêng để xác định hướng chính
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Lấy vector riêng chính (ứng với giá trị riêng lớn nhất)
        major_axis = eigenvectors[:, 1]
        angle = np.arctan2(major_axis[1], major_axis[0])
        
        # Xoay các điểm về hệ trục tọa độ chính
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        rotated_points = np.dot(centered_points, rotation_matrix)
        
        # Tính kích thước của OBB
        min_coords = np.min(rotated_points, axis=0)
        max_coords = np.max(rotated_points, axis=0)
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        
        # Tính điểm trung tâm trong hệ tọa độ gốc
        center = tuple(mean)

        # Tạo các điểm góc của OBB
        corners = np.array([
            [-width/2, -height/2],
            [width/2, -height/2],
            [width/2, height/2],
            [-width/2, height/2]
        ])
        
        # Xoay và dịch chuyển các góc
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
    
        rotated_corners = np.dot(corners, rotation_matrix.T) + center
    
        return rotated_corners

    def _draw_box(self, image, box2d_coords, points_2d, object_type):
        """Vẽ box và label lên ảnh"""
        
        # Tạo màu ngẫu nhiên nhưng cố định cho mỗi loại object
        color = self._get_color_for_object_type(object_type)
        
        # Vẽ 2D oriented bounding box
        for i in range(4):
            pt1 = tuple(map(int, box2d_coords[i]))
            pt2 = tuple(map(int, box2d_coords[(i+1)%4]))
            cv2.line(image, pt1, pt2, color, 2)
        
        # Vẽ các điểm góc của 3D box
        points = points_2d.astype(np.int32)
        for i in range(4):
            cv2.line(image, tuple(points[i]), tuple(points[(i+1)%4]), color, 1)
            cv2.line(image, tuple(points[i+4]), tuple(points[(i+1)%4+4]), color, 1)
            cv2.line(image, tuple(points[i]), tuple(points[i+4]), color, 1)
        
        # Vẽ label với background
        label = f"{object_type}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x_min = int(min(box2d_coords[:, 0]))
        y_min = int(min(box2d_coords[:, 1]))
        cv2.rectangle(image, (x_min, y_min - label_height - 5), 
                    (x_min + label_width, y_min), color, -1)
        cv2.putText(image, label, (x_min, y_min - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Vẽ các điểm góc để debug
        for i, point in enumerate(points_2d):
            self._draw_point(image, point, str(i), color)
            
    def _draw_point(self, image, point, label, color):
        cv2.circle(image, tuple(map(int, point)), 3, color, -1)
        cv2.putText(image, label, 
            tuple(map(int, point + np.array([5, 5]))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    def _get_color_for_object_type(self, object_type):
        """Tạo màu cố định cho mỗi loại object"""
        if not hasattr(self, 'color_map'):
            self.color_map = {}
        
        if object_type not in self.color_map:
            self.color_map[object_type] = tuple(np.random.randint(0, 255, 3).tolist())
        
        return self.color_map[object_type]