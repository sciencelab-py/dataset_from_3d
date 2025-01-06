from ..dataset_generator import DatasetGenerator
import os
import numpy as np
from PIL import Image
import cv2

class YOLODatasetGenerator(DatasetGenerator):
    def __init__(self, output_dir, image_size=(800, 600)):
        super().__init__(output_dir, image_size)
        self.categories = list()
        self.image_id = 0
        
        # Tạo thư mục images và labels
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
    def process_scene(self, scene_data, visualize=False):
        """Process scene và tạo YOLO annotations"""
        scene, objects_metadata, camera_pose, projection = scene_data
        
        # Render scene
        color, depth = self.render_scene(scene)
        
        # Process ảnh
        image = color.astype(float) / 255.0
        image = self.add_noise_and_augmentation(image)
        
        # Lưu ảnh
        image_filename = f'image_{self.image_id:06d}.png'
        label_filename = f'image_{self.image_id:06d}.txt'
        
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, label_filename)
        
        Image.fromarray((image * 255).astype(np.uint8)).save(image_path)

        if visualize:
            # Tạo ảnh với bounding boxes để visualize
            vis_image = self.visualize_bounding_boxes(color, objects_metadata, camera_pose, projection)
            vis_filename = f'vis_{self.image_id:06d}.png'
            vis_path = os.path.join(self.images_dir, vis_filename)
            Image.fromarray(vis_image).save(vis_path)
            
        # Tạo và lưu annotation theo format YOLO
        with open(label_path, 'w') as f:
            for obj in objects_metadata:
                if obj['object_type'] not in self.categories:
                    self.categories.append(obj['object_type'])
                category_id = self.categories.index(obj['object_type'])
                
                # Tính toán bbox
                obb = obj['obb']
                corners = self._get_box_corners(obb['extents'], obb['transform'])
                
                # Project sang 2D
                points_2d = self._project_points(corners, camera_pose, projection)
                if points_2d is None:
                    continue
                    
                box2d_coords = self._calculate_2d_obb(points_2d)
                
                # Tính bbox theo format YOLO (x_center, y_center, width, height)
                x_min = max(0, min(p[0] for p in box2d_coords))
                y_min = max(0, min(p[1] for p in box2d_coords))
                x_max = min(self.image_size[0], max(p[0] for p in box2d_coords))
                y_max = min(self.image_size[1], max(p[1] for p in box2d_coords))
                
                # Normalize về khoảng [0, 1]
                x_center = (x_min + x_max) / (2 * self.image_size[0])
                y_center = (y_min + y_max) / (2 * self.image_size[1])
                width = (x_max - x_min) / self.image_size[0]
                height = (y_max - y_min) / self.image_size[1]
                
                # Ghi annotation theo format YOLO:
                # <class_id> <x_center> <y_center> <width> <height>
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        self.image_id += 1
    
    def save_metadata(self):
        """Lưu file data.yaml chứa thông tin về classes và paths"""
        yaml_content = {
            'path': self.output_dir,
            'train': 'images/train',  # path to train images
            'val': 'images/val',      # path to validation images
            'test': 'images/test',    # path to test images
            
            'nc': len(self.categories),  # number of classes
            'names': self.categories  # class names
        }
        
        # Convert to YAML format
        yaml_str = 'path: ' + yaml_content['path'] + '\n'
        yaml_str += 'train: ' + yaml_content['train'] + '\n'
        yaml_str += 'val: ' + yaml_content['val'] + '\n'
        yaml_str += 'test: ' + yaml_content['test'] + '\n'
        yaml_str += 'nc: ' + str(yaml_content['nc']) + '\n'
        yaml_str += 'names: ' + str(yaml_content['names']) + '\n'
        
        # Save to data.yaml
        yaml_path = os.path.join(self.output_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_str)
        
        # Save class names to classes.txt
        classes_path = os.path.join(self.output_dir, 'classes.txt')
        with open(classes_path, 'w') as f:
            for category in self.categories:
                f.write(f"{category}\n")