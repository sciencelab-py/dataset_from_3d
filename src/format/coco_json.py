from ..dataset_generator import DatasetGenerator
import datetime
import json
import numpy as np
import os
from PIL import Image

class COCODatasetGenerator(DatasetGenerator):
    def __init__(self, output_dir, image_size=(800, 600)):
        super().__init__(output_dir, image_size)
        self.annotations = []
        self.images = []
        self.categories = list()
        self.image_id = 0
        self.annotation_id = 0
        
    def process_scene(self, scene_data, visualize=False):
        """Process scene và tạo COCO annotations"""
        scene, objects_metadata, camera_pose, projection = scene_data
        
        # Render scene
        color, depth = self.render_scene(scene)

        # Process ảnh
        image = color.astype(float) / 255.0
        image = self.add_noise_and_augmentation(image)
        
        # Lưu ảnh
        image_filename = f'image_{self.image_id:06d}.png'
        image_path = os.path.join(self.output_dir, image_filename)
        
        Image.fromarray((image * 255).astype(np.uint8)).save(image_path)

        if visualize:
            # Tạo ảnh với bounding boxes để visualize
            vis_image = self.visualize_bounding_boxes(color, objects_metadata, camera_pose, projection)
            vis_filename = f'vis_{self.image_id:06d}.png'
            vis_path = os.path.join(self.images_dir, vis_filename)
            Image.fromarray(vis_image).save(vis_path)
        
        # Tạo image info
        image_info = {
            'id': self.image_id,
            'file_name': image_filename,
            'width': self.image_size[0],
            'height': self.image_size[1],
            'date_captured': datetime.datetime.now().isoformat()
        }
        self.images.append(image_info)
        
        # Tạo annotations cho mỗi object
        for obj in objects_metadata:
            if obj['object_type'] not in self.categories:
                self.categories.append(obj['object_type'])
            category_id = self.categories.index(obj['object_type'])
            
            # Tính bbox từ projected points

            obb = obj['obb']
            # Tạo các điểm của box trong local space
            corners = self._get_box_corners(obb['extents'], obb['transform'])

            # Project sang 2D
            points_2d = self._project_points(corners, camera_pose, projection)
            if points_2d is None:
                continue
                
            box2d_coords = self._calculate_2d_obb(points_2d)

            x_min = min(p[0] for p in box2d_coords)
            y_min = min(p[1] for p in box2d_coords)
            x_max = max(p[0] for p in box2d_coords)
            y_max = max(p[1] for p in box2d_coords)

            # Tạo annotation
            bbox = [
                ((x_min + x_max) / 2) / self.image_size[0],  # x center
                ((y_min + y_max) / 2) / self.image_size[1],  # y center
                (x_max - x_min) / self.image_size[0],        # width
                (y_max - y_min) / self.image_size[1]         # height
            ]

            # Create annotation
            self.annotations.append({
                'id': self.annotation_id,
                'image_id': self.image_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })
            self.annotation_id += 1

        self.image_id += 1

    def save_metadata(self):
        """Save COCO format dataset to JSON file"""
        # Convert categories to list format
        categories = [
            {'id': i, 'name': category}
            for i, category in self.categories
        ]
        
        # Create COCO format dataset
        dataset = {
            'info': {
                'description': 'Synthetic Dataset',
                'version': '1.0',
                'year': datetime.datetime.now().year,
                'date_created': datetime.datetime.now().isoformat()
            },
            'images': self.images,
            'annotations': self.annotations,
            'categories': categories
        }
        
        # Save to JSON file
        output_path = os.path.join(self.output_dir, 'annotations.json')
        with open(output_path, 'w') as f:
            json.dump(dataset, f)