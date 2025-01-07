from ..dataset_generator import DatasetGenerator
import os
import numpy as np
from PIL import Image

class YOLOOBBDatasetGenerator(DatasetGenerator):
    def __init__(self, output_dir, image_size=(800, 600)):
        super().__init__(output_dir, image_size)
        self.categories = list()
        self.image_id = 0
        
        # Create images and labels directories
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
    def process_scene(self, scene_data, visualize=False):
        """Process scene and create YOLO-OBB points-based annotations"""
        scene, objects_metadata, camera_pose, projection = scene_data
        
        # Render scene
        color, depth = self.render_scene(scene)
        
        # Process image
        image = color.astype(float) / 255.0
        image = self.add_noise_and_augmentation(image)
        
        # Save image
        image_filename = f'image_{self.image_id:06d}.png'
        label_filename = f'image_{self.image_id:06d}.txt'
        
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, label_filename)
        
        Image.fromarray((image * 255).astype(np.uint8)).save(image_path)

        if visualize:
            vis_image = self.visualize_bounding_boxes(color, objects_metadata, camera_pose, projection)
            vis_filename = f'vis_{self.image_id:06d}.png'
            vis_path = os.path.join(self.images_dir, vis_filename)
            Image.fromarray(vis_image).save(vis_path)
            
        # Create and save YOLO-OBB points format annotations
        with open(label_path, 'w') as f:
            for obj in objects_metadata:
                if obj['object_type'] not in self.categories:
                    self.categories.append(obj['object_type'])
                category_id = self.categories.index(obj['object_type'])
                
                # Get OBB vertices
                obb = obj['obb']
                corners = np.array(obb['vertices'])
                
                # Project to 2D
                points_2d = self._project_points(corners, camera_pose, projection)
                if points_2d is None:
                    continue
                    
                # Calculate OBB corners in normalized coordinates
                box2d_coords = self._calculate_2d_obb_corners(points_2d)
                
                # Normalize coordinates
                box2d_coords[:, 0] /= self.image_size[0]
                box2d_coords[:, 1] /= self.image_size[1]
                
                # Write YOLO-OBB points format:
                # <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
                coords_str = ' '.join([f'{x:.6f}' for x in box2d_coords.flatten()])
                f.write(f"{category_id} {coords_str}\n")
        
        self.image_id += 1
    
    def save_metadata(self):
        """Save data.yaml containing classes and paths"""
        yaml_content = {
            'path': self.output_dir,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.categories),
            'names': self.categories
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
        
        # Save class names
        classes_path = os.path.join(self.output_dir, 'classes.txt')
        with open(classes_path, 'w') as f:
            for category in self.categories:
                f.write(f"{category}\n")