from .format.yolo import YOLODatasetGenerator
from .format.coco_json import COCODatasetGenerator

class DatasetGeneratorFactory:
    """Factory class to create appropriate dataset generator based on format"""
    
    @staticmethod
    def create_generator(format_type, output_dir, image_size=(800, 600)):
        """
        Create and return a dataset generator based on the specified format
        
        Args:
            format_type (str): The type of dataset format
            output_dir (str): Directory to save the generated dataset
            image_size (tuple): Tuple of (width, height) for generated images
            
        Returns:
            DatasetGenerator: An instance of the appropriate dataset generator
            
        Raises:
            ValueError: If format_type is not supported
        """
        format_type = format_type.lower()
        
        if format_type == 'yolo':
            return YOLODatasetGenerator(output_dir, image_size)
        elif format_type == 'coco':
            return COCODatasetGenerator(output_dir, image_size)
        else:
            raise ValueError(f"Unsupported dataset format: {format_type}")