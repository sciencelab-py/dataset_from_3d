import os
from src.scene_generator_factory import SceneGeneratorFactory
from src.dataset_generator_factory import DatasetGeneratorFactory
from src.config import Config
import traceback

def main(config: Config):   
    # Use models from config
    model_paths = config.models

    # Get image size from config
    image_size = (config.image_size['width'], config.image_size['height'])

    # Get output directory from config
    output_dir = config.output['output_dir']

    # Initialize generators
    factory = SceneGeneratorFactory()

    # Chọn generator dựa trên config
    # random_clutter, partially_occluded, multi_level
    scene_gen = factory.create_generator(config.scene_generation['type'], model_paths)

    # Tạo scene
    dataset_gen = DatasetGeneratorFactory.create_generator(
        config.output['format'], output_dir, image_size=image_size
    )

    # Generate multiple scenes
    num_scenes = config.output['quantity']
    
    print(f"Generating {num_scenes} scenes...")
    for i in range(num_scenes):
        if i % 10 == 0:
            print(f"Processing scene {i}/{num_scenes}")
            
        # Generate a new scene using config parameters
        scene_data = scene_gen.generate_scene(
            min_objects=config.scene_generation['min_objects'],
            max_objects=config.scene_generation['max_objects']
        )
        
        # Process the scene and generate dataset entries
        dataset_gen.process_scene(scene_data)

    # Save the dataset metadata
    dataset_gen.save_metadata()
    print("Dataset generation complete!")

if __name__ == "__main__":
    config = Config.get_instance()
    # Create output directory if it doesn't exist
    os.makedirs(config.output['output_dir'], exist_ok=True)
    
    try:
        main(config)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())