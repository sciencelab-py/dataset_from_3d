import os
from src.scence_generator import SceneGenerator
from src.format.coco_json import COCODatasetGenerator
from src.format.yolo import YOLODatasetGenerator
import traceback
import pyrender

def main():
    # Define paths to 3D model files
    model_paths = {
        'blue_sample': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\am-5401_blue\\am-5401_blue.obj",
        'yellow_sample': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\am-5401_yellow\\am-5401_yellow.obj",
        'red_sample': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\am-5401_red\\am-5401_red.obj",
        'blue_specimen': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\blue_specimen\\blue_specimen.obj",
        'red_specimen': "D:\\Projects\\ScienceLab\\ftc25-test\\element\\red_specimen\\red_specimen.obj",
    }

    # Initialize generators
    scene_gen = SceneGenerator(model_paths, image_size=(800, 600))
    dataset_gen = YOLODatasetGenerator(output_dir='output_dataset', image_size=(800, 600))

    # Generate multiple scenes
    num_scenes = 3  # Adjust number of scenes as needed
    
    print(f"Generating {num_scenes} scenes...")
    for i in range(num_scenes):
        if i % 10 == 0:
            print(f"Processing scene {i}/{num_scenes}")
            
        # Generate a new scene
        scene_data = scene_gen.generate_scene(min_objects=2, max_objects=5)
        
        # Process the scene and generate dataset entries
        dataset_gen.process_scene(scene_data)

        # Optional: Visualize the scene (uncomment to view)
        # pyrender.Viewer(scene_data[0])

    # Save the dataset metadata
    dataset_gen.save_metadata()
    print("Dataset generation complete!")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('output_dataset', exist_ok=True)
    
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())