import numpy as np

class CameraPositioner:
    """Helper class for camera positioning calculations"""
    
    @staticmethod
    def calculate_camera_position(center, distance, phi, theta):
        """Calculate camera position from spherical coordinates"""
        return center + distance * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
    
    @staticmethod
    def calculate_camera_rotation(camera_position, look_at_point):
        """Calculate camera rotation matrix to look at a point"""
        direction = look_at_point - camera_position
        direction = direction / np.linalg.norm(direction)
        
        # Calculate up vector
        up = np.array([0, 0, 1])
        if np.abs(np.dot(up, direction)) > 0.9:
            up = np.array([0, 1, 0])
        
        # Calculate right and adjusted up vectors    
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix
        rotation = np.eye(3)
        rotation[:, 0] = right
        rotation[:, 1] = up
        rotation[:, 2] = -direction
        
        return rotation
    
    @staticmethod
    def create_camera_pose(position, rotation):
        """Create camera pose matrix from position and rotation"""
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = position
        camera_pose[:3, :3] = rotation
        return camera_pose