import json
import os

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Basic configuration attributes
        self.debug_mode = False
        self.app_name = "ScienceLab"
        self.version = "1.0.0"
        
        # Load JSON configuration
        config_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(config_path, 'r') as f:
                self.params = json.load(f)
        except FileNotFoundError:
            print(f"Warning: config.json not found at {config_path}")
            self.params = dict()
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in config.json")
            self.params = dict()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance
    
    def __getattr__(self, name):
        return self.params.get(name, None)