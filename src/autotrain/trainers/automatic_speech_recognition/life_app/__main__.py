import os
import json
import logging
from typing import Dict, Any

from autotrain import logger
from .trainer import LifeAppASRTrainer

def main():
    """Main entry point for LiFE App ASR training."""
    try:
        # Load training config
        config_path = os.path.join(os.getcwd(), "training_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Training config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Initialize trainer
        trainer = LifeAppASRTrainer(config)
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Error during LiFE App ASR training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 