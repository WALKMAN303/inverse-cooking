"""
Application Configuration Module
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directory and persistence files
DATA_DIR = BASE_DIR / "data"
RECIPES_FILE = DATA_DIR / "recipes.json"

# HF / Vision Model Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "Qwen/Qwen2.5-VL-72B-Instruct")

# App Metadata
APP_TITLE = "Inverse Cooking API"
APP_DESCRIPTION = "Industrial Culinary Intelligence and Vision-Language Recipe API"
APP_VERSION = "1.0.0"
