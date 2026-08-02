"""
Recipe Persistence Repository Pattern
Handles read and write operations for recipes.json storage
"""
import json
import os
from typing import Dict, Any
from app.core.config import RECIPES_FILE, DATA_DIR

class RecipeRepository:
    """Repository class for persisting and querying chef-uploaded recipes."""
    
    def __init__(self, storage_path=RECIPES_FILE):
        self.storage_path = storage_path
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Ensure parent directories and storage file exist."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def load_recipes(self) -> Dict[str, Any]:
        """Load all stored recipes from JSON storage."""
        if not os.path.exists(self.storage_path):
            return {}
        with open(self.storage_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def save_recipes(self, recipes: Dict[str, Any]) -> None:
        """Save recipe dictionary to JSON storage."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(recipes, f, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_recipe_key(dish_name: str, restaurant: str) -> str:
        """Generate unique compound key for a recipe entry."""
        return f"{restaurant.strip().lower()}::{dish_name.strip().lower()}"

# Singleton repository instance
recipe_repo = RecipeRepository()
