"""
Recipe Matcher Service — Connects Qwen Vision dish recognition with Chef Portal verified recipes.
Uses SequenceMatcher fuzzy string matching against consented recipes stored in recipe_repo.
"""
from difflib import SequenceMatcher
from typing import Dict, Any, Optional
from app.storage.recipe_repo import recipe_repo, RecipeRepository

MATCH_THRESHOLD = 0.6  # Score threshold between 0.0 and 1.0


class MatcherService:
    """Service to evaluate string similarity and locate verified chef recipes."""

    def __init__(self, repository: RecipeRepository = recipe_repo, threshold: float = MATCH_THRESHOLD):
        self.repository = repository
        self.threshold = threshold

    @staticmethod
    def calculate_similarity(a: str, b: str) -> float:
        """Calculate normalized string similarity ratio."""
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

    def find_matching_recipe(self, identified_dish_name: str, restaurant_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Locate best matching recipe in repository with consent=True.
        Returns recipe dict if similarity score >= threshold, else None.
        """
        recipes = self.repository.load_recipes()
        if not recipes:
            return None

        best_match = None
        best_score = 0.0

        for _, recipe in recipes.items():
            if not recipe.get("consent", False):
                continue  # Never surface unconsented recipes

            score = self.calculate_similarity(identified_dish_name, recipe.get("dish_name", ""))

            # Optional boost for matching restaurant hint
            if restaurant_hint and self.calculate_similarity(restaurant_hint, recipe.get("restaurant", "")) > 0.6:
                score += 0.15

            if score > best_score:
                best_score = score
                best_match = recipe

        if best_score >= self.threshold:
            return best_match

        return None

# Singleton Service Instance
matcher_service = MatcherService()
