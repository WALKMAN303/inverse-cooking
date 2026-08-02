"""
Recipe Schemas & DTOs
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class RecipeRequest(BaseModel):
    image_base64: str = Field(..., description="Food dish image encoded as base64 string")


class RecipeResponse(BaseModel):
    recipe: str = Field(..., description="Generated markdown cooking recipe text")


class FoodAnalysisResponse(BaseModel):
    dish_name: str = Field(..., description="Recognized dish title")
    cuisine: str = Field(..., description="Origin cuisine category")
    visible_ingredients: List[str] = Field(default_factory=list, description="Detected visual ingredients")
    confidence: float = Field(..., description="Recognition confidence percentage (0 to 100)")
