"""
V1 Recipe API Endpoints
"""
from fastapi import APIRouter, HTTPException, status
from app.schemas.recipe import RecipeRequest, RecipeResponse, FoodAnalysisResponse
from app.services.vision_service import vision_service

router = APIRouter(prefix="/recipes", tags=["Recipes & Vision AI"])

@router.post("/predict", response_model=RecipeResponse, status_code=status.HTTP_200_OK)
async def predict_recipe(payload: RecipeRequest):
    """Generate detailed cooking instructions from base64 image."""
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="Base64 image string is required.")
        
    recipe_markdown = await vision_service.generate_recipe(payload.image_base64)
    return RecipeResponse(recipe=recipe_markdown)

@router.post("/analyze", response_model=FoodAnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_recipe(payload: RecipeRequest):
    """Analyze image to identify dish name, cuisine, ingredients, and confidence."""
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="Base64 image string is required.")
        
    try:
        analysis_data = await vision_service.analyze_food_image(payload.image_base64)
        return FoodAnalysisResponse(**analysis_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
