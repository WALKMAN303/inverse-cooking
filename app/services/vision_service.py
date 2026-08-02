"""
Vision AI Service — Interfacing with Hugging Face Qwen2.5-VL for dish recognition and recipe generation.
"""
import json
import re
from typing import Dict, Any
from huggingface_hub import InferenceClient
from app.core.config import HF_TOKEN, VISION_MODEL_ID

# Prompts
FOOD_JSON_ANALYSIS_PROMPT = """Analyze this food image. Respond ONLY with valid JSON, no markdown fences, no extra text:
{
  "dish_name": "<most likely dish name, generic e.g. 'Butter Chicken'>",
  "cuisine": "<cuisine type>",
  "visible_ingredients": ["<ingredient1>", "<ingredient2>"],
  "confidence": <0-100>
}"""

RECIPE_GENERATION_PROMPT = """Analyze this food image and generate a detailed recipe.
Include: 1. Recipe Name, 2. Prep & Cook Time, 3. Ingredients with measurements, 4. Step-by-step Instructions."""


class VisionService:
    """Service wrapper for Multimodal Vision-Language Model requests."""
    
    def __init__(self, api_key: str = HF_TOKEN, model_id: str = VISION_MODEL_ID):
        self.api_key = api_key
        self.model_id = model_id
        self._client = InferenceClient(api_key=self.api_key) if self.api_key else None

    def query_vision_model(self, image_base64: str, prompt: str) -> str:
        """Send image payload and prompt to the Qwen Vision inference model."""
        if not self.api_key:
            print("WARNING: HF_TOKEN is not set. Inference API call might fail or hit rate limits.")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=800
        )
        return response.choices[0].message.content

    def parse_qwen_json(self, raw_text: str) -> Dict[str, Any]:
        """Safely parse Qwen Vision JSON response, removing potential markdown code fences."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {
                "dish_name": "Unknown Dish",
                "cuisine": "Unknown",
                "visible_ingredients": [],
                "confidence": 50
            }

    async def analyze_food_image(self, image_base64: str) -> Dict[str, Any]:
        """Recognize food image and return structured JSON attributes."""
        raw_response = self.query_vision_model(image_base64, FOOD_JSON_ANALYSIS_PROMPT)
        return self.parse_qwen_json(raw_response)

    async def generate_recipe(self, image_base64: str) -> str:
        """Generate detailed step-by-step cooking recipe markdown."""
        return self.query_vision_model(image_base64, RECIPE_GENERATION_PROMPT)

# Singleton Service Instance
vision_service = VisionService()
