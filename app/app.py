"""
Inverse Cooking — Main Web Application (Phase 3)
================================================
Upload a food image -> Qwen Vision identifies dish -> Matcher checks chef DB (recipes.json).
If matched: displays verified chef recipe with restaurant info and Swiggy ordering option.
If unmatched: falls back to Qwen AI generated recipe.

Run:
    streamlit run app.py
"""

import streamlit as st
import os
import sys
import json
import re
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Add project root to path
from app.services.matcher_service import matcher_service

load_dotenv()

# =============================================================================
# QWEN VISION PROMPT & PARSER
# =============================================================================

QWEN_PROMPT = """Analyze this food image. Respond ONLY with valid JSON, no markdown fences, no extra text:
{
  "dish_name": "<most likely dish name, generic e.g. 'Butter Chicken'>",
  "cuisine": "<cuisine type>",
  "visible_ingredients": ["<ingredient1>", "<ingredient2>"],
  "confidence": <0-100>
}"""

QWEN_RECIPE_GEN_PROMPT = """Generate a detailed cooking recipe for the dish identified in this food image.
Include: 1. Recipe Name, 2. Prep & Cook Time, 3. Ingredients with measurements, 4. Step-by-step Instructions."""


def parse_qwen_response(raw_response: str) -> dict:
    """Safely parse Qwen Vision JSON output, cleaning markdown code fences if present."""
    cleaned = raw_response.strip()
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


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to JPEG Base64 string."""
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def run_qwen_vision_analysis(image_base64: str) -> dict:
    """Query Qwen Vision API for dish identification JSON."""
    hf_token = os.getenv("HF_TOKEN")
    client = InferenceClient(api_key=hf_token)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": QWEN_PROMPT}
            ]
        }
    ]
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=messages,
        max_tokens=500
    )
    raw_content = response.choices[0].message.content
    return parse_qwen_response(raw_content)


def run_qwen_recipe_generation(image_base64: str) -> str:
    """Query Qwen Vision API to generate fallback recipe markdown text."""
    hf_token = os.getenv("HF_TOKEN")
    client = InferenceClient(api_key=hf_token)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": QWEN_RECIPE_GEN_PROMPT}
            ]
        }
    ]
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=messages,
        max_tokens=800
    )
    return response.choices[0].message.content


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Inverse Cooking AI",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main-title {
            font-size: 2.8rem;
            color: #E63946;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #94A3B8;
            margin-bottom: 1.8rem;
        }
        .chef-badge {
            background-color: #DEF7EC;
            color: #03543F;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 10px;
        }
        .ai-badge {
            background-color: #E1EFFE;
            color: #1E429F;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 10px;
        }
        .card-box {
            background-color: #1E293B;
            color: #F8FAFC;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .card-box h4 {
            color: #F8FAFC !important;
            margin-top: 0;
            margin-bottom: 8px;
        }
        .card-box p {
            color: #CBD5E1 !important;
            margin-bottom: 4px;
        }
        .card-box b {
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🍽️ Inverse Cooking AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a dish photo → AI identifies the dish → Chef Verified recipe & ordering options</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("Uses Qwen2.5-VL-7B for vision recognition and matches against local Chef Portal verified recipes (`recipes.json`).")
        st.markdown("---")
        st.markdown("**Status**: Ready for Phase 4 (Swiggy MCP)")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("📤 1. Upload Dish Image")
        uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Dish", use_container_width=True)

            if st.button("🔮 Recognize Dish & Find Recipe", type="primary", use_container_width=True):
                with st.spinner("🧠 Qwen Vision is analyzing the image..."):
                    try:
                        img_b64 = image_to_base64(image)
                        analysis = run_qwen_vision_analysis(img_b64)
                        st.session_state.analysis = analysis
                        st.session_state.img_b64 = img_b64

                        # Wire Matcher: find matching chef recipe
                        dish_name = analysis.get("dish_name", "")
                        matched_chef_recipe = matcher_service.find_matching_recipe(dish_name)
                        st.session_state.matched_recipe = matched_chef_recipe

                        # If no chef recipe matched, generate AI recipe as fallback
                        if not matched_chef_recipe:
                            with st.spinner("🤖 No chef recipe matched. Generating AI fallback recipe..."):
                                ai_recipe = run_qwen_recipe_generation(img_b64)
                                st.session_state.ai_recipe = ai_recipe
                        else:
                            st.session_state.ai_recipe = None

                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
        else:
            st.info("👆 Upload an image of any food item (e.g. Butter Chicken, Paneer Butter Masala, Pizza) to test.")

    with col_result:
        st.subheader("📋 2. Recipe & Dish Identification")

        if "analysis" in st.session_state and st.session_state.analysis:
            analysis = st.session_state.analysis
            matched = st.session_state.get("matched_recipe")

            # Dish identification summary card
            st.markdown(f"""
            <div class="card-box">
                <h4>🔍 Identified: <b>{analysis.get('dish_name', 'Unknown')}</b></h4>
                <p><b>Cuisine:</b> {analysis.get('cuisine', 'N/A')} | <b>Confidence:</b> {analysis.get('confidence', 0)}%</p>
                <p><b>Visible Ingredients:</b> {', '.join(analysis.get('visible_ingredients', [])) or 'None detected'}</p>
            </div>
            """, unsafe_allow_html=True)

            # Match result display
            if matched:
                st.markdown('<div class="chef-badge">👨‍🍳 Verified Chef Recipe Found!</div>', unsafe_allow_html=True)
                st.success(f"Matched with chef recipe from **{matched['restaurant']}** by **{matched['chef']}**!")

                with st.expander("📖 View Verified Chef Recipe", expanded=True):
                    st.markdown(f"### {matched['dish_name']}")
                    st.write(f"**Chef:** {matched['chef']} | **Restaurant:** {matched['restaurant']}")
                    st.write(f"**Cuisine:** {matched.get('cuisine', '—')} | **Price:** ₹{matched['base_price']}")
                    if matched.get("description"):
                        st.caption(f"_{matched['description']}_")

                    st.markdown("#### 🥗 Verified Ingredients")
                    for ing in matched["ingredients"]:
                        st.write(f"- {ing}")

                    st.markdown("#### 👩‍🍳 Cooking Steps")
                    for idx, step in enumerate(matched["steps"], 1):
                        st.write(f"{idx}. {step}")

                    if matched.get("premium_add_ons"):
                        st.markdown("#### 🎁 Available Add-ons")
                        for addon in matched["premium_add_ons"]:
                            st.write(f"- {addon['name']} (+₹{addon['price']})")

            else:
                st.markdown('<div class="ai-badge">🤖 AI Generated Recipe (Fallback)</div>', unsafe_allow_html=True)
                st.warning("No verified chef recipe found in Chef Portal database. Showing AI generated recipe:")

                ai_text = st.session_state.get("ai_recipe")
                if ai_text:
                    st.markdown(ai_text)
                else:
                    st.info("Recipe generation in progress...")
        else:
            st.info("🍽️ Upload a dish photo and click 'Recognize Dish & Find Recipe' to see results.")


if __name__ == "__main__":
    main()
