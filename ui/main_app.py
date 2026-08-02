"""
Inverse Cooking — Consumer Web Application
===========================================
Upload a food image -> Qwen Vision identifies dish -> Matcher checks chef DB (recipes.json).
If matched: displays verified chef recipe with restaurant info and Swiggy ordering options.
If unmatched: falls back to Qwen AI generated recipe.

Run:
    streamlit run ui/main_app.py
"""

import streamlit as st
import os
import sys
import base64
from io import BytesIO
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.services.vision_service import vision_service, FOOD_JSON_ANALYSIS_PROMPT
from app.services.matcher_service import matcher_service
from app.services.swiggy_service import place_customized_order_sync, SwiggyOrderError


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to JPEG Base64 string."""
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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
            color: #4A5568;
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
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🍽️ Inverse Cooking AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a dish photo → AI identifies dish → Chef Verified recipe & ordering options</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("Uses Qwen2.5-VL-7B for vision recognition and matches against local Chef Portal verified recipes (`data/recipes.json`).")
        st.markdown("---")
        st.markdown("**Status**: Industrial Modular Architecture Ready")

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
                        
                        # 1. Vision AI Analysis
                        raw_analysis = vision_service.query_vision_model(img_b64, FOOD_JSON_ANALYSIS_PROMPT)
                        analysis = vision_service.parse_qwen_json(raw_analysis)
                        
                        st.session_state.analysis = analysis
                        st.session_state.img_b64 = img_b64

                        # 2. Matcher Engine Lookup
                        dish_name = analysis.get("dish_name", "")
                        matched_chef_recipe = matcher_service.find_matching_recipe(dish_name)
                        st.session_state.matched_recipe = matched_chef_recipe

                        # 3. Fallback AI Generation if unmatched
                        if not matched_chef_recipe:
                            with st.spinner("🤖 No chef recipe matched. Generating AI fallback recipe..."):
                                ai_recipe = vision_service.query_vision_model(img_b64, "Generate a detailed cooking recipe.")
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

            # Dish identification card
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
                    for ing in matched.get("ingredients", []):
                        st.write(f"- {ing}")

                    st.markdown("#### 👩‍🍳 Cooking Steps")
                    for idx, step in enumerate(matched.get("steps", []), 1):
                        st.write(f"{idx}. {step}")

                    selected_addons = []
                    if matched.get("premium_add_ons"):
                        st.markdown("#### 🎁 Available Add-ons")
                        addon_names = [a["name"] for a in matched["premium_add_ons"]]
                        selected_addons = st.multiselect(
                            "Customize your order (matched against this restaurant's real Swiggy menu):",
                            options=addon_names,
                            key=f"addons_{matched['dish_name']}"
                        )

                    st.markdown("---")
                    if st.button("🛵 Order via Swiggy", type="primary", use_container_width=True):
                        with st.spinner("Placing your order on Swiggy..."):
                            try:
                                order = place_customized_order_sync(
                                    dish_name=matched["dish_name"],
                                    restaurant_name=matched["restaurant"],
                                    premium_ingredients=selected_addons,
                                )
                                st.success(f"✅ Order placed! ID: {order['order_id']}")
                                st.write(f"**Restaurant:** {order['restaurant']}")
                                st.write(f"**Dish:** {order['dish']}")
                                if order["matched_add_ons"]:
                                    st.write(f"**Add-ons applied:** {', '.join(order['matched_add_ons'])}")
                                if order["unmatched_ingredients"]:
                                    st.warning(
                                        f"Couldn't find a matching Swiggy add-on for: "
                                        f"{', '.join(order['unmatched_ingredients'])}. Ordered without these."
                                    )
                                st.write(f"**Total (COD):** ₹{order['total']}")
                            except SwiggyOrderError as e:
                                st.error(f"❌ {str(e)}")
                            except Exception as e:
                                st.error(f"❌ Unexpected error: {str(e)}")

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
