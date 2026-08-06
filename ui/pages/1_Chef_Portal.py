"""
Chef Recipe Portal
==================
Lets chefs upload verified recipes, tag them to their restaurant,
and manage consent permissions for customer visibility & ordering.
"""

import streamlit as st
import os
import sys
from datetime import datetime

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.storage.recipe_repo import recipe_repo, RecipeRepository


def main():
    st.set_page_config(page_title="Chef Recipe Portal", layout="wide")
    st.title("Chef Recipe Portal")
    st.caption("Upload your dish's verified recipe. You control whether customers can order it.")

    recipes = recipe_repo.load_recipes()

    tab_upload, tab_manage = st.tabs(["Upload new recipe", "Manage my recipes"])

    # ── UPLOAD TAB ──
    with tab_upload:
        with st.form("recipe_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                chef_name = st.text_input("Chef name *")
                restaurant = st.text_input("Restaurant name *")
                dish_name = st.text_input("Dish name *", help="Use common dish name customers recognize, e.g. 'Butter Chicken'")
            with col2:
                cuisine = st.text_input("Cuisine type")
                base_price = st.number_input("Base price (₹) *", min_value=0, step=10)
                description = st.text_area("One-line description", height=100)

            st.markdown("**Ingredients** (one per line, with quantity)")
            ingredients_raw = st.text_area(
                "Ingredients",
                placeholder="500g chicken (boneless)\n200ml heavy cream\n2 tbsp butter",
                height=120,
                label_visibility="collapsed"
            )

            st.markdown("**Steps** (one per line, in order)")
            steps_raw = st.text_area(
                "Steps",
                placeholder="Marinate chicken for 2 hours\nGrill until charred\nSimmer in sauce for 15 min",
                height=120,
                label_visibility="collapsed"
            )

            st.markdown("**Premium add-ons** (optional — one per line, format: `Name +price`)")
            addons_raw = st.text_area(
                "Add-ons",
                placeholder="Extra cream +30\nTruffle oil drizzle +80",
                height=80,
                label_visibility="collapsed"
            )

            st.markdown("---")
            consent = st.checkbox(
                "I consent to this recipe being shown to customers and ordered via Swiggy with a premium fee",
                value=False
            )

            submitted = st.form_submit_button("Save recipe", type="primary", width="stretch")

            if submitted:
                if not (chef_name and restaurant and dish_name and base_price):
                    st.error("Please fill in all required fields (marked *).")
                else:
                    ingredients = [i.strip() for i in ingredients_raw.splitlines() if i.strip()]
                    steps = [s.strip() for s in steps_raw.splitlines() if s.strip()]

                    addons = []
                    for line in addons_raw.splitlines():
                        line = line.strip()
                        if "+" in line:
                            name, price = line.rsplit("+", 1)
                            try:
                                addons.append({"name": name.strip(), "price": int(price.strip())})
                            except ValueError:
                                pass

                    recipe_id = RecipeRepository.generate_recipe_key(dish_name, restaurant)
                    recipes[recipe_id] = {
                        "chef": chef_name,
                        "restaurant": restaurant,
                        "dish_name": dish_name,
                        "cuisine": cuisine,
                        "description": description,
                        "base_price": base_price,
                        "ingredients": ingredients,
                        "steps": steps,
                        "premium_add_ons": addons,
                        "consent": consent,
                        "created_at": datetime.now().isoformat(),
                    }
                    recipe_repo.save_recipes(recipes)
                    st.success(f"Recipe for '{dish_name}' saved! Consent: {'Yes' if consent else 'No — hidden from customers'}")

    # ── MANAGE TAB ──
    with tab_manage:
        if not recipes:
            st.info("No recipes uploaded yet.")
        else:
            for rid, r in recipes.items():
                with st.expander(f"{r['dish_name']} — {r['restaurant']} ({'consented' if r['consent'] else 'not visible'})"):
                    st.write(f"**Chef:** {r['chef']}  |  **Cuisine:** {r.get('cuisine', '—')}  |  **Base price:** ₹{r['base_price']}")
                    st.write(f"**Ingredients:** {', '.join(r['ingredients'])}")
                    st.write("**Steps:**")
                    for i, step in enumerate(r["steps"], 1):
                        st.write(f"{i}. {step}")
                    if r.get("premium_add_ons"):
                        addon_text = ", ".join(f"{a['name']} (+₹{a['price']})" for a in r["premium_add_ons"])
                        st.write(f"**Add-ons:** {addon_text}")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        new_consent = st.checkbox("Consent to Swiggy ordering", value=r["consent"], key=f"consent_{rid}")
                        if new_consent != r["consent"]:
                            recipes[rid]["consent"] = new_consent
                            recipe_repo.save_recipes(recipes)
                            st.rerun()
                    with col_b:
                        if st.button("Delete recipe", key=f"del_{rid}"):
                            del recipes[rid]
                            recipe_repo.save_recipes(recipes)
                            st.rerun()


main()
