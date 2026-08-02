"""
Swiggy MCP Client Service
====================================
Wraps the canonical 7-tool Swiggy Food ordering flow:
  get_addresses -> search_restaurants -> get_restaurant_menu ->
  search_menu -> update_food_cart -> get_food_cart -> place_food_order
  -> track_food_order

Also implements the "premium customization" matching described in
the project: a chef's recipe may suggest a premium ingredient
(e.g. "truffle oil drizzle"). Since Swiggy's cart only accepts real
menu items/add-ons, we search that specific restaurant's live menu
for the closest matching add-on instead of inventing a price.

v1 constraints from Swiggy docs:
  - COD only, no online payment
  - Hard ₹1000 cart cap
  - place_food_order is NOT idempotent — never blind-retry it
"""

import asyncio
from difflib import SequenceMatcher

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.services.swiggy_auth import get_valid_token

SWIGGY_FOOD_SERVER = "https://mcp.swiggy.com/food"
CART_CAP_INR = 1000


class SwiggyOrderError(Exception):
    pass


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


async def find_best_addon_match(session: ClientSession, restaurant_id: str,
                                  desired_addon_name: str, threshold: float = 0.45) -> dict | None:
    """
    Given a chef-suggested premium ingredient (e.g. 'truffle oil'),
    searches this restaurant's real menu add-ons for the closest match.
    Returns the add-on dict (with real Swiggy addOnId + price) or None.
    """
    menu_result = await session.call_tool("get_restaurant_menu", {"restaurantId": restaurant_id})
    menu = menu_result.data

    best_addon, best_score = None, 0.0
    for item in menu.get("items", []):
        for addon in item.get("addOns", []):
            score = _similarity(desired_addon_name, addon.get("name", ""))
            if score > best_score:
                best_score, best_addon = score, addon

    if best_score >= threshold:
        return best_addon
    return None


async def place_customized_order(dish_name: str, restaurant_name: str,
                                    premium_ingredients: list[str] | None = None) -> dict:
    """
    Full end-to-end order flow for one dish, with best-effort premium
    add-on matching. Returns a summary dict for display in the UI.

    Raises SwiggyOrderError with a user-facing message on any failure
    (closed restaurant, cart over cap, etc.) rather than a raw traceback.
    """
    premium_ingredients = premium_ingredients or []
    token = get_valid_token()
    headers = {"Authorization": f"Bearer {token}"}

    async with streamablehttp_client(SWIGGY_FOOD_SERVER, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Step 1 — resolve address
            addresses = await session.call_tool("get_addresses", {})
            addr_list = addresses.data
            if not addr_list:
                raise SwiggyOrderError("No saved Swiggy address found. Add one in the Swiggy app first.")
            home = next((a for a in addr_list if a.get("label") == "Home"), addr_list[0])

            # Step 2 — find the restaurant
            search = await session.call_tool("search_restaurants", {
                "addressId": home["id"], "query": restaurant_name
            })
            candidates = [r for r in search.data.get("restaurants", [])
                          if r.get("availabilityStatus") == "OPEN"]
            if not candidates:
                raise SwiggyOrderError(f"'{restaurant_name}' isn't open on Swiggy right now.")
            restaurant = candidates[0]

            # Step 3 — browse menu, find the dish
            menu_search = await session.call_tool("search_menu", {
                "restaurantId": restaurant["id"], "query": dish_name
            })
            items = menu_search.data.get("items", [])
            if not items:
                raise SwiggyOrderError(f"Couldn't find '{dish_name}' on {restaurant_name}'s current Swiggy menu.")
            dish_item = items[0]

            # Match premium ingredients to real menu add-ons
            matched_addons = []
            unmatched = []
            for ingredient in premium_ingredients:
                addon = await find_best_addon_match(session, restaurant["id"], ingredient)
                if addon:
                    matched_addons.append(addon)
                else:
                    unmatched.append(ingredient)

            # Step 4 — build the cart
            cart_item = {"itemId": dish_item["id"], "quantity": 1}
            if matched_addons:
                cart_item["addOns"] = [a["id"] for a in matched_addons]

            await session.call_tool("update_food_cart", {
                "restaurantId": restaurant["id"],
                "items": [cart_item],
            })

            # Step 5 — confirm total against the ₹1000 cap
            cart = await session.call_tool("get_food_cart", {})
            total = cart.data.get("total", 0)
            if total > CART_CAP_INR:
                raise SwiggyOrderError(
                    f"Cart total ₹{total} exceeds the ₹{CART_CAP_INR} Builders Club cap. "
                    "Remove an add-on and try again."
                )

            # Step 6 — place the order (NOT idempotent — call once)
            order = await session.call_tool("place_food_order", {"paymentMethod": "COD"})

            return {
                "order_id": order.data.get("orderId"),
                "restaurant": restaurant["name"],
                "dish": dish_item["name"],
                "matched_add_ons": [a["name"] for a in matched_addons],
                "unmatched_ingredients": unmatched,
                "total": total,
                "payment": "COD",
            }


def place_customized_order_sync(dish_name: str, restaurant_name: str,
                                   premium_ingredients: list[str] | None = None) -> dict:
    """Sync wrapper for use inside Streamlit (which isn't async-native)."""
    return asyncio.run(place_customized_order(dish_name, restaurant_name, premium_ingredients))
