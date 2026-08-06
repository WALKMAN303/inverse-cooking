# Inverse Cooking: Vision-Language Dish Recognition & Swiggy MCP Ordering

> An industrial culinary intelligence application that identifies dishes from food images using Vision-Language AI (Qwen2.5-VL-7B), matches verified chef recipes, and automates food ordering via the Swiggy Model Context Protocol (MCP).

---

## Overview

**Inverse Cooking** bridges visual food recognition with automated culinary ordering:
1. **Vision-Language Analysis**: Upload any food photo. The system uses **Qwen2.5-VL-7B-Instruct** (via Hugging Face Inference API) to recognize the dish, cuisine type, confidence level, and visible ingredients.
2. **Verified Chef Recipe Matcher**: Uses fuzzy string similarity (`SequenceMatcher`) to match the recognized dish against verified chef recipes in `data/recipes.json`.
3. **AI Fallback Recipe**: If no verified chef recipe is found, the system generates step-by-step cooking instructions using Qwen AI.
4. **Swiggy MCP Integration**: Automates food ordering via Swiggy's 7-tool Model Context Protocol with OAuth 2.1 + PKCE authentication and intelligent menu add-on matching.
5. **Multipage Streamlit Web Application**: Unified UI for both consumers (Dish Recognition & Ordering) and chefs (Recipe Upload & Consent Management).

---

## System Architecture

```
                                  [ Upload Food Image ]
                                            │
                                            ▼
                             [ Qwen2.5-VL-7B Vision AI ]
                                            │
                                  Structured JSON Output
                        (dish_name, cuisine, ingredients, confidence)
                                            │
                                            ▼
                                 [ Matcher Engine ]
                         (SequenceMatcher against recipes.json)
                                     /             \
                       Match Found  /               \  No Match
                                   /                 \
                                  ▼                   ▼
                     [ Verified Chef Recipe ]     [ AI Generated Recipe ]
                                  │
                                  ▼
                    [ Swiggy MCP Ordering Flow ]
             OAuth 2.1 PKCE → Address → Restaurant → Dish Menu 
             → Premium Add-on Match → Cart Cap Check → COD Order
```

---

## Key Features

- **Multimodal AI Vision**: Powered by `Qwen/Qwen2.5-VL-7B-Instruct` for zero-shot dish recognition and structured JSON extraction.
- **Fuzzy Recipe Matching**: Intelligent matcher engine (`matcher_service.py`) pairs identified dishes with consented chef recipes based on configurable similarity thresholds.
- **Swiggy MCP 7-Tool Pipeline**:
  - `get_addresses`: Resolves user delivery address.
  - `search_restaurants`: Locates target restaurant availability.
  - `get_restaurant_menu` & `search_menu`: Fetches live menu and selects dish item.
  - `find_best_addon_match`: Fuzzy matches chef-suggested premium ingredients to real Swiggy menu add-ons.
  - `update_food_cart` & `get_food_cart`: Builds cart and verifies spending caps (₹1,000 max).
  - `place_food_order`: Submits Cash on Delivery (COD) order.
- **OAuth 2.1 + PKCE Auth**: Secure authentication with Swiggy MCP servers using local RFC 8414 metadata discovery and callback handlers.
- **Chef Recipe Portal**: Dedicated page for chefs to upload verified recipes, set base pricing, define premium add-on costs, and toggle ordering consent.
- **Modular Architecture**: Clean separation of core config, Pydantic DTO schemas, repository persistence patterns, services, REST endpoints, and UI views.

---

## Project Structure

```
inverse-cooking/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── recipes.py           # FastAPI REST API endpoints
│   ├── core/
│   │   └── config.py                # Environment config, paths, model IDs
│   ├── schemas/
│   │   └── recipe.py                # Pydantic request & response models
│   ├── services/
│   │   ├── matcher_service.py       # SequenceMatcher chef recipe lookup
│   │   ├── swiggy_auth.py           # OAuth 2.1 + PKCE login flow
│   │   ├── swiggy_service.py        # 7-tool Swiggy MCP ordering service
│   │   └── vision_service.py        # Qwen2.5-VL Hugging Face Inference client
│   ├── storage/
│   │   └── recipe_repo.py           # Recipe JSON persistence repository
│   └── main.py                      # FastAPI server entry point
├── ui/
│   ├── main_app.py                  # Streamlit entry point (Home: Vision & Ordering)
│   └── pages/
│       └── 1_Chef_Portal.py         # Streamlit page (Chef Portal: Upload & Manage)
├── data/
│   └── recipes.json                 # Verified chef recipes storage
├── .env                             # Environment variables (HF_TOKEN) - gitignored
├── .gitignore                       # Ignored build & environment files
├── requirements.txt                 # Python dependencies
└── readme.md                        # Documentation
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Hugging Face API Token (with access to `Qwen/Qwen2.5-VL-7B-Instruct`)

### 2. Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/WALKMAN303/inverse-cooking.git
   cd inverse-cooking
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory:
   ```env
   HF_TOKEN=your_huggingface_access_token_here
   ```

---

## Running the Application

### Streamlit Multipage Web App (Recommended)

Run the unified Streamlit application:
```bash
streamlit run ui/main_app.py
```

Access the app at `http://localhost:8501`. Use the sidebar to switch between:
- **Home**: Upload food images, view AI dish recognition, and place Swiggy orders.
- **Chef Portal**: Upload new chef-verified recipes, set premium add-ons, and manage customer visibility consent.

---

### FastAPI Server (Optional REST Endpoints)

Start the REST API server:
```bash
uvicorn app.main:app --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

#### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health check |
| `POST` | `/api/v1/recipes/analyze` | Returns dish name, cuisine, ingredients, and confidence from base64 image |
| `POST` | `/api/v1/recipes/predict` | Generates detailed markdown cooking recipe from base64 image |

---

## Requirements

```text
streamlit>=1.32.0
pillow>=9.0.0
huggingface_hub>=0.24.0
python-dotenv>=1.0.0
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.6.0
httpx>=0.27.0
mcp>=1.0.0
```

---

## Security & Persistence Notes

- `.env` contains your `HF_TOKEN` and is strictly gitignored. Never commit `.env`.
- `.swiggy_token.json` stores local OAuth tokens and is gitignored.
- `venv/` should never be distributed or committed to version control.

---

## License

This project is licensed under the [MIT License](LICENSE).

