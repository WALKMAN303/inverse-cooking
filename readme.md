# 🍽️ Inverse Cooking: Recipe Generation from Food Images
> A deep learning project that generates cooking recipes from food images using CNN and Transformer architecture.

---

## 📖 Overview

This project implements an **image-to-text generation model** that can look at a food image and generate step-by-step cooking instructions. Built with PyTorch, it combines a CNN encoder (ResNet18) for visual feature extraction and a Transformer decoder for sequential text generation.

### Key Features
- 🖼️ **Image Understanding**: Uses pre-trained ResNet18 to extract visual features
- 📝 **Recipe Generation**: Transformer decoder generates coherent cooking instructions
- 🎯 **End-to-End Training**: Single pipeline from image input to text output
- 📊 **13,463 Recipes**: Trained on diverse food dataset with real recipes
- ⚡ **GPU Optimized**: Efficient training with CUDA support

---

## 🎯 Problem Statement

**Challenge**: Given an image of a finished dish, automatically generate the cooking recipe.

**Why It Matters**:
- Helps home cooks recreate dishes they see
- Assists people with dietary restrictions in understanding ingredients
- Educational tool for culinary students
- Accessibility feature for recipe discovery

---

## 🗃️ Architecture

### Model Design

```
Input Image (128×128×3)
         ↓
    CNN Encoder (ResNet18)
         ↓
    Feature Vector (256)
         ↓
  Transformer Decoder (2 layers, 4 heads)
         ↓
    Recipe Text Output
```

### Components

1. **CNN Encoder**
   - Pre-trained ResNet18 (ImageNet weights)
   - Fine-tuned last layer
   - Output: 256-dimensional feature vector

2. **Transformer Decoder**
   - 2 decoder layers
   - 4 attention heads
   - Embedding dimension: 256
   - Vocabulary size: 8,215 words
   - Max sequence length: 512 tokens

3. **Training Strategy**
   - Teacher forcing during training
   - Auto-regressive generation during inference
   - Cross-entropy loss
   - AdamW optimizer with learning rate scheduling

---

## 📊 Dataset

- **Source**: [Food Ingredients and Recipe Dataset](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)
- **Total Recipes**: 13,463
- **Train/Val Split**: 80/20
- **Image Size**: 128×128 RGB
- **Data Augmentation**: Random crop, horizontal flip, color jitter

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
CUDA 11.0+ (for GPU support)
```

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/WALKMAN303/inverse-cooking.git
cd inverse-cooking
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up your environment variables**
```bash
# Create a .env file with your Hugging Face token
echo HF_TOKEN=your_huggingface_token_here > .env
```

4. **Run the web app**
```bash
streamlit run ui/main_app.py
```

Both the **Home** (dish recognition & Swiggy ordering) and **Chef Portal** (recipe management) are accessible via the sidebar navigation in a single running app.

---

## 📦 Model Files

> **Note**: Due to GitHub's file size limitations (100 MB max), the pre-trained model weights are hosted on Google Drive.

### Download Pre-trained Models

**Option 1: Automatic Download (Recommended)**
```bash
python download_model.py
```

**Option 2: Manual Download**

Download the following files and place them in the project root directory:

| File | Size | Description | Download Link |
|------|------|-------------|---------------|
| `best_model.pth` | 118 MB | Best checkpoint model | [Download](https://drive.google.com/file/d/1HfSb-zVBlxTf22YrheEVGbT7KjZHoe-C/view?usp=sharing) |
| `vocab.pkl` | <1 MB | Vocabulary object | [Download](https://drive.google.com/file/d/13S6BM-Fc5uPCO-Tn1VJxxUnx3U0zFlHD/view?usp=sharing) |

**After downloading, your project structure should look like:**
```
inverse-cooking/
├── best_model.pth                ✅
├── vocab.pkl                     ✅
├── inverse-cooking.py
├── download_model.py
├── app.py
├── requirements.txt
└── README.md
```

### Alternative: Train Your Own Models

If you prefer to train from scratch instead of using pre-trained weights, you can run the training script:
```bash
python inverse-cooking.py
```

Training takes approximately 1 hour on Tesla T4 GPU and use kaggle or colab for the T4 GPU.

---

## 💻 Usage

### Quick Inference

```python
from app import load_model, generate_recipe
from PIL import Image

# Load trained model and vocabulary
model, vocab, device = load_model()

# Generate recipe from image
image = Image.open('your_image.jpg')
recipe = generate_recipe(image, model, vocab, device)
print(recipe)
```

---

## 📈 Results

### Training Performance

| Metric | Value |
|--------|-------|
| Best Validation Loss | 2.96 |
| Training Time | ~1 hours (15 epochs on Tesla T4) |
| Final Train Loss | 2.92 |
| Model Parameters | 17.7M |

### Sample Predictions

**Example: Pizza**
- **Generated**: "preheat oven to 450 degrees, roll out pizza dough spread tomato sauce add mozzarella cheese top with pepperoni bake for 15 minutes until cheese is melted and bubbly."
- **Quality**: Coherent, logical sequence ✅

---

## 🛠️ Technical Details

### Hyperparameters

```python
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 15
EMBED_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LENGTH = 512
```

### Model Architecture Details

```
CNNEncoder(
  (resnet): Sequential(...)
  (linear): Linear(512 → 256)
  (bn): BatchNorm1d(256)
  (relu): ReLU()
)

TransformerDecoder(
  (embedding): Embedding(8215, 256)
  (positional_encoding): Parameter(512, 256)
  (transformer_decoder): TransformerDecoder(2 layers)
  (fc_out): Linear(256 → 8215)
)

Total Parameters: 17,761,623
Trainable Parameters: 7,042,815
```

---

## 📁 Project Structure

```
inverse-cooking/
├── app/
│   ├── core/
│   │   └── config.py              # Environment variables, paths, model IDs
│   ├── services/
│   │   ├── swiggy_auth.py         # OAuth 2.1 + PKCE authentication
│   │   ├── swiggy_service.py      # Swiggy MCP ordering flow
│   │   ├── vision_service.py      # Qwen Vision AI (dish identification)
│   │   └── matcher_service.py     # Fuzzy match against chef recipes
│   ├── storage/
│   │   └── recipe_repo.py         # JSON persistence for chef recipes
│   ├── api/v1/
│   │   └── recipes.py             # FastAPI REST endpoints
│   ├── schemas/
│   │   └── recipe.py              # Pydantic request/response models
│   └── main.py                    # FastAPI server entry point
├── ui/
│   ├── main_app.py                # Streamlit home page (upload → identify → order)
│   └── pages/
│       └── 1_Chef_Portal.py       # Chef recipe management page
├── data/
│   └── recipes.json               # Chef-uploaded verified recipes
├── .env                           # HF_TOKEN (gitignored)
├── requirements.txt               # Python dependencies
└── readme.md                      # This file
```

---

## 🔧 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
pillow>=9.0.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
kagglehub>=0.1.0
gdown>=4.7.1
```
