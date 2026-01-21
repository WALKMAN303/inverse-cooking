"""
Inverse Cooking - Streamlit Web Application
Generate recipes from food images using deep learning
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# VOCABULARY CLASS (Must match training code)
# =============================================================================

class Vocabulary:
    """Vocabulary class to handle word-to-index mappings"""
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] = frequencies.get(word, 0) + 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        tokenized_text = text.split()
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class CNN_Encoder(nn.Module):
    def __init__(self, embed_size):
        super(CNN_Encoder, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        for param in resnet.parameters():
            param.requires_grad = False
        
        layers = list(resnet.children())
        self.resnet = nn.Sequential(*layers[:-1])
        self.linear = nn.Linear(512, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.relu = nn.ReLU()
    
    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.relu(self.bn(self.linear(features)))
        return features

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_layers, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, embed_size))
        self.feature_projection = nn.Linear(embed_size, embed_size * 8)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=nhead, 
            dim_feedforward=embed_size * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, features, captions):
        batch_size, seq_len = captions.shape
        embeddings = self.embedding(captions)
        embeddings = embeddings + self.positional_encoding[:seq_len, :].unsqueeze(0)
        embeddings = self.dropout(embeddings)
        
        projected = self.feature_projection(features)
        memory = projected.view(batch_size, 8, -1)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        tgt_mask = tgt_mask.to(captions.device)
        
        output = self.transformer_decoder(tgt=embeddings, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(output)

class InverseCookingModel(nn.Module):
    def __init__(self, embed_size, vocab_size, nhead, num_layers, max_seq_length):
        super(InverseCookingModel, self).__init__()
        self.encoder = CNN_Encoder(embed_size)
        self.decoder = TransformerDecoder(vocab_size, embed_size, nhead, num_layers, max_seq_length)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# =============================================================================
# MODEL LOADING AND INFERENCE
# =============================================================================

@st.cache_resource
def load_model():
    """Load model and vocabulary (cached for efficiency)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load vocabulary
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        
        # Load checkpoint
        checkpoint = torch.load('best_model.pth', map_location=device)
        
        # Get config - FIX THE TYPO FROM TRAINING
        config = checkpoint.get('config', {
            'embed_size': 256,
            'num_heads': 4,
            'num_layers': 2,
            'max_seq_length': 512
        })
        
        # Handle the typo in saved config: 'num_head' vs 'num_heads'
        if 'num_head' in config and 'num_heads' not in config:
            config['num_heads'] = config['num_head']
        
        vocab_size = len(vocab.stoi)
        
        # Initialize model
        model = InverseCookingModel(
            embed_size=config['embed_size'],
            vocab_size=vocab_size,
            nhead=config['num_heads'],
            num_layers=config['num_layers'],
            max_seq_length=config['max_seq_length']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, vocab, device
    
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Error: {str(e)}")
        st.error("Please ensure 'vocab.pkl' and 'best_inverse_cooking_model.pth' exist in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        st.stop()

def generate_recipe(image, model, vocab, device, max_len=200, temperature=0.7, top_k=50):
    """Generate recipe from PIL Image"""
    transform = transforms.Compose([
        transforms.Resize((132, 132)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    
    with torch.no_grad():
        img = transform(image).unsqueeze(0).to(device)
        features = model.encoder(img)
        recipe_ids = [vocab.stoi["<START>"]]
        
        for _ in range(max_len):
            captions = torch.tensor([recipe_ids]).to(device)
            outputs = model.decoder(features, captions)
            
            logits = outputs[0, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            predicted_id = torch.multinomial(probs, 1).item()
            
            recipe_ids.append(predicted_id)
            
            if predicted_id == vocab.stoi["<END>"]:
                break
        
        words = [vocab.itos[i] for i in recipe_ids]
        words = [w for w in words if w not in ['<START>', '<END>', '<PAD>']]
        recipe = ' '.join(words)
        
        return recipe

def format_recipe(recipe_text):
    """Format recipe text for better readability"""
    if not recipe_text or recipe_text.strip() == "":
        return "Recipe generation failed. Please try again with a different image or settings."
    
    # Capitalize first letter
    recipe_text = recipe_text.strip().capitalize()
    
    # Add periods if missing
    sentences = recipe_text.split('.')
    formatted = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Capitalize first letter of each sentence
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            formatted.append(sentence)
    
    result = '. '.join(formatted)
    if result and not result.endswith('.'):
        result += '.'
    
    return result

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Inverse Cooking AI",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .recipe-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #FF6B6B;
            margin: 20px 0;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF6B6B;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #FF5252;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🍳 Inverse Cooking AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a food image and let AI generate the recipe!</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('🔄 Loading AI model...'):
        model, vocab, device = load_model()
    
    st.success(f'✅ Model loaded successfully! Using: {device}')
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("---")
        st.subheader("Generation Parameters")
        
        temperature = st.slider(
            "🌡️ Temperature (Creativity)",
            min_value=0.3,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Lower = more conservative, Higher = more creative"
        )
        
        top_k = st.slider(
            "🎯 Top-K Sampling",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Number of words to consider at each step"
        )
        
        max_length = st.slider(
            "📏 Max Recipe Length",
            min_value=50,
            max_value=300,
            value=200,
            step=50,
            help="Maximum number of words to generate"
        )
        
        st.markdown("---")
        st.subheader("📊 Model Info")
        st.info(f"""
        **Vocabulary Size:** {len(vocab.stoi):,} words  
        **Model Parameters:** 17.7M  
        **Architecture:** ResNet18 + Transformer
        """)
        
        st.markdown("---")
        st.subheader("💡 Tips")
        st.markdown("""
        - **Clear images** work best
        - Try different temperatures for variety
        - Lower temperature for accurate recipes
        - Higher temperature for creative variations
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Food Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a food dish"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Generate button
            if st.button("🔮 Generate Recipe", type="primary"):
                with st.spinner('👨‍🍳 AI is cooking up a recipe...'):
                    try:
                        recipe = generate_recipe(
                            image, model, vocab, device,
                            max_len=max_length,
                            temperature=temperature,
                            top_k=top_k
                        )
                        
                        # Store in session state
                        st.session_state.recipe = recipe
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating recipe: {str(e)}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
        else:
            st.info("👆 Please upload a food image to get started!")
            
            # Sample images section
            st.markdown("---")
            st.subheader("🖼️ Or try a sample image")
            
            sample_images = {
                "Pizza": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=400",
                "Pasta": "https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=400",
                "Salad": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400"
            }
            
            selected_sample = st.selectbox("Choose a sample:", list(sample_images.keys()))
            
            if st.button("Use Sample Image"):
                st.info("📝 Note: Download the sample image and upload it above for best results!")
    
    with col2:
        st.header("📋 Generated Recipe")
        
        if 'recipe' in st.session_state and st.session_state.recipe:
            recipe = st.session_state.recipe
            formatted_recipe = format_recipe(recipe)
            
            # Action buttons
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.download_button(
                    label="📥 Download Recipe",
                    data=formatted_recipe,
                    file_name="recipe.txt",
                    mime="text/plain"
                )
            
            with col_b:
                if st.button("🔄 Generate Another"):
                    if uploaded_file is not None:
                        image = Image.open(uploaded_file).convert('RGB')
                        with st.spinner('👨‍🍳 Generating new variation...'):
                            try:
                                recipe = generate_recipe(
                                    image, model, vocab, device,
                                    max_len=max_length,
                                    temperature=temperature,
                                    top_k=top_k
                                )
                                st.session_state.recipe = recipe
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error generating recipe: {str(e)}")
            
            # Copy to clipboard (using text area)
            st.text_area(
                "Copy text:",
                formatted_recipe,
                height=150,
                help="Select all and copy (Ctrl+C / Cmd+C)"
            )
            
        else:
            st.info("🍽️ Your generated recipe will appear here!")
            
            st.markdown("---")
            st.subheader("✨ How it works")
            st.markdown("""
            1. **Upload** a food image
            2. **AI analyzes** the image using ResNet18 CNN
            3. **Transformer** generates recipe instructions
            4. **Get** your recipe in seconds!
            
            The model was trained on 13,000+ recipes and can generate 
            instructions for a wide variety of dishes.
            """)
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("### 🎓 About")
        st.markdown("Deep learning project using PyTorch")
    
    with col_f2:
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("ResNet18 • Transformer • PyTorch")
    
    with col_f3:
        st.markdown("### 📧 Contact")
        st.markdown("Built for portfolio and learning")

if __name__ == "__main__":
    main()
