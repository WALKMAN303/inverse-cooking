import gdown
import os

def download_models():
    """Download pre-trained models from Google Drive"""
    
    print("📥 Downloading model files from Google Drive...")
    files = {
        'best_model.pth': '1HfSb-zVBlxTf22YrheEVGbT7KjZHoe-C',
        'vocab.pkl': '13S6BM-Fc5uPCO-Tn1VJxxUnx3U0zFlHD'                       
    }
    
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)
            print(f"✅ {filename} downloaded successfully!")
        else:
            print(f"⏭️  {filename} already exists, skipping...")
    
    print("\n✨ All models downloaded successfully!")

if __name__ == "__main__":
    download_models()