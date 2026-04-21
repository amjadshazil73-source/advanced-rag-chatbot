from sentence_transformers import CrossEncoder
import os

def warmup():
    """
    Downloads and caches the CrossEncoder model files locally during the Docker build process.
    This prevents the 'Connection refused' and slow startup issues on Render.com.
    """
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"--- PRE-CACHING MODEL: {model_name} ---")
    
    # This force downloads the model to the default HF cache directory
    # which we will keep in the Docker image.
    CrossEncoder(model_name)
    
    print("--- MODEL CACHED SUCCESSFULLY ---")

if __name__ == "__main__":
    warmup()
