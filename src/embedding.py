import numpy as np
from sentence_transformers import SentenceTransformer

#USE
#embedding_manager = EmbeddingManager()

# EmbeddingManager
class EmbeddingManager:
    """Handles Document Embedding Generation using SentenceTransformer"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading Embedding Model:{self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model Loaded Succesfully , Embedding Dimensions :{self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error:{e}")
            raise

    def generate_embeddings(self, texts: list) -> np.ndarray: 
        """Generate an embedding vector for the given text"""
        if self.model is None: 
            raise ValueError("ModelNotLoaded")
        print(f"Generating Embeddings for {len(texts)} text(s)...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated Embedding Model with shape {embeddings.shape}")
        return embeddings
