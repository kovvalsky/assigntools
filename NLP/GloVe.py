import torch
import os
from tqdm import tqdm

class GloVeEmbeddings:
    """
    A class to load and access GloVe embeddings.
    Provides:
    - vectors: a torch tensor of shape (vocab_size, embedding_dim)
    - stoi: a dictionary mapping strings to indices in the vectors tensor
    - itos: a dictionary mapping indices to strings
    """
    def __init__(self, glove_path):
        """
        Initialize GloVe embeddings from a file.
        Args:
            glove_path (str): Path to the GloVe file (e.g. glove.6B.100d.txt)
        """
        self.stoi = {}  # string to index
        self.itos = {}  # index to string
        embeddings = []
        
        print(f"Loading GloVe embeddings from {glove_path}...")
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found at {glove_path}")
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                values = line.strip().split(' ')
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]])
                
                # Store the word and its vector
                self.stoi[word] = i
                self.itos[i] = word
                embeddings.append(vector)
        
        # Stack all embeddings into a single tensor
        self.vectors = torch.stack(embeddings)
        
        print(f"Loaded {len(self.stoi)} words with dimension {self.vectors.shape[1]}")
    
    def get_vector(self, word):
        """Get the embedding vector for a word"""
        if word in self.stoi:
            idx = self.stoi[word]
            return self.vectors[idx]
        return None
