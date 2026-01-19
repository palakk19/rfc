import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class RobustnessEvaluator:
    def __init__(self, embedding_model="emilyalsentzer/Bio_ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.model.eval()

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token embedding
        return outputs.last_hidden_state[:, 0, :].numpy()

    def calculate_metrics(self, original_texts, perturbed_texts, original_preds, perturbed_preds):
        """
        1. Input Semantic Preservation (Cosine sim of ClinicalBERT embeddings)
        2. Prediction Drift (Binary: Did the label change?)
        """
        sim_scores = []
        drift_count = 0
        
        for i in range(len(original_texts)):
            # 1. Input Similarity
            emb_orig = self.get_embedding(original_texts[i])
            emb_pert = self.get_embedding(perturbed_texts[i])
            sim = cosine_similarity(emb_orig, emb_pert)[0][0]
            sim_scores.append(sim)
            
            # 2. Prediction Drift (Assuming simple string match or class ID match)
            if original_preds[i] != perturbed_preds[i]:
                drift_count += 1
                
        return {
            "mean_semantic_similarity": np.mean(sim_scores),
            "prediction_drift_rate": drift_count / len(original_texts)
        }