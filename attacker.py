# import spacy
# import torch
# import random
# import warnings
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# # Suppress warnings for cleaner output
# warnings.filterwarnings("ignore")

# class MedicalPerturber:
#     def __init__(self, spacy_model="en_core_sci_sm", bert_model="dmis-lab/biobert-v1.1"):
#         print("Initializing Attack Engine (SciSpacy + BioBERT)...")
#         try:
#             self.nlp = spacy.load(spacy_model)
#         except OSError:
#             raise OSError(f"Could not find spaCy model '{spacy_model}'. Please install it via pip.")
            
#         self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
#         self.model = AutoModelForMaskedLM.from_pretrained(bert_model)
#         self.model.eval()

#     def get_synonyms(self, text, entity_span, top_k=10):
#         masked_text = text[:entity_span.start_char] + self.tokenizer.mask_token + text[entity_span.end_char:]
#         inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
        
#         with torch.no_grad():
#             outputs = self.model(**inputs)
            
#         mask_index = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
#         if len(mask_index) == 0: return [] # Safety check
        
#         probs = outputs.logits[0, mask_index[0], :]
#         top_k_ids = torch.topk(probs, top_k, dim=0).indices.tolist()
        
#         candidates = [self.tokenizer.decode([idx]).strip() for idx in top_k_ids]
#         # Filter: remove original word, short words, and subwords (start with ##)
#         candidates = [c for c in candidates if c.lower() != entity_span.text.lower() and len(c) > 3 and not c.startswith("##")]
#         return candidates

#     def perturb(self, text, perturbation_rate=0.15):
#         """
#         perturbation_rate: Percentage of entities to replace (0.15 = 15%)
#         """
#         doc = self.nlp(text)
#         entities = list(doc.ents)
        
#         if not entities:
#             return text

#         # Select random entities to attack
#         num_to_perturb = max(1, int(len(entities) * perturbation_rate))
#         target_entities = random.sample(entities, min(num_to_perturb, len(entities)))
        
#         # Sort reverse to handle string replacement indices correctly
#         target_entities.sort(key=lambda x: x.start_char, reverse=True)

#         perturbed_text = text
#         for ent in target_entities:
#             candidates = self.get_synonyms(text, ent)
#             if candidates:
#                 # Pick the first valid synonym
#                 replacement = candidates[0]
#                 perturbed_text = perturbed_text[:ent.start_char] + replacement + perturbed_text[ent.end_char:]
                
#         return perturbed_text