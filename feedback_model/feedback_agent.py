from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FeedbackAgent:
    def __init__(self, model_name='bert-base-uncased'):
        # initialize the feedback agent with a tokenizer and a model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    def predict_relevance(self, query, document):
        # prepare the inputs for the model
        inputs = self.tokenizer(query, document, return_tensors='pt', truncation=True, padding=True, max_length=512)
        # make predictions without gradient calculation
        with torch.no_grad():
            outputs = self.model(**inputs)
        # apply sigmoid to the model's output logits to get a relevance score between 0 and 1
        relevance_score = torch.sigmoid(outputs.logits).item()
        return relevance_score