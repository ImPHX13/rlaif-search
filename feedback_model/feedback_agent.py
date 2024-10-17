from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FeedbackAgent:
    """
    A feedback agent for predicting the relevance of a document to a given query.
    """
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initializes the feedback agent with a tokenizer and a model.

        Args:
            model_name (str, optional): The name of the pre-trained model to use. Defaults to 'bert-base-uncased'.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    def predict_relevance(self, query, document):
        """
        Predicts the relevance of a document to a given query.

        Args:
            query (str): The search query.
            document (str): The document to evaluate.

        Returns:
            float: A relevance score between 0 and 1.
        """
        inputs = self.tokenizer(query, document, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        relevance_score = torch.sigmoid(outputs.logits).item()
        return relevance_score