import torch
from model import LungCancerCNN

def load_model(model_path):
    model = LungCancerCNN(3,32,1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, X):
    prediction = model(X)
    return prediction

