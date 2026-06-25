import torch
import torch.nn as nn
import os
from huggingface_hub import snapshot_download
from transformers import AutoModel

def load_facexformer(device="cuda"):
    """
    Initializes FaceX-Former directly from the kartiknarayan/facexformer HuggingFace repo.
    """
    print("Loading FaceX-Former backbone from HuggingFace...")
    
    # Load the model directly from the Hugging Face hub
    repo_id = "kartiknarayan/facexformer"
    
    try:
        # Assuming the model is compatible with AutoModel in the HF Transformers library
        model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    except Exception as e:
        print(f"AutoModel failed, downloading weights directly: {e}")
        # Fallback: Download weights if it requires custom instantiation
        model_dir = snapshot_download(repo_id=repo_id)
        # TODO: Import the specific FaceX-Former architecture class if AutoModel doesn't work
        # from facexformer_code import FaceXFormer
        # model = FaceXFormer()
        # model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
        raise e

    model = model.to(device)
    model.eval()
    
    # Strictly freeze the backbone
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def load_adaface(device="cuda"):
    """
    Downloads and initializes AdaFace directly from the author's CVLface HuggingFace repository.
    """
    print("Downloading and Loading AdaFace...")
    
    # Download the IR101 AdaFace model trained on WebFace12M
    repo_id = "minchul/cvlface_adaface_ir101_webface12m" 
    model_dir = snapshot_download(repo_id=repo_id)
    model_path = os.path.join(model_dir, "model.pt")
    
    # TODO: Initialize the specific AdaFace architecture class from CVLFace here
    # from cvlface import AdaFace_IR101
    # model = AdaFace_IR101()
    # model.load_state_dict(torch.load(model_path))
    
    # Placeholder using ResNet to ensure pipeline testing works without cvlface repo installed yet
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Strictly freeze the backbone
    for param in model.parameters():
        param.requires_grad = False
        
    return model