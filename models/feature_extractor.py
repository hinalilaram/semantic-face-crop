import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Import the model architecture directly from the cloned repo
# Make sure the 'facexformer' folder is in the same directory as main.py
import sys
sys.path.append('./facexformer') 
from facexformer.network.models import FaceXFormer # Replace 'build_model' with their actual initialization function if different

class FaceXExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        print("Loading FaceX-Former...")
        
        # 1. Initialize the architecture
        self.model = FaceXFormer() 
        
        # 2. Load the checkpoint
        weights_path = "weights/facexformer.pt"
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # 3. FIX: Load ONLY the specific dictionary key containing the weights
        self.model.load_state_dict(checkpoint['state_dict_backbone'])
        self.model = self.model.to(self.device)
        
        # 4. Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def extract(self, images):
        with torch.no_grad():
            # FIX: We bypass their multi-task forward pass completely.
            # We pass the images directly into their Swin Transformer backbone 
            # to get the pure latent representation embeddings.
            
            # Use self.model.backbone instead of self.model
            features = self.model.backbone(images) 
            
            # Swin Transformers usually output spatial grids (e.g., [Batch, Channels, H, W] or [Batch, Seq, Channels])
            # We pool them into a flat 1D embedding vector per image [Batch, Channels] for your MLP
            if features.dim() == 4:
                embeddings = torch.flatten(nn.functional.adaptive_avg_pool2d(features, (1, 1)), 1)
            elif features.dim() == 3:
                embeddings = features.mean(dim=1)
            else:
                embeddings = features
                
        return embeddings


class AdaFaceExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        print("Loading AdaFace...")
        self.model = AutoModel.from_pretrained("minchul/cvlface_adaface_ir50_webface4m", trust_remote_code=True).to(self.device)
        
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def extract(self, images):
        with torch.no_grad():
            embeddings = self.model(images)
        return embeddings