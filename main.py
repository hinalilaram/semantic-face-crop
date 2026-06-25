import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import CelebAMaskHQDataset
from models.feature_extractor import AdaFaceExtractor, FaceXExtractor
from models.mlp import train_mlp

def extract_and_save_embeddings(args, device):
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 1. Initialize the specified model
    if args.model == "facex":
        extractor = FaceXExtractor(device)
    elif args.model == "adaface":
        extractor = AdaFaceExtractor(device)
    else:
        raise ValueError("Model must be 'facex' or 'adaface'")

    # 2. Setup Dataset
    image_dir = "./data/CelebAMask-HQ/CelebA-HQ-img"
    mask_dir = "./data/CelebAMask-HQ/CelebA-HQ-mask-anno"
    mapping_file = "./data/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt"

    # 2. Setup Dataset (It handles the mapping file automatically now!)
    dataset = CelebAMaskHQDataset(
        image_dir=image_dir, 
        mask_dir=mask_dir,   
        mapping_file=mapping_file, 
        strategy=args.strategy
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    # 3. Extraction Loop
    all_embeddings = []
    all_labels = []
    
    print(f"Extracting features using {args.model} with strategy '{args.strategy}'...")
    # --- Wrap the dataloader in tqdm ---
    for images, labels in tqdm(dataloader, desc=f"Processing {args.strategy}", unit="batch"):
        images = images.to(device)
        embeddings = extractor.extract(images)
        
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
    # -----------------------------------
        
    final_embeddings = torch.cat(all_embeddings)
    final_labels = torch.cat(all_labels)

    # 4. Save the file
    save_name = f"{args.model}_{args.strategy}_embeddings.pt"
    save_path = os.path.join(args.results_dir, save_name)
    torch.save({'embeddings': final_embeddings, 'labels': final_labels}, save_path)
    print(f"Successfully saved embeddings to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="HLCV Foundation Model Pipeline")
    parser.add_argument("--action", type=str, required=True, choices=["extract", "train"], help="Choose whether to extract features or train the MLP")
    parser.add_argument("--model", type=str, choices=["facex", "adaface"], help="Which model to use for extraction")
    parser.add_argument("--strategy", type=str, default="no_aug", help="Augmentation strategy to apply")
    parser.add_argument("--embeddings_file", type=str, help="Path to the .pt file (required for --action train)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save output files")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.action == "extract":
        if not args.model:
            raise ValueError("--model is required when action is 'extract'")
        extract_and_save_embeddings(args, device)
        
    elif args.action == "train":
        if not args.embeddings_file:
            raise ValueError("--embeddings_file is required when action is 'train'")
        
        print("Starting MLP Training...")
        model = train_mlp(args.embeddings_file)
        
        # Save the trained classifier weights
        save_path = args.embeddings_file.replace("_embeddings.pt", "_mlp.ckpt")
        torch.save(model.state_dict(), save_path)
        print(f"MLP weights saved to {save_path}")

if __name__ == '__main__':
    main()