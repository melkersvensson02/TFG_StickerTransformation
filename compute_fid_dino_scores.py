#!/usr/bin/env python3
"""
1. FID (Fréchet Inception Distance) - measures image quality and diversity
2. DINO (Vision Transformer-based) - measures structural similarity

Usage:
    python compute_fid_dino_scores.py \
        --dataset1_dir /path/to/photos \
        --dataset2_dir /path/to/stickers \
        --output_dir ./results

// Execution example:
python compute_fid_dino_scores.py --dataset1_dir /data/upftfg19/mfsvensson/Data_TFG/myPairedDatasets/train_A 
--dataset2_dir /data/upftfg19/mfsvensson/Data_TFG/myPairedDatasets/train_B --output_dir .
"""

import os
import sys
import json
import argparse
import warnings
import gc
from pathlib import Path
from glob import glob
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

# Import FID utilities
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from my_utils.dino_struct import DinoStructureLoss

warnings.filterwarnings('ignore')


class DeviceManager:
    """Handles device selection (GPU/CPU) with AMD GPU support"""
    
    @staticmethod
    def get_device() -> torch.device:
        """
        Auto-detect and select appropriate device.
        Priority: AMD GPU (ROCm) > NVIDIA GPU > CPU
        """
        if torch.cuda.is_available():
            # Try to detect AMD GPU
            try:
                # Check if we're using ROCm (AMD)
                if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    print(f"✓ GPU detected: {device_name}")
                    return torch.device("cuda")
            except Exception as e:
                print(f"Warning: Could not detect GPU type: {e}")
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("ℹ Using CPU (GPU not available)")
        return torch.device("cpu")
    
    @staticmethod
    def empty_cache():
        """Clear GPU cache if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ImagePairer:
    """Handles pairing of images from two datasets by filename"""
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    
    @staticmethod
    def find_images(directory: str) -> List[str]:
        """Find all image files in directory recursively"""
        images = []
        for ext in ImagePairer.IMAGE_EXTENSIONS:
            # Search recursively
            pattern = os.path.join(directory, "**", f"*{ext}")
            images.extend(glob(pattern, recursive=True))
            pattern = os.path.join(directory, "**", f"*{ext.upper()}")
            images.extend(glob(pattern, recursive=True))
        
        return sorted(list(set(images)))  # Remove duplicates and sort
    
    @staticmethod
    def pair_images(dir1: str, dir2: str) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
        images1 = ImagePairer.find_images(dir1)
        images2 = ImagePairer.find_images(dir2)
        
        # Get basenames for pairing
        basenames1 = {os.path.basename(img): img for img in images1}
        basenames2 = {os.path.basename(img): img for img in images2}
        
        # Find common basenames
        common = set(basenames1.keys()) & set(basenames2.keys())
        
        paired_images = [(basenames1[bn], basenames2[bn]) for bn in sorted(common)]
        only_in_dir1 = sorted([basenames1[bn] for bn in basenames1 if bn not in common])
        only_in_dir2 = sorted([basenames2[bn] for bn in basenames2 if bn not in common])
        
        return paired_images, only_in_dir1, only_in_dir2


class DatasetPreparer:
    @staticmethod
    # Takes images from a dataset and computes reference statistics (mean and covariance) that will be used as a baseline for comparison
    def prepare_fid_reference(dataset_dir: str, output_dir: str, 
                             transform: Optional[transforms.Compose] = None) -> Tuple[np.ndarray, np.ndarray]:
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all images that have a paired version 
        images = ImagePairer.find_images(dataset_dir)
        if not images:
            raise ValueError(f"No images found in {dataset_dir}")
        
        print(f"  Preparing {len(images)} reference images...")
        
        # Transform and save images
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ])
        
        for img_path in tqdm(images, desc="  Transforming images", leave=False):
            try:
                img = Image.open(img_path).convert("RGB")
                img_transformed = transform(img)
                
                # Save as PNG
                basename = os.path.basename(img_path)
                out_path = os.path.join(output_dir, os.path.splitext(basename)[0] + ".png")
                
                if not os.path.exists(out_path):
                    # Convert back to PIL and save
                    transforms.ToPILImage()(img_transformed).save(out_path)
            except Exception as e:
                print(f"  Warning: Could not process {img_path}: {e}")
        
        # Compute FID features
        print(f"  Computing FID features...")
        DeviceManager.empty_cache()
        # Runs them through a pre-trained feature extractor (InceptionV3) to get feature vectors
        feat_model = build_feature_extractor("clean", "cuda" if torch.cuda.is_available() else "cpu", 
                                             use_dataparallel=False)
        # Computes mu (mean) and sigma (covariance) of those features
        features = get_folder_features(
            output_dir, 
            model=feat_model, 
            num_workers=0, 
            num=None,
            shuffle=False, 
            seed=0, 
            batch_size=8, 
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            mode="clean", 
            custom_fn_resize=None, 
            description="", 
            verbose=False,
            custom_image_tranform=None
        )
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma


class FIDComputer:
    # Measures the distance between generated images and reference statistics
    @staticmethod
    def compute_fid(generated_dir: str, ref_mu: np.ndarray, ref_sigma: np.ndarray,
                   device: torch.device) -> float:
        #   - Takes the generated/test images (from dataset1)
        #   - Extracts their features using the same pre-trained model
        #   - Computes their mean (gen_mu) and covariance (gen_sigma)
        feat_model = build_feature_extractor("clean", device, use_dataparallel=False)
        
        # Get features from generated images
        gen_features = get_folder_features(
            generated_dir,
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=device,
            mode="clean",
            custom_fn_resize=None,
            description="",
            verbose=False,
            custom_image_tranform=None
        )
        
        # Compute FID
        gen_mu = np.mean(gen_features, axis=0)
        gen_sigma = np.cov(gen_features, rowvar=False)
        
        fid_score = frechet_distance(ref_mu, ref_sigma, gen_mu, gen_sigma)
        
        return fid_score


class DINOComputer:
    # Loads the pre-trained DINO model (a self-supervised vision transformer)
    def __init__(self, device: torch.device):
        """Initialize DINO model"""
        self.device = device
        self.dino = DinoStructureLoss()
        print("✓ DINO model loaded")
    
    def compute_score(self, img1_pil: Image.Image, img2_pil: Image.Image) -> float:
        try:
            # Resize second image to match first
            img2_resized = img2_pil.resize(img1_pil.size, Image.LANCZOS)
            
            # Preprocess images
            img1_tensor = self.dino.preprocess(img1_pil).unsqueeze(0).to(self.device)
            img2_tensor = self.dino.preprocess(img2_resized).unsqueeze(0).to(self.device)
            
            # Compute DINO score
            with torch.no_grad():
                score = self.dino.calculate_global_ssim_loss(img1_tensor, img2_tensor).item()
            
            return score
        except Exception as e:
            print(f"    Warning: DINO computation failed: {e}")
            return None


class ResultsManager:
    """Manages results storage and reporting"""
    
    def __init__(self, output_dir: str):
        """Initialize results manager"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "script": "compute_fid_dino_scores.py"
            },
            "dataset_info": {},
            "pairing_info": {
                "total_paired": 0,
                "skipped_unpaired": 0
            },
            "fid_scores": {},
            "dino_scores": {
                "per_image": [],
                "statistics": {}
            }
        }
    
    def add_dataset_info(self, name: str, path: str, image_count: int):
        """Add dataset information"""
        self.results["dataset_info"][name] = {
            "path": path,
            "image_count": image_count
        }
    
    def add_pairing_info(self, paired: int, skipped: int):
        """Add pairing statistics"""
        self.results["pairing_info"]["total_paired"] = paired
        self.results["pairing_info"]["skipped_unpaired"] = skipped
    
    def add_fid_score(self, dataset_pair: str, score: float):
        """Add FID score"""
        self.results["fid_scores"][dataset_pair] = score
    
    def add_dino_score(self, image_name: str, score: float):
        """Add per-image DINO score"""
        self.results["dino_scores"]["per_image"].append({
            "image": image_name,
            "score": score
        })
    
    def compute_dino_statistics(self):
        """Compute DINO statistics"""
        scores = [item["score"] for item in self.results["dino_scores"]["per_image"] if item["score"] is not None]
        
        if scores:
            self.results["dino_scores"]["statistics"] = {
                "count": len(scores),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores))
            }
    
    def save_json(self, filename: str = "results.json"):
        """Save results to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {filepath}")
    
    def print_summary(self):
        """Print results summary to console"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS SUMMARY")
        print("="*70)
        
        # Dataset info
        print("\n📊 DATASET INFORMATION:")
        for name, info in self.results["dataset_info"].items():
            print(f"  {name}: {info['image_count']} images")
        
        # Pairing info
        print("\n🔗 PAIRING INFORMATION:")
        paired = self.results["pairing_info"]["total_paired"]
        skipped = self.results["pairing_info"]["skipped_unpaired"]
        print(f"  Paired images: {paired}")
        if skipped > 0:
            print(f"  Skipped (unpaired): {skipped}")
        
        # FID scores
        print("\n📈 FID SCORES (lower is better):")
        for dataset_pair, score in self.results["fid_scores"].items():
            print(f"  {dataset_pair}: {score:.4f}")
        
        # DINO statistics
        if self.results["dino_scores"]["statistics"]:
            stats = self.results["dino_scores"]["statistics"]
            print("\n🎯 DINO STRUCTURAL SIMILARITY (lower is better):")
            print(f"  Mean:   {stats['mean']:.6f}")
            print(f"  Std:    {stats['std']:.6f}")
            print(f"  Min:    {stats['min']:.6f}")
            print(f"  Max:    {stats['max']:.6f}")
            print(f"  Median: {stats['median']:.6f}")
            print(f"  Count:  {stats['count']}")
        
        print("\n" + "="*70)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compute FID and DINO scores for paired image datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset1_dir', type=str, required=True,
                       help='Path to first dataset (e.g., photos)')
    parser.add_argument('--dataset2_dir', type=str, required=True,
                       help='Path to second dataset (e.g., stickers)')
    parser.add_argument('--output_dir', type=str, default='./fid_dino_results',
                       help='Directory to save results')
    parser.add_argument('--skip_fid', action='store_true',
                       help='Skip FID computation')
    parser.add_argument('--skip_dino', action='store_true',
                       help='Skip DINO computation')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for FID computation')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (auto-detect by default)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Select device
    if args.device == 'auto':
        device = DeviceManager.get_device()
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print("FID & DINO SCORE COMPUTATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Dataset 1: {args.dataset1_dir}")
    print(f"Dataset 2: {args.dataset2_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize results manager
    results = ResultsManager(args.output_dir)
    
    # Step 1: Pair images
    print("\n1️⃣  PAIRING IMAGES...")
    print("-" * 70)
    
    try:
        paired_images, only_in_1, only_in_2 = ImagePairer.pair_images(
            args.dataset1_dir, args.dataset2_dir
        )
    except Exception as e:
        print(f"❌ Error pairing images: {e}")
        return
    
    if not paired_images:
        print("❌ No paired images found!")
        return
    
    # Get all images for reference
    all_images_1 = ImagePairer.find_images(args.dataset1_dir)
    all_images_2 = ImagePairer.find_images(args.dataset2_dir)
    
    print(f"✓ Dataset 1: {len(all_images_1)} images")
    print(f"✓ Dataset 2: {len(all_images_2)} images")
    print(f"✓ Paired images: {len(paired_images)}")
    
    if only_in_1:
        print(f"⚠ Only in Dataset 1 (skipped): {len(only_in_1)}")
    if only_in_2:
        print(f"⚠ Only in Dataset 2 (skipped): {len(only_in_2)}")
    
    # Add to results
    results.add_dataset_info("dataset1", args.dataset1_dir, len(all_images_1))
    results.add_dataset_info("dataset2", args.dataset2_dir, len(all_images_2))
    results.add_pairing_info(len(paired_images), len(only_in_1) + len(only_in_2))
    
    # Step 2: FID Computation
    if not args.skip_fid:
        print("\n2️⃣  FID COMPUTATION...")
        print("-" * 70)
        
        try:
            # These statistics represent "what this domain looks like" in feature space
            print("Preparing FID reference (from dataset2)...")
            ref_output_dir = os.path.join(args.output_dir, "fid_reference")
            ref_mu, ref_sigma = DatasetPreparer.prepare_fid_reference(
                args.dataset2_dir, ref_output_dir
            )
            print(f"✓ Reference statistics computed")
            
            # Generate images from dataset1 and compute FID
            print("Computing FID on paired images from dataset1...")
            generated_output_dir = os.path.join(args.output_dir, "fid_generated")
            os.makedirs(generated_output_dir, exist_ok=True)
            
            transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ])
            
            for img1_path, img2_path in tqdm(paired_images, desc="Preparing generated images"):
                try:
                    img = Image.open(img1_path).convert("RGB")
                    img_transformed = transform(img)
                    basename = os.path.basename(img1_path)
                    out_path = os.path.join(generated_output_dir, os.path.splitext(basename)[0] + ".png")
                    
                    if not os.path.exists(out_path):
                        transforms.ToPILImage()(img_transformed).save(out_path)
                except Exception as e:
                    print(f"  Warning: Could not process {img1_path}: {e}")
            
            # Compute FID score
            print("Computing FID score...")
            DeviceManager.empty_cache()
            fid_score = FIDComputer.compute_fid(generated_output_dir, ref_mu, ref_sigma, device)
            print(f"✓ FID Score (Dataset1 → Dataset2): {fid_score:.4f}")
            results.add_fid_score("dataset1_to_dataset2", fid_score)
            
        except Exception as e:
            print(f"❌ Error computing FID: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 3: DINO Computation
    if not args.skip_dino:
        print("\n3️⃣  DINO COMPUTATION...")
        print("-" * 70)
        
        try:
            dino_computer = DINOComputer(device)
            print(f"Computing DINO scores for {len(paired_images)} image pairs...")
            
            dino_scores = []
            for img1_path, img2_path in tqdm(paired_images, desc="  Computing DINO"):
                try:
                    img1 = Image.open(img1_path).convert("RGB")
                    img2 = Image.open(img2_path).convert("RGB")
                    
                    score = dino_computer.compute_score(img1, img2)
                    if score is not None:
                        dino_scores.append(score)
                        basename = os.path.basename(img1_path)
                        results.add_dino_score(basename, score)
                except Exception as e:
                    print(f"  Warning: Could not compute DINO for {os.path.basename(img1_path)}: {e}")
            
            results.compute_dino_statistics()
            print(f"✓ DINO scores computed for {len(dino_scores)} image pairs")
            
            # Clear DINO model from memory
            del dino_computer
            DeviceManager.empty_cache()
            
        except Exception as e:
            print(f"❌ Error computing DINO: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Save and display results
    print("\n4️⃣  SAVING RESULTS...")
    print("-" * 70)
    results.save_json()
    results.print_summary()
    
    print("\n✅ Computation complete!")


if __name__ == "__main__":
    main()
