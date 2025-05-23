# Placeholder for dataset_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import io
import pyarrow.parquet as pq
import os
import torchvision.transforms as transforms

# Configuration (could be moved to a config file or passed as arguments)
# Image dimensions are H, W, C based on previous inspection
IMAGE_HEIGHT = 218
IMAGE_WIDTH = 178
IMAGE_CHANNELS = 3

class CelebAParquetDataset(Dataset):
    """CelebA dataset loaded from Parquet files."""
    def __init__(self, parquet_root_dir, transform=None):
        """
        Args:
            parquet_root_dir (string): Directory containing the Parquet files 
                                       (e.g., '/home/cs/OT-diffusion-flow/dataset/data').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.parquet_root_dir = parquet_root_dir
        self.transform = transform
        self.parquet_files = sorted([
            os.path.join(parquet_root_dir, f)
            for f in os.listdir(parquet_root_dir)
            if f.endswith(".parquet")
        ])

        self.tables = [pq.read_table(f) for f in self.parquet_files]
        # Concatenate all tables into one. 
        # This might be memory-intensive for very large datasets.
        # Consider lazy loading or a more advanced way to handle multiple Parquet files if memory becomes an issue.
        # For CelebA (around 1.3GB total), this should be acceptable.
        self.concatenated_table = pd.concat([t.to_pandas() for t in self.tables], ignore_index=True)
        
        # Filter out any rows where the image data might be missing or malformed, just in case
        # Assuming image data is in a column named 'image' and is a dict with a 'bytes' key
        self.concatenated_table = self.concatenated_table[
            self.concatenated_table['image'].apply(lambda x: isinstance(x, dict) and 'bytes' in x and x['bytes'] is not None)
        ]
        self.num_samples = len(self.concatenated_table)
        print(f"Loaded {self.num_samples} samples from {len(self.parquet_files)} Parquet files in {parquet_root_dir}.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_data = self.concatenated_table.iloc[idx]['image']
        image_bytes = image_data['bytes']
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a placeholder or skip. For now, let's return a black image.
            # This should ideally not happen if preprocessing/filtering is robust.
            image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image

def get_data_loader(dataset_path, batch_size, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the CelebA Parquet dataset.
    Args:
        dataset_path (string): Path to the directory containing the data 
                               (e.g., '/home/cs/OT-diffusion-flow/dataset/data')
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
    """
    # Define standard transformations for images
    # Resize to the determined dimensions (though CelebA images are already 218x178)
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), # Ensure size, though likely redundant if source is consistent
        transforms.ToTensor(), # Converts to [C, H, W] and scales to [0.0, 1.0]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1.0, 1.0]
    ])

    dataset = CelebAParquetDataset(parquet_root_dir=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader

# --- Utility function to get image properties (can be kept or removed after confirmation) ---
def get_image_properties(dataset_data_dir):
    # ... (previous implementation can be here if needed for standalone check)
    # For now, it's integrated into the understanding, so not strictly needed in the final loader.
    # However, having it as a callable utility might be useful for debugging.
    try:
        parquet_files = [f for f in os.listdir(dataset_data_dir) if f.endswith(".parquet")]
        if not parquet_files:
            print(f"No Parquet files found in {dataset_data_dir}")
            return None

        first_parquet_file = os.path.join(dataset_data_dir, parquet_files[0])
        table = pq.read_table(first_parquet_file, columns=['image'])
        if 'image' not in table.column_names:
            print("Error: 'image' column not found in Parquet file.")
            return None
        
        image_bytes_dict = table.column('image')[0].as_py()
        if not image_bytes_dict or 'bytes' not in image_bytes_dict:
            print("Error: Image data format is not as expected (missing 'bytes' field).")
            # Try to handle cases where 'path' might be present (e.g. Hugging Face datasets format)
            if isinstance(image_bytes_dict, dict) and 'path' in image_bytes_dict and image_bytes_dict['path']:
                # This requires the image files to be accessible relative to the parquet file or a known root.
                # For simplicity, we'll assume the 'bytes' field is primary based on initial inspection.
                print(f"Found 'path' field: {image_bytes_dict['path']}. This loader currently prioritizes 'bytes' field.")
                # If 'bytes' is truly missing, this path would need robust handling.
                return None # Or attempt to load from path if that's the new confirmed structure
            return None

        image_bytes = image_bytes_dict['bytes']
        if image_bytes is None:
            print("Error: Image bytes are None.")
            return None

        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        channels = len(image.getbands())
        return width, height, channels
    except Exception as e:
        print(f"An error occurred while reading image properties: {e}")
        return None

if __name__ == "__main__":
    # Path to the directory containing the data Parquet files
    # This is /home/cs/OT-diffusion-flow/dataset/data/
    celeba_data_dir = "/home/cs/OT-diffusion-flow/dataset/data" 
    
    print(f"Attempting to get image properties from: {celeba_data_dir}")
    img_props = get_image_properties(celeba_data_dir)
    if img_props:
        print(f"Confirmed image properties: Width={img_props[0]}, Height={img_props[1]}, Channels={img_props[2]}")
        # Update constants if necessary, though they should match.
        assert IMAGE_WIDTH == img_props[0], "IMAGE_WIDTH mismatch!"
        assert IMAGE_HEIGHT == img_props[1], "IMAGE_HEIGHT mismatch!"
        assert IMAGE_CHANNELS == img_props[2], "IMAGE_CHANNELS mismatch!"
    else:
        print("Could not confirm image properties. Using defaults.")

    print(f"\nAttempting to create DataLoader with batch size 4...")
    try:
        data_loader = get_data_loader(dataset_path=celeba_data_dir, batch_size=4, num_workers=0) # num_workers=0 for easier debugging
        print(f"DataLoader created. Number of batches: {len(data_loader)}")
        
        # Get one batch to test
        for i, batch in enumerate(data_loader):
            print(f"Batch {i+1} shape: {batch.shape}") # batch is expected to be images
            print(f"Batch {i+1} dtype: {batch.dtype}")
            print(f"Batch {i+1} min/max: {batch.min().item()}/{batch.max().item()}")
            
            # Save the first image of the first batch for visual inspection
            if i == 0:
                first_image_tensor = batch[0]
                # Denormalize if needed for saving: (tensor * std) + mean
                # Since normalized to [-1, 1] (mean 0.5, std 0.5): (tensor * 0.5) + 0.5
                first_image_denorm = (first_image_tensor * 0.5) + 0.5
                first_image_pil = transforms.ToPILImage()(first_image_denorm.cpu())
                save_path = "OT-flow/logs/sample_loaded_image.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                first_image_pil.save(save_path)
                print(f"Saved a sample image to {save_path}")
            break # Only inspect the first batch

    except Exception as e:
        print(f"Error during DataLoader test: {e}")
        import traceback
        traceback.print_exc()

# TODO: Implement CelebA Parquet Dataset 