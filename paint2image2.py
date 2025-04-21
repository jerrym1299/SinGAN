import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from generation.models.generators import g_multivanilla
from generation.models.discriminators import d_vanilla
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Paint')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    parser.add_argument('--paint_start_scale', help='paint injection scale', type=int, required=True)
    parser.add_argument('--model_dir', help='directory containing trained models', required=True)
    parser.add_argument('--results_dir', help='results directory', default='results')
    parser.add_argument('--quantization_flag', help='specify if to perform color quantization training', type=bool, default=False)
    parser.add_argument('--device', help='device to use', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_image(image_path, device):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image

def quantize_colors(image, num_colors=8):
    """Quantize colors using k-means clustering."""
    from sklearn.cluster import KMeans
    
    # Reshape image to 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the quantized colors
    quantized = kmeans.cluster_centers_.astype(np.uint8)
    
    # Replace each pixel with its nearest cluster center
    labels = kmeans.labels_
    quantized_image = quantized[labels].reshape(image.shape)
    
    return quantized_image

def load_trained_models(model_dir, device):
    """Load trained generator and discriminator models with correct feature dimensions."""
    # Load generator
    g_path = os.path.join(model_dir, 'g_multivanilla.pt')
    g_state_dict = torch.load(g_path, map_location=device)
    
    # Determine feature dimensions from the smallest scale in the state dict
    base_feature_dim = None
    for key, value in g_state_dict.items():
        if 'prev.s0.features.0.conv.weight' in key:
            base_feature_dim = value.shape[0]  # Output channels of first layer in scale 0
            print(f"Detected generator base feature dimension: {base_feature_dim}")
            break
    
    if base_feature_dim is None:
        base_feature_dim = 32  # Default fallback
        print(f"Using default generator base feature dimension: {base_feature_dim}")
    
    # Count the number of scales
    scales = [int(key.split('.')[1][1:]) for key in g_state_dict.keys() if key.startswith('prev.s')]
    num_scales = max(scales) + 2 if scales else 1
    print(f"Detected {num_scales} scales in the model")
    
    # Create generator with correct base feature dimensions
    print(f"Creating generator with base features={base_feature_dim}")
    generator = g_multivanilla(
        in_channels=3, 
        min_features=base_feature_dim,
        max_features=base_feature_dim,
        num_blocks=5, 
        kernel_size=3, 
        padding=0
    )
    generator = generator.to(device)
    
    # Add scales to match the checkpoint
    print(f"Adding {num_scales-1} scales to the generator")
    for scale in range(num_scales - 1):
        print(f"Adding scale {scale+1}")
        generator.add_scale(device)
    
    # Load generator state dict
    try:
        generator.load_state_dict(g_state_dict)
        print("Successfully loaded generator state dict")
    except Exception as e:
        print(f"Error loading generator state dict: {e}")
        print("Trying to load with strict=False...")
        try:
            generator.load_state_dict(g_state_dict, strict=False)
            print("Successfully loaded generator state dict with strict=False")
        except Exception as e2:
            print(f"Error loading with strict=False: {e2}")
    
    generator.eval()
    
    # Add num_scales attribute to generator
    generator.num_scales = num_scales
    
    # Load discriminator
    d_path = os.path.join(model_dir, 'd_vanilla.pt')
    if os.path.exists(d_path):
        d_state_dict = torch.load(d_path, map_location=device)
        
        # Determine discriminator feature dimensions
        d_base_feature_dim = None
        for key, value in d_state_dict.items():
            if 'features.0.conv.weight' in key:
                d_base_feature_dim = value.shape[0]  # Output channels of first layer
                print(f"Detected discriminator base feature dimension: {d_base_feature_dim}")
                break
                
        if d_base_feature_dim is None:
            d_base_feature_dim = base_feature_dim  # Use same as generator as fallback
            print(f"Using generator's base feature dimension for discriminator: {d_base_feature_dim}")
        
        # Create discriminator with correct dimensions
        print(f"Creating discriminator with base features={d_base_feature_dim}")
        discriminator = d_vanilla(
            in_channels=3, 
            min_features=d_base_feature_dim,
            max_features=d_base_feature_dim, 
            num_blocks=5, 
            kernel_size=3, 
            padding=0
        )
        discriminator = discriminator.to(device)
        
        # Load discriminator state dict
        try:
            discriminator.load_state_dict(d_state_dict)
            print("Successfully loaded discriminator state dict")
        except Exception as e:
            print(f"Error loading discriminator state dict: {e}")
            try:
                discriminator.load_state_dict(d_state_dict, strict=False)
                print("Successfully loaded discriminator state dict with strict=False")
            except Exception as e2:
                print(f"Error loading discriminator with strict=False: {e2}")
    else:
        # No discriminator checkpoint found, use generator's dimensions
        print(f"No discriminator checkpoint found. Creating with generator's dimensions: {base_feature_dim}")
        discriminator = d_vanilla(
            in_channels=3, 
            min_features=base_feature_dim,
            max_features=base_feature_dim, 
            num_blocks=5, 
            kernel_size=3, 
            padding=0
        )
        discriminator = discriminator.to(device)
    
    discriminator.eval()
    
    return generator, discriminator
def generate_paint2image(generator, input_image, paint_image, paint_start_scale, device):
    """Generate image using paint2image functionality."""
    with torch.no_grad():
        # Initialize with input image
        current_image = input_image
        
        # Create reals dictionary for each scale
        reals = {}
        amps = {}
        noises = {}
        
        # Process through scales
        for scale in range(paint_start_scale, generator.num_scales):
            # Generate noise
            noise = torch.randn_like(current_image)
            
            # If this is the paint injection scale, use paint image
            if scale == paint_start_scale:
                current_image = paint_image
            
            # Store current image and noise for this scale
            reals[f's{scale}'] = current_image
            noises[f's{scale}'] = noise
            amps[f's{scale}'] = torch.tensor(0.1).to(device)  # Default amplitude
            
            # Upsample for next scale if not the last scale
            if scale < generator.num_scales - 1:
                current_image = nn.functional.interpolate(
                    current_image,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False
                )
        
        # Generate image using the MultiVanilla generator
        output_image = generator(reals, amps, noises)
        
        return output_image

def main():
    args = get_arguments()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load models
    generator, discriminator = load_trained_models(args.model_dir, args.device)
    
    # Load input and paint images
    input_path = os.path.join(args.input_dir, args.input_name)
    paint_path = os.path.join(args.ref_dir, args.ref_name)
    
    input_image = load_image(input_path, args.device)
    paint_image = load_image(paint_path, args.device)
    
    # Verify paint_start_scale is valid
    if args.paint_start_scale < 0 or args.paint_start_scale >= generator.num_scales:
        raise ValueError(f"paint_start_scale must be between 0 and {generator.num_scales-1}")
    
    # Perform color quantization if requested
    if args.quantization_flag:
        paint_image = quantize_colors(paint_image.cpu().numpy())
        paint_image = torch.from_numpy(paint_image).float().to(args.device)
    
    # Generate image
    output_image = generate_paint2image(
        generator,
        input_image,
        paint_image,
        args.paint_start_scale,
        args.device
    )
    
    # Save result
    output_path = os.path.join(
        args.results_dir,
        f"paint2image_{args.input_name.split('.')[0]}_scale_{args.paint_start_scale}.png"
    )
    
    # Convert from normalized range [-1, 1] to [0, 1]
    output_image = (output_image + 1) / 2
    
    # Convert to PIL Image and save
    output_image = output_image.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(output_image)
    output_image.save(output_path)
    
    print(f"Generated image saved to: {output_path}")

if __name__ == '__main__':
    main() 