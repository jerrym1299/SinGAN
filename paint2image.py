import argparse
import torch
import logging
import os
import sys
from PIL import Image
from torchvision import transforms
from generation.utils.core import imresize
from generation.models import *
from generation.trainer import Trainer
from generation.utils.misc import save_image_grid, mkdir, setup_logging
from generation.models.generators import g_multivanilla
import math
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_arguments():
    parser = argparse.ArgumentParser(description='SinGAN: Paint to Image')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Paint')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    parser.add_argument('--paint_start_scale', help='paint injection scale', type=int, required=True)
    parser.add_argument('--quantization_flag', help='specify if to perform color quantization training', type=bool, default=False)
    parser.add_argument('--model_path', help='path to trained model', required=True)
    parser.add_argument('--amps_path', help='path to trained amplitudes', required=True)
    parser.add_argument('--output_dir', help='output directory', default='results/paint2image')
    parser.add_argument('--min_size', default=25, type=int, help='minimum scale size (default: 25)')
    parser.add_argument('--max_size', default=250, type=int, help='maximum scale size (default: 250)')
    parser.add_argument('--scale_factor_init', default=0.75, type=float, help='initialize scaling factor (default: 0.75)')
    parser.add_argument('--min_features', default=32, type=int, help='minimum features (default: 32)')
    parser.add_argument('--max_features', default=32, type=int, help='maximum features (default: 32)')
    parser.add_argument('--num_blocks', default=5, type=int, help='number of blocks (default: 5)')
    parser.add_argument('--kernel_size', default=3, type=int, help='kernel size (default: 3)')
    parser.add_argument('--padding', default=0, type=int, help='padding (default: 0)')
    return parser.parse_args()

def load_image(path, device):
    """Load and preprocess an image"""
    to_tensor = transforms.ToTensor()
    image = Image.open(path).convert('RGB')
    image = to_tensor(image).unsqueeze(0)
    image = (image - 0.5) * 2
    return image.to(device)

def adjust_scales(image, min_size, max_size, scale_factor_init):
    """Calculate scale parameters based on image size"""
    num_scales = math.ceil((math.log(math.pow(min_size / (min(image.size(2), image.size(3))), 1), scale_factor_init))) + 1
    scale_to_stop = math.ceil(math.log(min([max_size, max([image.size(2), image.size(3)])]) / max([image.size(2), image.size(3)]), scale_factor_init))
    stop_scale = num_scales - scale_to_stop

    scale_one = min(max_size / max([image.size(2), image.size(3)]), 1)
    image_resized = imresize(image, scale_one)

    scale_factor = math.pow(min_size/(min(image_resized.size(2), image_resized.size(3))), 1 / (stop_scale))
    scale_to_stop = math.ceil(math.log(min([max_size, max([image_resized.size(2), image_resized.size(3)])]) / max([image_resized.size(2), image_resized.size(3)]), scale_factor_init))
    stop_scale = num_scales - scale_to_stop

    return scale_one, scale_factor, stop_scale

def set_reals(real, scale_factor, stop_scale):
    """Create multi-scale versions of the real image"""
    reals = {}
    for i in range(stop_scale + 1):
        s = math.pow(scale_factor, stop_scale - i)
        reals.update({f's{i}': imresize(real.clone().detach(), s).squeeze(dim=0)})
    return reals

def generate_noise(tensor_like, device, repeat=False):
    """Generate noise tensor"""
    if not repeat:
        noise = torch.randn(tensor_like.size()).to(device)
    else:
        noise = torch.randn((tensor_like.size(0), 1, tensor_like.size(2), tensor_like.size(3)))
        noise = noise.repeat((1, 3, 1, 1)).to(device)
    return noise

def quantize(image):
    """Perform color quantization on the image"""
    # Reshape image to 2D array of pixels
    pixels = image.view(-1, 3)
    
    # Convert to numpy for k-means
    pixels_np = pixels.cpu().numpy()
    
    # Perform k-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=8, random_state=0).fit(pixels_np)
    
    # Get centers and labels
    centers = torch.from_numpy(kmeans.cluster_centers_).to(image.device)
    labels = torch.from_numpy(kmeans.labels_).to(image.device)
    
    # Create quantized image
    quantized = centers[labels].view(image.shape)
    
    return quantized, centers

def quantize_to_centers(image, centers):
    """Quantize image using pre-computed centers"""
    # Reshape image to 2D array of pixels
    pixels = image.view(-1, 3)
    
    # Find nearest center for each pixel
    distances = torch.cdist(pixels, centers)
    labels = torch.argmin(distances, dim=1)
    
    # Create quantized image
    quantized = centers[labels].view(image.shape)
    
    return quantized

def generate_paint2image(g_model, reals, amps, in_s, n, stop_scale, scale_factor, device):
    """Generate image using paint injection"""
    # Initialize output
    out = in_s
    
    # Generate through scales
    for i in range(n, stop_scale + 1):
        key = f's{i}'
        
        # Add noise
        noise = generate_noise(reals[key].unsqueeze(0), device, repeat=(i == n))
        z = out + amps[key].view(-1, 1, 1, 1) * noise
        
        # Generate at this scale
        out = g_model.prev[key](z, out)
        
        # Resize for next scale if not the last one
        if i < stop_scale:
            out = imresize(out, 1 / scale_factor)
            out = out[:, :, :reals[f's{i+1}'].shape[2], :reals[f's{i+1}'].shape[3]]
            
    return out

def main():
    # Parse arguments
    args = get_arguments()
    logger.info("Arguments parsed successfully")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    mkdir(args.output_dir)
    setup_logging(os.path.join(args.output_dir, 'log.txt'))
    logger.info(f"Created output directory: {args.output_dir}")
    
    # Load input image
    input_path = os.path.join(args.input_dir, args.input_name)
    logger.info(f"Loading input image from: {input_path}")
    real = load_image(input_path, device)
    logger.info(f"Input image loaded successfully, shape: {real.shape}")
    
    # Calculate scales
    scale_one, scale_factor, stop_scale = adjust_scales(real, args.min_size, args.max_size, args.scale_factor_init)
    logger.info(f"Calculated scales: scale_one={scale_one}, scale_factor={scale_factor}, stop_scale={stop_scale}")
    
    # Resize input image
    real = imresize(real, scale_one)
    logger.info(f"Resized input image, new shape: {real.shape}")
    
    # Create multi-scale versions
    reals = set_reals(real, scale_factor, stop_scale)
    logger.info(f"Created multi-scale versions, scales: {list(reals.keys())}")
    
    # Load paint image
    paint_path = os.path.join(args.ref_dir, args.ref_name)
    logger.info(f"Loading paint image from: {paint_path}")
    paint_image = load_image(paint_path, device)
    logger.info(f"Paint image loaded successfully, shape: {paint_image.shape}")
    
    # Get the scale to inject paint
    n = args.paint_start_scale
    if n < 1 or n > stop_scale:
        raise ValueError(f"Paint start scale should be between 1 and {stop_scale}")
    logger.info(f"Using paint start scale: {n}")
    
    # Process paint image
    N = stop_scale
    in_s = imresize(paint_image, pow(scale_factor, (N - n + 1)))
    in_s = in_s[:, :, :reals[f's{n-1}'].shape[2], :reals[f's{n-1}'].shape[3]]
    in_s = imresize(in_s, 1 / scale_factor)
    in_s = in_s[:, :, :reals[f's{n}'].shape[2], :reals[f's{n}'].shape[3]]
    logger.info(f"Processed paint image, shape: {in_s.shape}")
    
    # Optional color quantization
    if args.quantization_flag:
        logger.info("Performing color quantization")
        real_s = imresize(real, pow(scale_factor, (N - n)))
        real_s = real_s[:, :, :reals[f's{n}'].shape[2], :reals[f's{n}'].shape[3]]
        real_quant, centers = quantize(real_s)
        in_s = quantize_to_centers(paint_image, centers)
        in_s = imresize(in_s, pow(scale_factor, (N - n)))
        in_s = in_s[:, :, :reals[f's{n}'].shape[2], :reals[f's{n}'].shape[3]]
        logger.info("Color quantization completed")
    
    # Initialize generator model
    model_config = {
        'max_features': args.max_features,
        'min_features': args.min_features,
        'num_blocks': args.num_blocks,
        'kernel_size': args.kernel_size,
        'padding': args.padding
    }
    logger.info("Initializing generator model")
    g_model = g_multivanilla(**model_config)
    g_model.scale_factor = scale_factor
    
    # Add scales
    for scale in range(1, stop_scale + 1):
        g_model.add_scale(device)
    logger.info(f"Added {stop_scale} scales to the model")
    
    # Load trained model
    logger.info(f"Loading model from {args.model_path}")
    g_model.load_state_dict(torch.load(args.model_path, map_location=device))
    amps = torch.load(args.amps_path, map_location=device)
    logger.info("Model and amplitudes loaded successfully")
    
    # Generate output
    logger.info("Generating output image")
    with torch.no_grad():
        out = generate_paint2image(g_model, reals, amps, in_s, n, stop_scale, scale_factor, device)
    
    # Save result
    out = (out + 1.) / 2.
    save_path = os.path.join(args.output_dir, f'paint2image_scale_{n}.png')
    save_image_grid(out.data.cpu(), save_path)
    logger.info(f'Result saved to {save_path}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True) 