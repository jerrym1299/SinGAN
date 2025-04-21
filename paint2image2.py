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
    parser.add_argument('--input_dir', default='Input/Images')
    parser.add_argument('--input_name', required=True)
    parser.add_argument('--ref_dir', default='Input/Paint')
    parser.add_argument('--ref_name', required=True)
    parser.add_argument('--paint_start_scale', type=int, required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--quantization_flag', type=bool, default=False)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--stop_scale', type=int, default=None, help='Optional maximum scale to use during generation')
    return parser.parse_args()

def load_image(image_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image

def quantize_colors(image, num_colors=8):
    from sklearn.cluster import KMeans
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    quantized = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_
    quantized_image = quantized[labels].reshape(image.shape)
    return quantized_image

def load_trained_models(model_dir, device):
    g_path = os.path.join(model_dir, 'g_multivanilla.pt')
    g_state_dict = torch.load(g_path, map_location=device)
    
    base_feature_dim = None
    for key, value in g_state_dict.items():
        if 'prev.s0.features.0.conv.weight' in key:
            base_feature_dim = value.shape[0]
            print(f"Detected generator base feature dimension: {base_feature_dim}")
            break
    if base_feature_dim is None:
        base_feature_dim = 32
        print(f"Using default generator base feature dimension: {base_feature_dim}")
    
    scales = [int(key.split('.')[1][1:]) for key in g_state_dict.keys() if key.startswith('prev.s')]
    num_scales = max(scales) + 2 if scales else 1
    print(f"Detected {num_scales} scales in the model")

    print(f"Creating generator with base features={base_feature_dim}")
    generator = g_multivanilla(3, base_feature_dim, base_feature_dim, 5, 3, 0).to(device)

    print(f"Adding {num_scales-1} scales to the generator")
    for scale in range(num_scales - 1):
        print(f"Adding scale {scale+1}")
        generator.add_scale(device)

    try:
        generator.load_state_dict(g_state_dict)
        print("Successfully loaded generator state dict")
    except Exception as e:
        print(f"Error loading generator state dict: {e}")
        try:
            generator.load_state_dict(g_state_dict, strict=False)
            print("Successfully loaded generator state dict with strict=False")
        except Exception as e2:
            print(f"Error loading with strict=False: {e2}")

    generator.eval()
    generator.num_scales = num_scales

    d_path = os.path.join(model_dir, 'd_vanilla.pt')
    if os.path.exists(d_path):
        d_state_dict = torch.load(d_path, map_location=device)
        d_base_feature_dim = None
        for key, value in d_state_dict.items():
            if 'features.0.conv.weight' in key:
                d_base_feature_dim = value.shape[0]
                print(f"Detected discriminator base feature dimension: {d_base_feature_dim}")
                break
        if d_base_feature_dim is None:
            d_base_feature_dim = base_feature_dim
            print(f"Using generator's base feature dimension for discriminator: {d_base_feature_dim}")
        print(f"Creating discriminator with base features={d_base_feature_dim}")
        discriminator = d_vanilla(3, d_base_feature_dim, d_base_feature_dim, 5, 3, 0).to(device)
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
        print(f"No discriminator checkpoint found. Creating with generator's dimensions: {base_feature_dim}")
        discriminator = d_vanilla(3, base_feature_dim, base_feature_dim, 5, 3, 0).to(device)

    discriminator.eval()
    return generator, discriminator

def generate_paint2image(generator, input_image, paint_image, paint_start_scale, device, stop_scale=None):
    with torch.no_grad():
        current_image = input_image
        reals, amps, noises = {}, {}, {}

        max_scale = stop_scale if stop_scale is not None else generator.num_scales

        for scale in range(paint_start_scale, max_scale):
            torch.cuda.empty_cache()  # Clear memory between scales
            noise = torch.randn_like(current_image)

            if scale == paint_start_scale:
                current_image = paint_image

            reals[f's{scale}'] = current_image
            noises[f's{scale}'] = noise
            amps[f's{scale}'] = torch.tensor(0.1).to(device)

            if scale < max_scale - 1:
                current_image = nn.functional.interpolate(
                    current_image,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False
                )

        output_image = generator(reals, amps, noises)
        return output_image

def main():
    args = get_arguments()
    os.makedirs(args.results_dir, exist_ok=True)
    
    generator, discriminator = load_trained_models(args.model_dir, args.device)
    
    input_path = os.path.join(args.input_dir, args.input_name)
    paint_path = os.path.join(args.ref_dir, args.ref_name)
    input_image = load_image(input_path, args.device)
    paint_image = load_image(paint_path, args.device)

    if args.paint_start_scale < 0 or args.paint_start_scale >= generator.num_scales:
        raise ValueError(f"paint_start_scale must be between 0 and {generator.num_scales-1}")
    
    if args.quantization_flag:
        paint_image = quantize_colors(paint_image.cpu().numpy())
        paint_image = torch.from_numpy(paint_image).float().to(args.device)

    output_image = generate_paint2image(
        generator,
        input_image,
        paint_image,
        args.paint_start_scale,
        args.device,
        args.stop_scale
    )

    output_path = os.path.join(
        args.results_dir,
        f"paint2image_{args.input_name.split('.')[0]}_scale_{args.paint_start_scale}.png"
    )

    output_image = (output_image + 1) / 2
    output_image = output_image.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(output_image)
    output_image.save(output_path)
    print(f"Generated image saved to: {output_path}")

if __name__ == '__main__':
    main()
