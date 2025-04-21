import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import OrderedDict
import math
from sklearn.cluster import KMeans

# Define the necessary model classes
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        return self.lrelu(self.batch_norm(self.conv(x)))

class Vanilla(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, min_features=32, max_features=32, num_blocks=5, kernel_size=3, padding=0):
        super(Vanilla, self).__init__()
        
        # Calculate number of features for each block
        features = []
        for i in range(num_blocks):
            features.append(min(max_features, min_features * (2 ** i)))
        
        # Features extraction
        layers = []
        layers.append(BasicBlock(in_channels, features[0], kernel_size, 1, padding))
        for i in range(1, num_blocks):
            layers.append(BasicBlock(features[i-1], features[i], kernel_size, 1, padding))
        self.features = nn.Sequential(*layers)
        
        # Features to image
        self.features_to_image = nn.Sequential(
            nn.Conv2d(features[-1], out_channels, kernel_size, 1, padding),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.features_to_image(x)

class MultiVanilla(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, min_features=32, max_features=32, num_blocks=5, kernel_size=3, padding=0):
        super(MultiVanilla, self).__init__()
        self.curr = Vanilla(in_channels, out_channels, min_features, max_features, num_blocks, kernel_size, padding)
        self.prev = None
        
    def forward(self, x, prev=None):
        if prev is None:
            prev = self.prev
        return self.curr(x, prev)

class VanillaDiscriminator(nn.Module):
    def __init__(self, in_channels=3, min_features=32, max_features=32, num_blocks=5, kernel_size=3, padding=0):
        super(VanillaDiscriminator, self).__init__()
        
        # Calculate number of features for each block
        features = []
        for i in range(num_blocks):
            features.append(min(max_features, min_features * (2 ** i)))
        
        # Features extraction
        layers = []
        layers.append(BasicBlock(in_channels, features[0], kernel_size, 1, padding))
        for i in range(1, num_blocks):
            layers.append(BasicBlock(features[i-1], features[i], kernel_size, 1, padding))
        self.features = nn.Sequential(*layers)
        
        # Classifier
        self.classifier = nn.Conv2d(features[-1], 1, kernel_size, 1, padding)
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Function to load and preprocess an image
def load_image(image_path, min_size=25, max_size=250, scale_factor_init=0.75):
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Calculate scales
    min_dim = min(img.size)
    max_dim = max(img.size)
    
    # Calculate number of scales
    num_scales = math.ceil(math.log(min_dim / min_size, 1 / scale_factor_init))
    scale_to_stop = math.ceil(math.log(max_dim / max_size, 1 / scale_factor_init))
    stop_scale = num_scales - scale_to_stop
    
    # Resize image to target size
    target_size = (max_size, int(max_size * img.size[1] / img.size[0]))
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, stop_scale

# Function to generate noise
def generate_noise(ref, device, repeat=False):
    noise = torch.randn(ref.shape, device=device)
    if repeat:
        noise = noise.repeat(1, 1, 1, 1)
    return noise

# Function to quantize colors
def quantize(image, num_colors=8):
    # Convert to numpy array
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    
    # Reshape for k-means
    h, w, c = img_np.shape
    img_reshape = img_np.reshape(-1, c)
    
    # Apply k-means
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    labels = kmeans.fit_predict(img_reshape)
    
    # Get centers
    centers = kmeans.cluster_centers_
    
    # Reconstruct image
    img_quantized = centers[labels].reshape(h, w, c)
    
    # Convert back to tensor
    img_quantized = torch.from_numpy(img_quantized).permute(2, 0, 1).unsqueeze(0)
    img_quantized = img_quantized * 2 - 1  # Normalize back to [-1, 1]
    
    return img_quantized, centers

# Function to quantize using pre-computed centers
def quantize_to_centers(image, centers):
    # Convert to numpy array
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    
    # Reshape for k-means
    h, w, c = img_np.shape
    img_reshape = img_np.reshape(-1, c)
    
    # Find closest center for each pixel
    distances = np.zeros((len(img_reshape), len(centers)))
    for i, center in enumerate(centers):
        distances[:, i] = np.sum((img_reshape - center) ** 2, axis=1)
    labels = np.argmin(distances, axis=1)
    
    # Reconstruct image
    img_quantized = centers[labels].reshape(h, w, c)
    
    # Convert back to tensor
    img_quantized = torch.from_numpy(img_quantized).permute(2, 0, 1).unsqueeze(0)
    img_quantized = img_quantized * 2 - 1  # Normalize back to [-1, 1]
    
    return img_quantized

# Main paint2image function
def paint2image(input_image_path, paint_image_path, model_path, amps_path, 
                paint_start_scale=5, quantization_flag=False, num_colors=8, 
                device='cuda', min_size=25, max_size=250, scale_factor_init=0.75):
    """
    Transform a paint image into a realistic image using a trained SinGAN model.
    
    Args:
        input_image_path: Path to the input image used to train the model
        paint_image_path: Path to the paint image to transform
        model_path: Path to the trained model weights
        amps_path: Path to the trained amplitudes
        paint_start_scale: Scale at which to inject the paint image (default: 5)
        quantization_flag: Whether to perform color quantization (default: False)
        num_colors: Number of colors for quantization (default: 8)
        device: Device to run the model on (default: 'cuda')
        min_size: Minimum size for the smallest scale (default: 25)
        max_size: Maximum size for the largest scale (default: 250)
        scale_factor_init: Scale factor between consecutive scales (default: 0.75)
    
    Returns:
        The generated image as a PIL Image
    """
    # Load input image and calculate scales
    input_tensor, stop_scale = load_image(input_image_path, min_size, max_size, scale_factor_init)
    
    # Load paint image
    paint_img = Image.open(paint_image_path).convert('RGB')
    paint_img = paint_img.resize((input_tensor.shape[3], input_tensor.shape[2]), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    paint_tensor = transform(paint_img).unsqueeze(0).to(device)
    
    # Load model
    model = MultiVanilla().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load amplitudes
    amps = torch.load(amps_path, map_location=device)
    
    # Initialize with noise
    z = generate_noise(input_tensor, device, repeat=False)
    z = z.to(device)
    
    # Generate image
    with torch.no_grad():
        # Process through scales
        for i in range(stop_scale + 1):
            key = f's{i}'
            
            # If this is the paint start scale, inject the paint image
            if i == paint_start_scale:
                if quantization_flag:
                    # Quantize the paint image
                    paint_quantized, centers = quantize(paint_tensor, num_colors)
                    paint_quantized = paint_quantized.to(device)
                    z = paint_quantized
                else:
                    z = paint_tensor
            
            # Generate at this scale
            out = model(z)
            
            # Add noise if not the last scale
            if i < stop_scale:
                noise = generate_noise(out, device, repeat=(i == stop_scale - 1))
                z = out + amps[key].view(-1, 1, 1, 1) * noise
            
            # Upsample for next scale if not the last scale
            if i < stop_scale:
                z = nn.functional.interpolate(z, scale_factor=1/scale_factor_init, mode='bilinear', align_corners=False)
    
    # Convert output to PIL Image
    output = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    output_img = Image.fromarray(output)
    
    return output_img

# Example usage in a notebook cell:
"""
# Example usage
input_image_path = 'path/to/your/input/image.jpg'
paint_image_path = 'path/to/your/paint/image.jpg'
model_path = 'path/to/your/model.pt'
amps_path = 'path/to/your/amps.pt'

# Generate the image
output_img = paint2image(
    input_image_path=input_image_path,
    paint_image_path=paint_image_path,
    model_path=model_path,
    amps_path=amps_path,
    paint_start_scale=5,
    quantization_flag=True
)

# Display the result
from IPython.display import display
display(output_img)

# Save the result
output_img.save('paint2image_result.png')
""" 