import os
import math
from PIL import Image

# Default parameters
min_size = 25
max_size = 250
scale_factor_init = 0.75

# Load the image
image_path = os.path.join('images', 'singan_vsmal.jpg')
image = Image.open(image_path)
width, height = image.size

print(f"Image dimensions: {width}x{height}")

# Calculate number of scales
min_dim = min(width, height)
max_dim = max(width, height)

# Calculate num_scales
num_scales = math.ceil((math.log(math.pow(min_size / min_dim, 1), scale_factor_init))) + 1

# Calculate scale_to_stop
scale_to_stop = math.ceil(math.log(min([max_size, max_dim]) / max_dim, scale_factor_init))

# Calculate stop_scale
stop_scale = num_scales - scale_to_stop

print(f"Number of scales: {stop_scale + 1} (from scale 0 to scale {stop_scale})")

# Calculate approximate dimensions for each scale
print("\nApproximate dimensions for each scale:")
for i in range(stop_scale + 1):
    scale_factor = math.pow(scale_factor_init, stop_scale - i)
    scale_width = int(width * scale_factor)
    scale_height = int(height * scale_factor)
    print(f"Scale {i}: {scale_width}x{scale_height} pixels") 