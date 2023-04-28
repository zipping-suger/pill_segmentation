from utils import batch_resize

raw_image_path = './dataset/pills'
output_path = './dataset/pills_resized_256_192'

batch_resize(256, 192, raw_image_path, output_path)
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
