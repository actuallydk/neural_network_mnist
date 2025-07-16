import numpy as np
from PIL import Image
import io
import base64
from scipy.ndimage import center_of_mass, shift

def preprocess_image(data):
    # Remove header if present
    if ',' in data:
        data = data.split(',')[1]
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    arr = np.array(img)

    # Binarize
    arr = np.where(arr < 128, 0, 255).astype(np.uint8)

    # Find bounding box
    coords = np.column_stack(np.where(arr < 255))
    if coords.size:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        arr = arr[y0:y1+1, x0:x1+1]

    # If digit is much taller than wide, pad sides before resizing
    h, w = arr.shape
    if h > w:
        pad = (h - w) // 2
        arr = np.pad(arr, ((0, 0), (pad, h - w - pad)), constant_values=255)
    elif w > h:
        pad = (w - h) // 2
        arr = np.pad(arr, ((pad, w - h - pad), (0, 0)), constant_values=255)

    # Convert back to PIL for resizing and padding
    img = Image.fromarray(arr)
    img = img.resize((20, 20), Image.Resampling.LANCZOS)
    new_img = Image.new('L', (28, 28), 255)
    upper_left = ((28 - 20) // 2, (28 - 20) // 2)
    new_img.paste(img, upper_left)
    arr = np.array(new_img)

    # Invert and normalize
    arr = 1.0 - arr / 255.0

    # Only center if there are non-background pixels
    if np.any(arr < 1.0):
        with np.errstate(invalid='ignore'):
            com = center_of_mass(arr.reshape(28, 28).astype(float))
        if (
            isinstance(com, tuple)
            and len(com) == 2
            and all(isinstance(v, (float, int)) and not np.isnan(v) for v in com)
        ):
            cy = float(com[0])
            cx = float(com[1])
            shift_y = int(np.round(14 - cy))
            shift_x = int(np.round(14 - cx))
            arr_shifted = shift(arr.reshape(28, 28), shift=(shift_y, shift_x), order=1, mode='constant', cval=0.0)
            arr = arr_shifted.reshape(1, -1)
        else:
            arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(1, -1)
    return arr 