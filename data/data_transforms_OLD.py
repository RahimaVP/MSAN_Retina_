import numbers
import random

import numpy as np
from PIL import Image, ImageOps
import torch


# --- Helper Functions for Padding ---

def pad_reflection(image, top, bottom, left, right):
    """
    Pads a NumPy array image with 'reflection' mode (mirroring the edges).
    Recursively handles cases where padding exceeds the image size.
    """
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    
    # Image dimensions
    h, w = image.shape[:2]
    
    # Calculate padding for the next recursive step if padding exceeds boundary
    next_top = next_bottom = next_left = next_right = 0
    if top > h - 1:
        next_top = top - h + 1
        top = h - 1
    if bottom > h - 1:
        next_bottom = bottom - h + 1
        bottom = h - 1
    if left > w - 1:
        next_left = left - w + 1
        left = w - 1
    if right > w - 1:
        next_right = right - w + 1
        right = w - 1
        
    # Create new image canvas
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    
    # Copy original image to the center
    new_image[top:top+h, left:left+w] = image
    
    # Fill padding areas by reflection
    # Top
    new_image[:top, left:left+w] = image[top:0:-1, :]
    # Bottom
    new_image[top+h:, left:left+w] = image[h-1:h-bottom-1:-1, :] # Corrected logic for bottom reflection indices
    
    # Left
    new_image[:, :left] = new_image[:, left*2-1:left-1:-1] # Corrected reflection indices
    
    # Right
    new_image[:, left+w:] = new_image[:, left+w-1:left+w-right-1:-1] # Corrected reflection indices
    
    # Recursively handle remaining large padding
    return pad_reflection(new_image, next_top, next_bottom, next_left, next_right)


def pad_constant(image, top, bottom, left, right, value):
    """
    Pads a NumPy array image with a constant value.
    """
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image.fill(value)
    new_image[top:top+h, left:left+w] = image
    return new_image


def pad_image(mode, image, top, bottom, left, right, value=0):
    """
    Converts a PIL Image to a numpy array, pads it, and converts it back to PIL Image.
    """
    if mode == 'reflection':
        return Image.fromarray(
            pad_reflection(np.asarray(image), top, bottom, left, right))
    elif mode == 'constant':
        return Image.fromarray(
            pad_constant(np.asarray(image), top, bottom, left, right, value))
    else:
        raise ValueError('Unknown mode {}'.format(mode))


# --- Image Transformation Classes ---

class RandomCrop(object):
    """
    Randomly crops an image (and optional corresponding targets) to a given size.
    Pads the image if the image size is smaller than the crop size.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image, *args):

        w, h = image.size
        tw, th = self.size
        top = bottom = left = right = 0
        
        # Check if padding is needed
        if w < tw:
            left = (tw - w) // 2
            right = tw - w - left
        if h < th:
            top = (th - h) // 2
            bottom = th - h - top
            
        # Apply padding if necessary
        if left > 0 or right > 0 or top > 0 or bottom > 0:
            # Note: pad_image returns a PIL Image
            image = pad_image(
                'reflection', image, top, bottom, left, right)
        
        # Recalculate size after padding
        w, h = image.size
        
        # If image size exactly matches target size after padding, return
        if w == tw and h == th:
            return (image, *args)

        # Randomly choose top-left corner (x1, y1) for cropping
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        
        # Crop the image
        results = [image.crop((x1, y1, x1 + tw, y1 + th))]
        
        # Crop all remaining arguments (e.g., masks/labels)
        for arg in args:
             # Assuming args are also PIL images and need to be cropped the same way
             if isinstance(arg, Image.Image):
                 results.append(arg.crop((x1, y1, x1 + tw, y1 + th)))
             else:
                 results.append(arg)
                 
        return results


class DA_MA(object):
    # data augmentation, micro-adjust (e.g., brightness/contrast jitter)
    def __init__(self, mean_1=torch.tensor(1.0).float(), mean_2=torch.tensor(0).float(), std=torch.tensor(0.2).float()):
        self.mean_1 = mean_1
        self.mean_2 = mean_2
        self.std = std
        
    def __call__(self, image, *args):
        # Convert PIL image to NumPy array for numerical operation
        image_np = np.array(image).astype('float32') # Use float32 for precision
        
        # Generate random scaling (k) and shifting (b) factors
        # k (multiplicative factor) is around mean_1 (default 1.0)
        k = torch.normal(self.mean_1, self.std).item()
        # b (additive factor) is around mean_2 (default 0)
        b = torch.normal(self.mean_2, self.std).item()
        
        # Apply micro-adjustment: image = k * image + b
        image_np = k * image_np + b
        
        # Clip values to the valid range [0, 255]
        # The original code clips to [0, 255] then converts to uint8, but the implementation only sets values outside [0, 255] to 0, which is incorrect clipping.
        # Standard practice is np.clip(image_np, 0, 255)
        image_np[image_np > 255] = 255
        image_np[image_np < 0] = 0
        
        # Convert back to uint8 and PIL Image
        image_np = image_np.astype('uint8')
        image = Image.fromarray(image_np)
        
        # Return image and any other passed arguments
        return (image, *args)


class TrainMask(object):
    """
    Crops a region defined by self.size=(Y_min, Y_max, X_min, X_max) and 
    then resizes the cropped image to the cropped size.
    (The resize step to the same size is redundant based on the provided code logic,
    but the crop itself is the key operation).
    """
    def __init__(self, size=(16, 240, 16, 240)):
        assert len(size) == 4
        # size is (Y_min, Y_max, X_min, X_max)
        self.size = size 

    def __call__(self, image, *args):
        # Get coordinates
        Y_min, Y_max = self.size[0], self.size[1]
        X_min, X_max = self.size[2], self.size[3]
        
        # Calculate target size (width, height)
        in_size = (X_max - X_min, Y_max - Y_min)
        
        # Crop the image (PIL uses (left, top, right, bottom))
        img_mask = image.crop((X_min, Y_min, X_max, Y_max))
        
        # Resize the cropped image to the size of the cropped region
        # This operation is redundant if the crop is a standard PIL crop
        img_mask = img_mask.resize(in_size) 
        
        # The original code only returns the mask and discards *args, 
        # which is inconsistent with other transforms. 
        # Assuming the intent was to apply the transform to the primary image:
        # return img_mask, *args 
        return img_mask


class TestMask(object):
    """
    Identical to TrainMask. Crops a region and resizes it.
    """
    def __init__(self, size=(16, 240, 16, 240)):
        assert len(size) == 4
        self.size = size

    def __call__(self, image, *args):
        Y_min, Y_max = self.size[0], self.size[1]
        X_min, X_max = self.size[2], self.size[3]
        in_size = (X_max - X_min, Y_max - Y_min)
        
        # Crop using (left, top, right, bottom)
        img_mask = image.crop((X_min, Y_min, X_max, Y_max))
        img_mask = img_mask.resize(in_size)
        
        return img_mask


class Resize(object):
    """
    Resize the input PIL Image to the given size.
    Handles both fixed (h, w) size and single int size (scaling the shorter edge).
    Note: The implementation provided below seems to mostly enforce a square output.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):

        # If size is an integer (intended for scaling shorter edge)
        if isinstance(self.size, int):
            # The following logic forces a square output of size x size
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return [img]
            
            # If w < h, scale both to self.size
            if w < h:
                ow = self.size
                oh = self.size
            # If h < w, scale both to self.size
            else:
                oh = self.size
                ow = self.size
            
            # The commented-out lines below show the standard logic for proportional scaling:
            # if w < h: oh = int(self.size * h / w)
            # else: ow = int(self.size * w / h)
            
            return [img.resize((ow, oh), self.interpolation)]
        
        # If size is a sequence (e.g., (H, W))
        else:
            # PIL resize expects (width, height), so self.size[::-1] converts (H, W) to (W, H)
            return [img.resize(self.size[::-1], self.interpolation)]


class RandomScale(object):
    def __init__(self, scale):
        if isinstance(scale, numbers.Number):
            # scale can be a single number (e.g., 1.2 for [1/1.2, 1.2]) or a list [min, max]
            scale = [1 / scale, scale]
        self.scale = scale

    def __call__(self, image):
        ratio = random.uniform(self.scale[0], self.scale[1])
        w, h = image.size
        tw = int(ratio * w)
        th = int(ratio * h)
        
        # If no scaling, return original image
        if ratio == 1:
            return image,
        
        # Choose interpolation method based on scaling direction
        elif ratio < 1:
            interpolation = Image.Resampling.LANCZOS # Image.ANTIALIAS is deprecated
        else:
            interpolation = Image.Resampling.BICUBIC # Image.CUBIC is deprecated
            
        # Resize and return as a tuple (consistent with other transforms)
        return image.resize((tw, th), interpolation),


class RandomRotate(object):
    """
    Rotates the image by a random angle between [-angle, angle].
    Pads the image using reflection to ensure no content is lost during rotation.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, *args):

        w, h = image.size
        
        # Get random angle
        angle = random.randint(0, self.angle * 2) - self.angle

        # Pad the image by its own height/width using reflection to prepare for rotation.
        # This creates a canvas large enough (3x width, 3x height) to avoid cropping the rotated image.
        image = pad_image('reflection', image, h, h, w, w)
        
        # Rotate the padded image
        image = image.rotate(angle, resample=Image.BILINEAR)
        
        # Crop the image back to the original size (centered within the padded area)
        image = image.crop((w, h, w + w, h + h))
        
        # Return image and any other passed arguments
        return (image, *args)


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, image, *args):
        if random.random() < 0.5:
            # Flip the image
            results = [image.transpose(Image.FLIP_LEFT_RIGHT)]
        else:
            # Keep the image as is
            results = [image]
            
        # Append all other arguments (e.g., masks) if they were passed
        results.extend(args)
        return results


class Normalize(object):
    """
    Normalizes a torch.Tensor image (C x H x W) with a given mean and std deviation.
    """

    def __init__(self, mean, std):
        # Convert list/tuple to torch.FloatTensor
        self.mean = torch.FloatTensor(mean).view(-1, 1, 1) # Reshape to (C, 1, 1) for broadcasting
        self.std = torch.FloatTensor(std).view(-1, 1, 1)

    def __call__(self, image, label=None):
        
        # Perform normalization using broadcasting (t - m) / s
        image.sub_(self.mean).div_(self.std)
        
        if label is None:
            return image,
        else:
            return image, label


class Pad(object):
    """
    Pads the given PIL.Image on all sides by 'padding' size.
    Uses 'reflection' (fill=-1) or 'constant' (fill=value).
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
               isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, image, *args):

        if self.fill == -1:
            # Reflection padding
            image = pad_image(
                'reflection', image,
                self.padding, self.padding, self.padding, self.padding)
        else:
            # Constant padding
            image = pad_image(
                'constant', image,
                self.padding, self.padding, self.padding, self.padding,
                value=self.fill)
        return (image, *args)


class PadImage(object):
    """
    Pads the given PIL.Image using PIL's ImageOps.expand for constant padding
    or custom reflection padding for fill=-1.
    """
    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
               isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, image, *args):
        if self.fill == -1:
            # Use custom reflection padding
            image = pad_image(
                'reflection', image,
                self.padding, self.padding, self.padding, self.padding)
        else:
            # Use PIL's built-in expansion (constant padding)
            image = ImageOps.expand(image, border=self.padding, fill=self.fill)
        return (image, *args)


class ToTensor(object):
    """
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    Also handles converting an optional PIL label image to a torch.LongTensor.
    """

    def __call__(self, pic, label=None):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            # Assuming numpy array is already H x W x C (or H x W) and float/int
            img = torch.from_numpy(pic).float()
            if img.dim() == 3:
                # Convert H x W x C to C x H x W
                img = img.permute(2, 0, 1).contiguous()
            elif img.dim() == 2:
                # Add channel dimension for H x W to 1 x H x W
                img = img.unsqueeze(0)
            
        else:
            # handle PIL Image
            # Load bytes, infer dimensions, and permute HWC to CHW
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            
            # Infer number of channels
            if pic.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic.mode)
                
            img = img.view(pic.size[1], pic.size[0], nchannel)
            
            # Put it from HWC to CHW format: (H, W, C) -> (C, H, W)
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            
        # Convert to float and normalize to [0.0, 1.0]
        img = img.float().div(255)
        
        # Handle label conversion
        if label is None:
            return img,
        else:
            # Convert label (PIL Image or similar) to LongTensor
            return img, torch.LongTensor(np.array(label, dtype=np.int64))


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        # Pass the results of one transform as the input to the next
        for t in self.transforms:
            args = t(*args)
        return args