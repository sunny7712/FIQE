import cv2
import numpy as np
from typing import Tuple, Union
import random

class DegradationModels:
    """Implementation of image degradation models for simulating low quality images."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize degradation models.
        
        Args:
            target_size: Target size for resizing input images (default: 256x256)
        """
        self.target_size = target_size
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, self.target_size)
    
    def _apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 0, sigma: float = 0) -> np.ndarray:
        """Apply Gaussian blur to image."""
        # Ensure kernel size is odd
        if kernel_size == 0:
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def _add_gaussian_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def _jpeg_compression(self, image: np.ndarray, quality: int) -> np.ndarray:
        """Apply JPEG compression."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    def apply_degradation(self, image: np.ndarray, 
                         mode: str = 'BicC',
                         scale_factor: int = 8) -> np.ndarray:
        """
        Apply degradation to input image.
        
        Args:
            image: Input high-quality image
            mode: Degradation mode ('BicC', 'BicN', or 'BBilN')
            scale_factor: Downsampling scale factor
        
        Returns:
            Degraded image
        """
        # First resize to target size
        image = self._resize_image(image)
        
        if mode == 'BicC':  # Mild degradation
            # Downsample using bicubic interpolation
            h, w = image.shape[:2]
            downsampled = cv2.resize(image, (w//scale_factor, h//scale_factor), 
                                   interpolation=cv2.INTER_CUBIC)
            upsampled = cv2.resize(downsampled, (w, h), 
                                 interpolation=cv2.INTER_CUBIC)
            # Apply JPEG compression with random quality factor
            quality = random.randint(90, 95)
            return self._jpeg_compression(upsampled, quality)
            
        elif mode == 'BicN':  # Moderate degradation
            # Downsample using bicubic interpolation
            h, w = image.shape[:2]
            downsampled = cv2.resize(image, (w//scale_factor, h//scale_factor), 
                                   interpolation=cv2.INTER_CUBIC)
            upsampled = cv2.resize(downsampled, (w, h), 
                                 interpolation=cv2.INTER_CUBIC)
            # Add Gaussian noise
            noise_std = random.uniform(20, 25)
            return self._add_gaussian_noise(upsampled, noise_std)
            
        elif mode == 'BBilN':  # Severe degradation
            # Apply Gaussian blur
            sigma = random.uniform(5, 7)
            blurred = self._apply_gaussian_blur(image, sigma=sigma)
            
            # Downsample using bilinear interpolation
            h, w = blurred.shape[:2]
            downsampled = cv2.resize(blurred, (w//scale_factor, h//scale_factor), 
                                   interpolation=cv2.INTER_LINEAR)
            upsampled = cv2.resize(downsampled, (w, h), 
                                 interpolation=cv2.INTER_LINEAR)
            
            # Add Gaussian noise
            noise_std = random.uniform(10, 15)
            return self._add_gaussian_noise(upsampled, noise_std)
            
        else:
            raise ValueError(f"Unknown degradation mode: {mode}")