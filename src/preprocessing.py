import cv2
import numpy as np 
from pathlib import Path 
from typing import Tuple, Union 

class DocumentPreprocessor:
    def __init__(
        self, 
        target_width: int = 1200,
        blur_kernel: Tuple[int, int] = (5, 5),
        adaptive_block_size: int = 15,
        adaptive_c: int = 8,
    ):
        self.target_width = target_width
        self.blur_kernel = blur_kernel
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        path_str = str(image_path)
        image = cv2.imread(path_str)
        if image is None:
            raise FileNotFoundError(f"Error while loading image : {path_str}")
        return image

    def resize_maintain_aspect(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image in the approrpriate fixed width, cause the model dectection is not 
        performant on small image (< 600 px) and can cause overflow memory on big image (4K > 3000px)
        """
        height, width = image.shape[:2]
        if width == self.target_width:
            return image

        ratio = self.target_width / float(width)
        target_height = int(height * ratio)

        interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(image, (self.target_width, target_height), interpolation=interpolation)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert an image to grayscale if not already"""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise(self, gray_image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray_image, self.blur_kernel, 0)

    def deskew(self, gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
        
        # 1. Invert the image to have white text on a black background
        _, thresh = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 2. Get the (x, y) coordinates of all ink pixels (value > 0)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return gray_image, 0.0

        # 3. Compute the angle from the orientation of the fitted ellipse
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust the angle to be in the range [-45, 45]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # if angle is close to 0, skip
        if abs(angle) < 0.5:
            return gray_image, 0.0

        # 4. Rotate the image
        height, width = gray_image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 5. Apply the rotation
        rotated = cv2.warpAffine(
            gray_image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return rotated, round(float(angle), 2)

    def binarize(self, gray_image: np.ndarray) -> np.ndarray:
        
        return cv2.adaptiveThreshold(
            gray_image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.adaptive_block_size,
            C=self.adaptive_c,
        )

    def run_pipeline(
        self, image_path: Union[str, Path], apply_binarization: bool = True
    ) -> dict:
        
        raw_img = self.load_image(image_path)
        resized_img = self.resize_maintain_aspect(raw_img)
        gray_img = self.to_grayscale(resized_img)
        denoised_img = self.denoise(gray_img)
        deskewed_img, detected_angle = self.deskew(denoised_img)

        final_img = (
            self.binarize(deskewed_img) if apply_binarization else deskewed_img
        )

        return {
            "original_shape": raw_img.shape,
            "processed_shape": final_img.shape,
            "deskew_angle_deg": detected_angle,
            "processed_image": final_img,
            "deskewed_gray": deskewed_img,
        }
    