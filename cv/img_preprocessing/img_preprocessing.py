import cv2
import numpy as np


def resize(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized_image


def apply_clahe(img, adaptive_contrast=None):
    if img.shape[-1] != 3:
        raise Exception('Method available only for RGB images')

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_val, a_val, b_val = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_val)

    if adaptive_contrast:
        l_hist_eq = cv2.equalizeHist(l_val)
        l_mask = l_val > adaptive_contrast
        l_clahe = np.where(l_mask, l_hist_eq, l_clahe)

    lab = cv2.merge((l_clahe, a_val, b_val))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result


def apply_gaussian_blur(img, filter_size):
    return cv2.GaussianBlur(img, filter_size, 0)


def apply_median_blur(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)
