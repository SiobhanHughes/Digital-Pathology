# Classes for image processing pipelines to count macrophages
# Pipeline 1: Segmentation of the red stain channel
# Pipeline 2: Segmentation of the nuclei of red stained cells

# Slides stained as follows:
# (h) Haematoxylin - nuclei
# (d) DAB (brown) - PDL-1
# (r) Red stain - CD163 positive macrophages (Alkaline Phosphatase)
# hdr image


import numpy as np
import skimage as sk
from skimage import data, io, segmentation, morphology, feature, future, filters, util
from skimage.measure import label, regionprops
from skimage.color import label2rgb, separate_stains
from scipy import ndimage as ndi
import pandas as pd
import math


class Deconvolute:
    """
    Class to carry out colour deconvolution of a RGB image
    """
    def get_OD_vector(a):
        """
        Function that takes RGB values as a list.
        Returns normalised OD vector.
        """
        a = np.array([a], dtype=np.uint8)
        a_float = sk.img_as_float(a)
        OD = -np.log10(a_float)
        normalised_OD = OD / np.sqrt(np.sum(OD**2))
        OD_vector = np.around(normalised_OD, decimals=3)
        return OD_vector[0]

    def deconvolute(slide):
        """ Function that takes a hdr image
        and performs stain separation"""
        # Normalised OD vector matrix:   R      G      B
        rgb_from_hdr = np.array([[0.651, 0.701, 0.29],   # (h) Haematoxylin OD vector - channel [0]
                                 [ 0.269, 0.568, 0.778], # (d) DAB OD vector - channel [1]
                                 [0.185, 0.78, 0.598]])  # (r) Red stain OD vector - channel [2]

        # Colour deconvolution matrix (inverse of OD vector matrix)
        hdr_from_rgb = np.linalg.inv(rgb_from_hdr)
        deconvolute = separate_stains(slide, hdr_from_rgb)
        return deconvolute

    
class Red_stain_channel:
    """
    Class for processing the red stain channel - CD163 positive macrophages
    """
    
    def __init__(self, red_channel):
        self.red = red_channel
    
    def smooth_red(self, image, red_filter='median', red_filter_size=3, red_filter_sigma=1):
        # 1. Smoothing filter for red stain channel
        if red_filter == 'gaussian':
            red_smooth = ndi.gaussian_filter(util.img_as_float(image), sigma=red_filter_sigma)
        else:
            red_smooth = ndi.median_filter(util.img_as_float(image), size=red_filter_size)
        return red_smooth
    
    def fill_nuclei(self, red_smooth):
        # 2. Fill in nuclei on red stain channel by erosion
        seed = np.copy(red_smooth)
        seed[1:-1, 1:-1] = red_smooth.max()
        mask = red_smooth
        filled = morphology.reconstruction(seed, mask, method='erosion')
        return filled
    
    def red_mask(self, filled, red_small_objects=100):
        #3. Threshold for red stain channel
        red_threshold = filters.threshold_otsu(filled)
        red_mask = filled > red_threshold
    
        #4. remove small objects - clean up mask (default value for min_size is 64)
        red_mask_rm = morphology.remove_small_objects(red_mask, min_size=red_small_objects)
        return red_mask_rm
    
    
class Segment:
    """
    Class for image segmentation using the watershed algorithm
    """
     
    # Distance transform of binary image (mask) and markers for watershed segmentation
    def markers_red(self, mask, peak_min_distance=15):
        distance = ndi.distance_transform_edt(mask) 
        local_peak = feature.peak_local_max(distance, min_distance=peak_min_distance, indices=False)
        markers_peak = ndi.label(local_peak)[0]
        return markers_peak, distance
    
    # Watershed segmentation using denoised red stain channel (with or without nuclei filled)
    def segment_red(self, image, markers_peak, mask): # image can be red_smooth or filled
        labels_red_channel = segmentation.watershed(image, markers_peak, mask=mask, connectivity=2)
        return labels_red_channel
    
    # Watershed segmentation using the distance transform
    def segment_dist(self, distance, markers_peak, mask): # input image is distance transform
        labels_distance = segmentation.watershed(-distance, markers_peak, mask=mask, connectivity=2)
        return labels_distance
    
    # Watershed segmentation using the gradient of the image
    def segment_gradient(self, image, markers_peak, mask): # image can be red_smooth or filled
        gradient = filters.rank.gradient(util.img_as_ubyte(image), morphology.disk(2))
        labels_gradient = segmentation.watershed(gradient, markers_peak, mask=mask, connectivity=2)
        return labels_gradient
