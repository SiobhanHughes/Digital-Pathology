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

    def deconvolute_hdr(slide):
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


class Count_cells:
    """
    Class to count cells (CD163 positive macropahges (red stain channel) for these pipelines),
    including benchmark analysis to output precision, recall and F1 scores when compared to 
    annotated ground truth.
    
    Initialise class with labels (output) from watershed segmentation.
    Ground truth default is none. 
    If ground truth available, it is initialised as a numpy array of the x,y coordinates of the centre of positive cells. 
    """
    
    def __init__(self, labels, ground_truth=None):
        self.labels = labels
        self.ground_truth = ground_truth
        
    def count_regions(self):
        # count number of segmented regions - initial count for cells 
        # (before merging centroids to account for over-segmentation)
        count = 0
        for region in regionprops(self.labels):
            count += 1
        return count
    
    def get_centroids(self):
        # Get centroids of regions from segmentation
        centroids = []
        for region in regionprops(self.labels):
            centroids.append(region.centroid)
        # convert to numpy array of x,y coordinates
        rows = len(centroids)
        centroids_array = np.zeros(shape=(rows, 2))
        for i in range(len(centroids)):
            centroids_array[i][1] = centroids[i][0] # x coordinate  (column)
            centroids_array[i][0] = centroids[i][1] # y coordinate (row)
        return centroids_array
    
    @staticmethod
    def distance(x1, x2, y1, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) # euclidean distance
    
    def merge_centroids(self, centroids_array):
        # reduce over-segmentation by keeping only one of a pair of centroids that mark the same cell or nucleus
        # return final set of centroids marking each cell/nucleus as well as the final count for number of cells
        num = 25 #within 25 pixels distance - set distance for merging pairs of centroids
        still_to_merge = 1000
        while still_to_merge !=0 :
            dist_centroids = []
            # calculate euclidian distance between all pairs of centroids
            for i in range(len(centroids_array)):
                for j in range((i+1), len(centroids_array)): # i+1 to len (upper triangle of matrix)
                    dist = self.distance(centroids_array[i][0], centroids_array[j][0],
                                         centroids_array[i][1], centroids_array[j][1])
                    row = [centroids_array[i][0], centroids_array[i][1], centroids_array[j][0],
                           centroids_array[j][1], dist]
                    dist_centroids.append(row)
            
            # Get centroids to merge - distance bewteen centroids is less than or equal to num
            to_merge = []
            for row in dist_centroids:
                if row[4] <= num:
                    to_merge.append(row)
    
            still_to_merge = len(to_merge) # terminates while loop

            # keep only one of each pair to be merged
            remove = []
            for row in to_merge:
                if [row[0], row[1]] not in remove:
                    remove.append([row[0], row[1]])
        
            centroids = centroids_array.tolist()
            # remove chosen centroids
            for row in centroids:
                if row in remove:
                    centroids.remove(row)
            # update centroids array
            centroids_array = np.array(centroids)
        
        count_macrophages = len(centroids_array)
        return centroids_array, count_macrophages #final centroids of red nuclei, final count of macrohages
    
    def distance_to_ground_truth(self, centroids_array):
        distances = [] #distance from each centroid to each x,y point of ground truth
        for i in range(len(centroids_array)):
            for j in range(len(self.ground_truth)):
                dist = self.distance(centroids_array[i][0], self.ground_truth[j][0],  centroids_array[i][1], self.ground_truth[j][1])
                row = [centroids_array[i][0], centroids_array[i][1], self.ground_truth[j][0], self.ground_truth[j][1], dist]
                distances.append(row)
        return distances
    
    def matches(self, distances):
        # Centroids within 25 pixels of ground truth x,y coordinate taken as a match - True Positive  
        df = np.array(distances)
        df_match = pd.DataFrame({'Centroid x': df[:, 0], 'Centroid y': df[:, 1], 'Count x': df[:, 2], 'Count y': df[:, 3], 'Distance': df[:, 4]})
        df_ordered = df_match.sort_values(by=['Distance'])
        df_min_dist = df_ordered[df_ordered.Distance <= 25] 
        # Code to check true positives - remove duplicated centroids or coords. Ordered list, so just check rest of list as go through row by row
        matches = df_min_dist.to_numpy()
        return matches
    
    # lists of true positives, false positives and false negatives 
    # based on matches bewteen centroids and ground truth x,y
    def missed(self, matches, centroids_array):
        centroids = centroids_array.tolist() #final list of centroids of cells after merging
        ground_truth = self.ground_truth.tolist() #list of ground truth x,y coordinates
        true_positives = (matches[:, :2]).tolist() #list of centroids that match the ground truth
        x_y_matches = (matches[:, 2:4]).tolist() #list of ground truth x,y values that have centroid matches
        
        false_positives = []
        for row in centroids:
            if row not in true_positives:
                false_positives.append(row)
                
        false_negatives = []       
        for row in ground_truth:
            if row not in x_y_matches:
                false_negatives.append(row)
                
        return false_positives, false_negatives, true_positives
    
    # Benchmark analysis - F1 score
    def f1(self, false_positives, false_negatives, true_positives):
        true_pos = len(true_positives)
        false_pos = len(false_positives)
        false_neg = len(false_negatives)
        precision = true_pos/(true_pos + false_pos)
        recall = true_pos/(true_pos + false_neg)
        f1 = 2* ((precision * recall)/(precision + recall))
        return true_pos, false_pos, false_neg, precision, recall, f1
