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
    def get_OD_vector(self, rgb):
        """
        Function that takes RGB values as a list.
        Returns normalised OD vector.
        """
        rgb = np.array([rgb], dtype=np.uint8)
        rgb_float = sk.img_as_float(rgb)
        OD = -np.log10(rgb_float)
        normalised_OD = OD / np.sqrt(np.sum(OD**2))
        OD_vector = np.around(normalised_OD, decimals=3)
        return OD_vector[0]

    def deconvolute_hdr(self, slide):
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
        # image can be red_smooth or filled
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
    
    def red_mask(self, filled, threshold='otsu', red_small_objects=100):
        #3. Threshold for red stain channel
        if threshold == 'otsu':
            otsu_threshold = filters.threshold_otsu(filled)
            red_mask = filled > otsu_threshold
        else:
            mean_threshold = filters.threshold_mean(filled)
            red_mask = filled > mean_threshold
        #4. remove small objects - clean up mask (default value for min_size is 64)
        red_mask_rm = morphology.remove_small_objects(red_mask, min_size=red_small_objects)
        return red_mask_rm
    
    
class Nuclei_channel:
    """
    Class for processing the haematoxylin channel - nuclei
    """
    
    def __init__(self, nuclei_channel):
        self.nuclei = nuclei_channel
    
    def smooth_nuclei(self, image, nuclei_filter='median', nuclei_filter_size=3, nuclei_filter_sigma=1):
        # 1. Smoothing filter for haematoxylin channel
        if nuclei_filter == 'gaussian':
            nuclei_smooth = ndi.gaussian_filter(util.img_as_float(image), sigma=nuclei_filter_sigma)
        else:
            nuclei_smooth = ndi.median_filter(util.img_as_float(image), size=nuclei_filter_size)
        return nuclei_smooth
    
    def nuclei_mask(self, nuclei_smooth, nuclei_small_objects=100):
        #2. Threshold for haematoxylin channel
        nuclei_threshold = filters.threshold_otsu(nuclei_smooth)
        nuclei_mask = nuclei_smooth > nuclei_threshold
    
        #3. remove small objects - clean up mask. Default for min_size in skimage is 64.
        nuclei_mask_rm = morphology.remove_small_objects(nuclei_mask, min_size=nuclei_small_objects)
        #4. fill in holes
        nuclei_mask_fill = ndi.morphology.binary_fill_holes(nuclei_mask_rm)
        #5. binary opening
        nuclei_mask_open = morphology.binary_opening(nuclei_mask_fill)
        return nuclei_mask_open

class Segment:
    """
    Class for image segmentation using the watershed algorithm
    """
     
    # Distance transform of binary image (mask) and markers for watershed segmentation
    def markers(self, mask, peak_min_distance=15):
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
    
    # For segmentation using nuclei of red stained cells
    def red_nuclei(self, red_mask, nuclei_mask, red_nuclei_min_size=100):
        red_nuclei = red_mask & nuclei_mask
        red_nuclei_rm = morphology.remove_small_objects(red_nuclei, min_size = red_nuclei_min_size)
        return  red_nuclei_rm


class Count_cells:
    """
    Class to count cells (CD163 positive macropahges (red stain channel) for these pipelines),
    including benchmark analysis to output precision, recall and F1 scores when compared to 
    annotated ground truth.
    
    Initialise class with labels (output) from watershed segmentation.
    Ground truth default is None. 
    
    If ground truth available, it is initialised as a numpy array of the (x, y) coordinates of the centre of positive cells.
    (Typically annotations are done using QuPath counting function and (x, y) coordinates exported)
    """
    
    def __init__(self, labels, ground_truth=None):
        self.labels = labels # output of watershed segmentation
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
    
    def merge_centroids(self, centroids_array, max_dist=25):
        # reduce over-segmentation by keeping only one of a pair of centroids that mark the same cell or nucleus
        # return final set of centroids marking each cell/nucleus as well as the final count for number of cells
        num = max_dist #within max_dist (e.g. 25) pixels distance - set max distance for merging pairs of centroids
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
                dist = self.distance(centroids_array[i][0], self.ground_truth[j][0],
                                     centroids_array[i][1], self.ground_truth[j][1])
                row = [centroids_array[i][0], centroids_array[i][1], self.ground_truth[j][0], self.ground_truth[j][1], dist]
                distances.append(row)
        return distances
    
    def matches(self, distances, max_dist=25):
        # A centroid within max_dist pixels (set to 25 or 35) of a ground truth (x, y) coordinate taken as a match 
        # - True Positive   
        df = np.array(distances)
        df_match = pd.DataFrame({'Centroid x': df[:, 0], 'Centroid y': df[:, 1], 'Coord x': df[:, 2],
                                 'Coord y': df[:, 3], 'Distance': df[:, 4]})
        df_ordered = df_match.sort_values(by=['Distance'])
        df_min_dist = df_ordered[df_ordered.Distance <= max_dist] # set the max distance for true positive matches
        
        # from ordered array (smallest to largest distance), for a match, remove rows below that contain
        # the centroid or (x, y) coordinate that has already been matched
        matches_array = df_min_dist.to_numpy()
        matches_array = np.round(matches_array, decimals=5)
        matches = matches_array.tolist()
        
        remove = []
        for i in range(len(matches)):
            row = matches[i]
            if (i+1) < len(matches): # exit once reach end of array
                for j in range((i+1), len(matches)): # check rest of array to find rows to remove
                    if ((matches[j][0] == row[0] and matches[j][1] == row[1])
                        or (matches[j][2] == row[2] and matches[j][3] == row[3])):
                        remove.append(matches[j])

        # remove already matched pairs
        final_matches = []
        for pairs in matches:
            if pairs not in remove:
                final_matches.append(pairs)
        
        final_matches = np.array(final_matches)
        return final_matches
    
    # lists of true positives, false positives and false negatives 
    # based on matches bewteen centroids and ground truth (x, y) coordinates
    def fp_fn_tp(self, centroids_array, matches):
        centroids_array = np.round(centroids_array, decimals=5)
        centroids = centroids_array.tolist() #final list of centroids of cells after merging
        gt_array = np.round(self.ground_truth, decimals=5)
        ground_truth = gt_array.tolist() #list of ground truth x,y coordinates
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
        precision = round(precision, 2)
        recall = true_pos/(true_pos + false_neg)
        recall = round(recall, 2)
        f1 = 2* ((precision * recall)/(precision + recall))
        f1 = round(f1, 2)
        return true_pos, false_pos, false_neg, precision, recall, f1
