# Addapted from https://github.com/schaugf/HEnorm_python to determine the OD vectors for three stains:
# 1. Haematoxylin (H) - blue
# 2. Red (R) - Alkaline Phosphatase
# 3. DAB (D) - brown

import argparse
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 11135877120


def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        R: red image
        D: DAB image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
        
        
    # Haematoxylin, Red, DAB reference (where Red is Alkaline Phosphatase)
    # Stain OD  reference matrix, used to artificially colour the output images
    HRDref = np.array([[0.651, 0.185, 0.269],
                      [0.701, 0.78, 0.568],
                      [0.29, 0.598, 0.778]])
    

    # Reference stain concentrations - determines intensity of colour in output images
    maxCRef = np.array([0.5, 1.0308, 0.5])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    print("EIGENVALS = " + str(eigvals))
    print("Eigen vector 1 = " + str(eigvecs[:, 0]))
    print("Eigen vector 2 = " + str(eigvecs[:, 1]))
    print("Eigen vecotr 3 = " + str(eigvecs[:, 2]))
    
    #eigvecs *= -1
    
    # project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # project on the plane spanned by the eigenvectors corresponding to the smallest 
    # and largest eigenvalues (determine thrid OD vector - residual)
    That2 = ODhat.dot(eigvecs[:,(0,2)])

    phi2 = np.arctan2(That2[:,1],That2[:,0])

    minPhi2 = np.percentile(phi2, alpha)
    maxPhi2 = np.percentile(phi2, 100-alpha)
    
    vMin2 = eigvecs[:,(0,2)].dot(np.array([(np.cos(minPhi2), np.sin(minPhi2))]).T)
    vMax2 = eigvecs[:,(0,2)].dot(np.array([(np.cos(maxPhi2), np.sin(maxPhi2))]).T)
    
    
    # a heuristic to assign the above determined OD vectors to the correct stain/colour
    # Use the R channel to assign haematoxylin [0]
    if (vMin[0] > vMax[0]) and (vMin[0] > vMin2[0]) and (vMin[0] > vMax2[0]):
        H = vMin[:,0]
        print(H)
    elif (vMax[0] > vMin[0]) and (vMax[0] > vMin2[0]) and (vMax[0] > vMax2[0]):
        H = vMax[:,0]
        print(H)
    elif (vMin2[0] > vMin[0]) and (vMin2[0] > vMax[0]) and (vMin2[0] > vMax2[0]):
        H = vMin2[:,0]
        print(H)
    else:
        H = vMax2[:,0]
        print(H)
        
    # Use the G channel to assign the red stain [1]
    if (vMin[1] > vMax[1]) and (vMin[1] > vMin2[1]) and (vMin[1] > vMax2[1]):
        R = vMin[:,0]
        print(R)
    elif (vMax[1] > vMin[1]) and (vMax[1] > vMin2[1]) and (vMax[1] > vMax2[1]):
        R = vMax[:,0]
        print(R)
    elif (vMin2[1] > vMin[1]) and (vMin2[1] > vMax[1]) and (vMin2[1] > vMax2[1]):
        R = vMin2[:,0]
        print(R)
    else:
        R = vMax2[:,0]
        print(R)
        
    # Use the B channel to assign DAB [2]
    if (vMin[2] > vMax[2]) and (vMin[2] > vMin2[2]) and (vMin[2] > vMax2[2]):
        D = vMin[:,0]
        print(D)
    elif (vMax[2] > vMin[2]) and (vMax[2] > vMin2[2]) and (vMax[2] > vMax2[2]):
        D = vMax[:,0]
        print(D)
    elif (vMin2[2] > vMin[2]) and (vMin2[2] > vMax[2]) and (vMin2[2] > vMax2[2]):
        D = vMin2[:,0]
        print(D)
    else:
        D = vMax2[:,0]
        print(D)

    # Stain OD vectors used to deconvolute image    
    HRD = np.array((H, R, D)).T
    print(HRD)
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains (same as scikit-image separate_stains method)
    # Stain separation - Colour deconvolution
    C = np.linalg.lstsq(HRD,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99), np.percentile(C[2,:], 99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix - atrificially colour the output images
    Inorm = np.multiply(Io, np.exp(-HRDref.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin, red and DAB (use reference to artificially colur the output images)
    H = np.multiply(Io, np.exp(np.expand_dims(-HRDref[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    R = np.multiply(Io, np.exp(np.expand_dims(-HRDref[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    R[R>255] = 254
    R = np.reshape(R.T, (h, w, 3)).astype(np.uint8)
    
    D = np.multiply(Io, np.exp(np.expand_dims(-HRDref[:,2], axis=1).dot(np.expand_dims(C2[2,:], axis=0))))
    D[D>255] = 254
    D = np.reshape(D.T, (h, w, 3)).astype(np.uint8)
    
    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile+'.png')
        Image.fromarray(H).save(saveFile+'_H.png')
        Image.fromarray(R).save(saveFile+'_R.png')
        Image.fromarray(D).save(saveFile+'_D.png')

    return Inorm, H, R, D
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFile', type=str, default='example1.tif', help='RGB image file')
    parser.add_argument('--saveFile', type=str, default='output', help='save file')
    parser.add_argument('--Io', type=int, default=240)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.15)
    args = parser.parse_args()
    
    img = np.array(Image.open(args.imageFile))

    normalizeStaining(img = img,
                      saveFile = args.saveFile,
                      Io = args.Io,
                      alpha = args.alpha,
                      beta = args.beta)
