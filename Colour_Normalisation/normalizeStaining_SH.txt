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
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
        
        
    # Haematoxylin, Red, DAB reference (where Red is Alkaline Phosphatase) - OD matrix
    HRDref = np.array([[0.651, 0.185, 0.269],
                      [0.701, 0.78, 0.568],
                      [0.29, 0.598, 0.778]])
    

        
    maxCRef = np.array([0.7, 1.0308, 0.5])
    
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
    
    #eigvecs *= -1
    print("DYE VECTORS 0 = " + str(eigvecs[:, 0]))
    print("DYE VECTORS 1 = " + str(eigvecs[:, 1]))
    print("DYE VECTORS 2 = " + str(eigvecs[:, 2]))
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to red second
    if vMin[0] > vMax[0]:
        HRD = np.array((vMin[:,0], vMax[:,0], eigvecs[:, 0])).T
    else:
        HRD = np.array((vMax[:,0], vMin[:,0], eigvecs[:, 0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HRD,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99), np.percentile(C[2,:], 99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HRDref.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
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
