# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from optparse import OptionParser
import os
import openslide
from openslide import open_slide, ImageSlide
from optparse import OptionParser
from PIL import Image
import numpy as np


Image.MAX_IMAGE_PIXELS = 9519680000

def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.30):
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

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    print("DYE VECTORS 0 = " + str(eigvecs[:, 0]))
    print("DYE VECTORS 1 = " + str(eigvecs[:, 1]))
    print("DYE VECTORS 2 = " + str(eigvecs[:, 2]))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile + '.tiff')
        Image.fromarray(H).save(saveFile + '_H.tiff')
        Image.fromarray(E).save(saveFile + '_E.tiff')

    return Inorm, H, E

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>')
    parser.add_option('-o', '--output', metavar='NAME', dest='basename',
                help='base name of output file')
    parser.add_option('-l', '--level', metavar='PIXELS', dest='level',
                type='int', default=0,
                help='level [0]')
    parser.add_option('-x', '--xcoord', metavar='PIXELS', dest='x_coord',
                type='int', default=0,
                help='x coordinate [0]')
    parser.add_option('-y', '--ycoord', metavar='PIXELS', dest='y_coord',
                type='int', default=0,
                help='y coordinate [0]')
    parser.add_option('-w', '--width', metavar='PIXELS', dest='width',
                type='int', default=254,
                help='tile width [254]')
    parser.add_option('-g', '--height', metavar='PIXELS', dest='height',
                type='int', default=254,
                help='tile height [254]')

    (opts, args) = parser.parse_args()
    try:
        slidepath = args[0]
    except IndexError:
        parser.error('Missing slide argument')
    if opts.basename is None:
        opts.basename = os.path.splitext(os.path.basename(slidepath))[0]

    # Read WTS ...
    wsi = open_slide(slidepath)
    # print ("PROPERTIES = " + str(wsi.properties))

    print ("WIDTH = " + wsi.properties['openslide.level[0].width'])
    print ("HEIGHT = " + wsi.properties['openslide.level[0].height'])

    tile = wsi.read_region((opts.x_coord, opts.y_coord), opts.level, (opts.width, opts.height)).convert('RGB')

    tile.save(opts.basename + "_raw.tiff")

    pix = np.asarray(tile)
#    pix = np.array(tile.).reshape(tile.size[0], tile.size[1], 3)
    normalizeStaining(pix, saveFile=opts.basename)
