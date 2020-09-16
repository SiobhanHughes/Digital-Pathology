import qupath.lib.roi.RoiTools
import java.awt.image.BufferedImage
import java.awt.Color
import javax.imageio.ImageIO
import qupath.lib.gui.scripting.QPEx

// Get java.awt.Shape objects for each annotation
def shapes = getAnnotationObjects().collect({it.getROI().getShape()})

def pathOutput = buildFilePath(QPEx.PROJECT_BASE_DIR, 'cell_annotations_mask')
mkdirs(pathOutput)

def imageExportType = 'JPG'

// Create a grayscale image, here it's 10% of the full image size
double downsample = 1.0
def server = getCurrentImageData().getServer()
int w = (server.getWidth() / downsample) as int
int h = (server.getHeight() / downsample) as int
def img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)

// Paint the shapes (this is just 'standard' Java - you might want to modify)
def g2d = img.createGraphics()
g2d.scale(1.0/downsample, 1.0/downsample)
g2d.setColor(Color.WHITE)
for (shape in shapes)
    g2d.fill(shape)
g2d.dispose()



// Export the mask
def fileMask = new File(pathOutput, 'CD163-mask.png')
ImageIO.write(img, 'PNG', fileMask)


// Save the result
//def outputFile = getQuPath().getDialogHelper().promptToSaveFile("Save binary image", null, null, "PNG", ".png")
//ImageIO.write(img, 'PNG', outputFile)