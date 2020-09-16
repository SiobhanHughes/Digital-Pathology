import static qupath.lib.roi.PathROIToolsAwt.getShape;
import java.awt.image.BufferedImage
import java.awt.Color
import javax.imageio.ImageIO

// Get java.awt.Shape objects for each annotation
def shapes = getAnnotationObjects().collect({getShape(it.getROI())})

// Create a grayscale image, here it's 10% of the full image size
double downsample = 10.0
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

// Save the result
def outputFile = getQuPath().getDialogHelper().promptToSaveFile("Save binary image", null, null, "PNG", ".png")
ImageIO.write(img, 'PNG', outputFile)