# GradientEnhance
Gradient Enhancement

Implement from this paper https://www.cs.huji.ac.il/~danix/hdr/hdrc.pdf

The main Ideal of this enhancement is enhance in gradient space

By analysis gradient space, we know that which regions are low contrast, and by emphasize the magnitude of the gradient, we enhance contrast of this regions

After That Form poisson euqation and solve this by Intel MKL library

# original Image <After stretch 0-255 to display on the screen>
![image](https://github.com/user-attachments/assets/b94737ef-63a1-4939-9370-be387b188165)


# Processed Image 
![image](https://github.com/user-attachments/assets/38add2ec-629c-4aac-b609-7201f934c76b)


PRoject:
require OpenCV window variable that is direction of opencv folder
