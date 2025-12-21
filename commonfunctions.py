import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram, equalize_hist , equalize_adapthist
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv,rgba2rgb
import cv2 as cv2
# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median, gaussian, threshold_local
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb
from skimage.transform import hough_line, hough_line_peaks,probabilistic_hough_line, rotate
from skimage.exposure import adjust_gamma

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

# Morphological
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening, skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.transform import ProjectiveTransform, warp

# Excel Generation
import pandas as pd

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 


def showHist(img,histogramImg):
    plt.figure()
    bar(histogramImg[1]*255, histogramImg[0], width=0.8, align='center')


def order_points(pts): # take 4 points and orders them as follows:top left,top right,bottom left,bottom right
    rectangle=np.zeros((4,2), dtype= np.float32 )
    
    s=pts.sum(axis=1) # the top left 3andaha smallest x+y w bottom rught akbar
    rectangle[0]=pts[np.argmin(s)]
    rectangle[2]=pts[np.argmax(s)]
    
    difference=np.diff(pts,axis=1) # top right 3andaha smallest y-x w bottom right akbar
    rectangle[1]=pts[np.argmin(difference)]
    rectangle[3]=pts[np.argmax(difference)]
    
    return rectangle

#takes 4 corners of paper and then warps them into a perfect rectangular zy akenak bt3ml scan l war2a on camscanner
def four_point(image,pts):
    rectangle=order_points(pts)
    
    top_left,top_right,bottom_right,bottom_left=rectangle
    
    bottom_edge_width=np.linalg.norm(bottom_right-bottom_left)
    top_edge_width=np.linalg.norm(top_right-top_left)
    maxwidth=int(max(bottom_edge_width,top_edge_width)) # 34an amna3 hetta tkoon cropped f ba5od el max

    right_edge_height=np.linalg.norm(top_right-bottom_right)
    left_edge_height=np.linalg.norm(top_left-bottom_left)
    maxheight=int(max(right_edge_height,left_edge_height))
    
    #ba7ot el 4 ordered points into a rectangle
    final_rectangle=np.array([[0,0],[maxwidth-1,0],[maxwidth-1,maxheight-1],[0,maxheight-1] ],dtype=np.float32 )
    #this produces a 3x3 homography matrix which encodes rotation,translation,scaling
    mapping=cv2.getPerspectiveTransform(rectangle,final_rectangle)
    #to apply warping for every pixel in the paper from the top till the bottom
    result=cv2.warpPerspective(image,mapping,(maxwidth,maxheight))
    
    return result

#it takes image and tries to find the 4 corner points to apply warping on it
def detect_document_contour(image):
    if image.shape[2] > 3:
        image = image[:, :, :3]
    gray=rgb2gray(image)
    gray_blurred=gaussian(gray,sigma=1)
    edges = canny(gray_blurred, sigma=1, low_threshold=30/255, high_threshold=100/255).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#retr-external to retun the outer contour , msh 3ayez el table grid contour
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True) # 34an a sort el contour from largest(paper) to smallest

    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        for eps in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)

    # to get the best rectangle even if not perfect contour
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)
    

def deskew(image):
    result=np.zeros_like(image)
    edge=canny(image,sigma=1,low_threshold=10,high_threshold=70)
    lines=probabilistic_hough_line(edge,line_length=80,line_gap=5)
  
    if not lines:
        return image
    
    #Storing angles of lines detected
    angles=[]
    
    for(x0,y0),(x1,y1) in lines:
        delta_x=x1-x0
        delta_y=y1-y0
        ang=np.degrees(np.arctan2(delta_y,delta_x))
        
        while ang>90:
            ang-=180
           
        while ang<-90:
            ang+=180
            
        angles.append(ang)
        
    horizontal=[a for a in angles if abs(a)<45] #to keep the sllightly tilted rows 
    if len(horizontal)<3:
        return image  
        
    skew=float(np.median(horizontal)) #to avoid extreme outlier eno ybawaz el angles ely tal3a
    skew = float(np.clip(skew, -10, 10))  # prevent over-rotation

    rotated =rotate(image,angle=-skew,resize=False, preserve_range=True )
    return rotated 
           
def preprocessing(image):
    doc_cont=detect_document_contour(image)
    warped=four_point(image,doc_cont)
    
    image_gray = rgb2gray(warped)    
 
    image_deskewed=deskew(image_gray)

    return image_deskewed

def export_excel(data, file_name,column_names=None):
    
    if len(data)==0:
        print( "Empty Data Array Is Provided" )
        
    num_columns=len(data[0])    
        
    for row in data:
        if len(row) !=num_columns:
            print( "Every student must have the same number of entries" )
            
    if column_names is not None:
        columns=["Student Code"]+[f"Grade_{i}"for i in range (1,num_columns)]     
           
    df = pd.DataFrame(data,columns=columns)
    df.to_excel(file_name + ".xlsx", index=False)
    
    return file_name