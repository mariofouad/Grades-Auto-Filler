from skimage.transform import hough_line, hough_line_peaks,probabilistic_hough_line, rotate
from skimage.transform import ProjectiveTransform, warp
from skimage.measure import find_contours
import pytesseract
import easyocr
from skimage import transform, filters, exposure, util
from PIL import Image

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


def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) #Rotate image to keep it vertical
    return np.array(img)

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


def trim_border(image, border_size=10):
    H, W = image.shape
    return image[border_size:H-border_size, border_size:W-border_size]

def edge_detection(image):
    H, W = image.shape
    diag = int(np.hypot(H, W))

    image = image.astype(np.float32) / 255.0 

    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    gx_abs = np.abs(gx)
    gy_abs = np.abs(gy)

    gx_n = gx_abs / (np.percentile(gx_abs, 90) + 1e-6)
    gy_n = gy_abs / (np.percentile(gy_abs, 90) + 1e-6)

    mag_balanced = np.maximum(gx_n, gy_n)
    mag_balanced = np.clip(mag_balanced, 0, 1)
    
    edges = mag_balanced > 0.6 
    edges = edges.astype(np.uint8) * 255

    return edges

def line_detection(edges, image):
    H, W = image.shape
    diag = int(np.hypot(H, W))

    horizontal_lines = np.zeros_like(image)
    vertical_lines = np.zeros_like(image)
    intersections = np.zeros_like(image)

    acc, angles, dists = hough_line(edges) 
    acc, angles, dists = hough_line_peaks(acc, angles, dists, threshold=0.75 * np.max(acc),  
                                        min_distance = int(0.01*diag), num_peaks=40) 
    
    for i in range(len(angles)): 
        theta = abs(angles[i]) 
        if not (theta < np.radians(5) or theta > np.radians(85)): 
            continue 
        if theta < np.radians(45): 
            a = math.cos(angles[i]) 
            b = math.sin(angles[i]) 
            x0 = a * dists[i] 
            y0 = b * dists[i] 
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a))) 
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a))) 
            cv2.line(vertical_lines, pt1, pt2, (255, 255, 255), 1) 
        elif theta > np.radians(45): 
            a = math.cos(angles[i]) 
            b = math.sin(angles[i]) 
            x0 = a * dists[i] 
            y0 = b * dists[i] 
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a))) 
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a))) 
            cv2.line(horizontal_lines, pt1, pt2, (255, 255, 255), 1) 

    intersections = np.bitwise_and(horizontal_lines > 0, vertical_lines > 0) 
    points = np.argwhere(intersections == 1)

    #show_images([horizontal_lines, vertical_lines, intersections])

    return points

def cluster_1d(values, tol):
    values = sorted(values)
    clusters = [[values[0]]]

    for v in values[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    return [int(np.mean(c)) for c in clusters]

def cluster_rows_columns(points, y_tol, x_tol):
    xs = [p[1] for p in points]
    ys = [p[0] for p in points]

    row_ys = cluster_1d(ys, y_tol)
    col_xs = cluster_1d(xs, x_tol)

    return row_ys, col_xs

def cell_extraction(image):
    image = trim_border(image, border_size=3)

    H, W = image.shape
    diag = int(np.hypot(H, W))

    edges = edge_detection(image)
    points = line_detection(edges, image)
    
    y_tol = 0.01 * H
    x_tol = 0.01 * W

    rows, columns = cluster_rows_columns(points, y_tol, x_tol) 
    
    num_rows = len(rows) - 1
    num_cols = len(columns) - 1

    cell_points = np.empty((num_rows, num_cols), dtype=object)
    cell_images = np.empty((num_rows, num_cols), dtype=object)
        
    for r in range(num_rows):
        for c in range(num_cols):
            x1, y1 = columns[c],   rows[r]
            x2, y2 = columns[c+1], rows[r+1]

            tl = (x1, y1)
            br = (x2, y2)

            cell_points[r, c] = (tl, br)
            cell_images[r, c] = image[y1:y2, x1:x2]

    return cell_images



def enhance_cell_for_ocr_skimage(cell_gray, out_size=96):
    cell = util.img_as_float(cell_gray)  # ensure float [0..1]

    cell = exposure.rescale_intensity(cell, in_range="image", out_range=(0, 1))  # normalize contrast
    cell_up = transform.resize(cell, (out_size, out_size), anti_aliasing=True, preserve_range=True)  # resample :contentReference[oaicite:5]{index=5}

    thresh = filters.threshold_sauvola(cell_up, window_size=21, k=0.2)  # local thresholding :contentReference[oaicite:6]{index=6}
    binary = cell_up < thresh  # dark ink = foreground

    return (binary.astype(np.uint8) * 255) 

def cell_preprocessing(cell_image, ocr = False):

    if ocr:
        thresholded = enhance_cell_for_ocr_skimage(cell_image)
    else:
        cell_image = cell_image * 255.0
        cell_image = median(cell_image, np.ones((2,2)))

        local_thresh = threshold_local(cell_image, block_size = 7, offset=4)
        thresholded = cell_image > local_thresh
        thresholded = np.bitwise_invert(thresholded)


    #show_images([cell_image, thresholded])
    # Trimming border
    size = thresholded.shape
    thresholded = thresholded[int(size[0]*0.1):int(size[0]*0.95),
                              int(size[1]*0.05):int(size[1]*0.9)]  # remove top 10% and right 10%

    #thresholded = np.bitwise_invert(thresholded)

    row_ink = np.sum(thresholded > 0, axis=1)
    col_ink = np.sum(thresholded > 0, axis=0)

    min_row_ink = thresholded.shape[1] * 0.6   # 1% of row width
    min_col_ink = thresholded.shape[0] * 0.6   # 1% of column height

    valid_rows = np.where(row_ink < min_row_ink)[0]
    valid_cols = np.where(col_ink < min_col_ink)[0]
    
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        raise ValueError("No content detected")

    y_min, y_max = valid_rows[[0, -1]]
    x_min, x_max = valid_cols[[0, -1]]

    trimmed = thresholded[y_min+1:y_max, x_min+1:x_max]
    trimmed = binary_closing(trimmed, np.ones((3,3)))
    #"""""" Sha8ala """""" 90% (3ala damanet oufes)

    return trimmed   

def check_empty_cell(cell_image):
    ink_pixels = np.sum(cell_image > 0)
    total_pixels = cell_image.size
    ink_ratio = ink_pixels / total_pixels

    if ink_ratio < 0.02:  # less than 2% ink
        return True
    return False

def detect_question_mark(cell_image):
    return False

def detect_line_symbols(cell_image):
    H, W = cell_image.shape
    diag = int(np.hypot(H, W))

    horizontal_lines = np.zeros_like(cell_image).astype(np.uint8)
    vertical_lines = np.zeros_like(cell_image).astype(np.uint8)
    diagonal_lines = np.zeros_like(cell_image).astype(np.uint8)

    count_h = 0
    count_v = 0
    count_d = 0

    acc, angles, dists = hough_line(cell_image) 
    acc, angles, dists = hough_line_peaks(acc, angles, dists, threshold=0.7 * np.max(acc),  
                                        min_distance = int(0.07*diag), num_peaks=10) 
    
    for i in range(len(angles)): 
        theta = abs(angles[i]) 
        if not (theta < np.radians(15) or theta > np.radians(75)):
            a = math.cos(angles[i]) 
            b = math.sin(angles[i]) 
            x0 = a * dists[i] 
            y0 = b * dists[i] 
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a))) 
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a))) 
            cv2.line(diagonal_lines, pt1, pt2, (255, 255, 255), 1)
            count_d += 1
        elif theta < np.radians(45): 
            a = math.cos(angles[i]) 
            b = math.sin(angles[i]) 
            x0 = a * dists[i] 
            y0 = b * dists[i] 
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a))) 
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a))) 
            cv2.line(vertical_lines, pt1, pt2, (255, 255, 255), 1)
            count_v += 1
        elif theta > np.radians(45): 
            a = math.cos(angles[i]) 
            b = math.sin(angles[i]) 
            x0 = a * dists[i] 
            y0 = b * dists[i] 
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a))) 
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a))) 
            cv2.line(horizontal_lines, pt1, pt2, (255, 255, 255), 1)
            count_h += 1
        
    #show_images([horizontal_lines, vertical_lines, diagonal_lines])

    if count_h > 0 and count_v > 0:
        #print("Box")
        return True, 0
    elif count_d > 0 and count_d < 2:
        #print("Check")
        return True, 5
    elif count_d > 1:
        #print("X")
        return True, 0
    elif count_h == 1:
        #print("Dashline")
        return True, 0
    elif count_h > 1:
        #print(count_h, "Horizontal")
        return True, 5 - count_h
    elif count_v > 0:
        #print(count_v, "Vertical")
        return True, count_v
    return False, 0

def ocr_check_id(cell_image):
    cell_image = (cell_image * 255.0).astype(np.uint8)

    pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    config = (
        "--oem 3 "
        "--psm 7 digits"
        "-c tessedit_char_whitelist=0123456789"
    )

    extracted_text = pytesseract.image_to_string(cell_image, config="--psm 7 digits")
    #print(f"Extracted Text: {extracted_text.strip()}")
    return extracted_text

def ocr_check_handwriting(cell_image):
    cell_image = enhance_cell_for_ocr_skimage(cell_image)
    cell_image = np.bitwise_invert(cell_image)
    cell_image = (cell_image * 255.0).astype(np.uint8)

    cell_image = binary_opening(cell_image, np.ones((3,3)))
    cell_image = (cell_image * 255.0).astype(np.uint8)
    
    #show_images([cell_image])

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(cell_image, allowlist='0123456789oO', detail=0)
    #print(result)
    if result != []:
        return True, result[0]
    return False, " "

def read_cell(cell_image, ocr = "None"):
    if ocr == "id":
        return ocr_check_id(cell_image)
    
    preprocessed = cell_preprocessing(cell_image)
    #show_images([preprocessed, cell_image])

    if not check_empty_cell(preprocessed):
        if detect_question_mark(preprocessed):
            return "?"
        else:
            detection, text = ocr_check_handwriting(cell_image)
            if detection: 
                return text
            detection, line_val = detect_line_symbols(preprocessed)
            if detection:
                return line_val

    return " "

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

def Module1(import_image, export_filename):
    image = load_image(import_image)

    preprocessed_image = preprocessing(image)
    cells = cell_extraction(preprocessed_image)

    values = np.empty((cells.shape[0]-1, cells.shape[1]-2), dtype=object)

    for r in range(1, cells.shape[0]):
        values[r-1, 0] = read_cell(cells[r, 0], ocr="id")
        for c in range(3, cells.shape[1]):
            cell_image = cells[r, c]
            values[r-1, c-2] = read_cell(cell_image, ocr="None")

    export_excel(values, export_filename, column_names = 1)