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

from commonfunctions import *

def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) #Rotate image to keep it vertical
    return np.array(img)

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