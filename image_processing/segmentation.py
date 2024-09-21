import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

def segment_image(image, show_contours):
    # preprocessing steps
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.resize(gray_image, (gray_image.shape[1] * 5, gray_image.shape[0] * 5))
    # find the average pixel value
    avg_pixel_value = np.mean(gray_image)
    # if the average pixel value is less than 127, invert the image
    if avg_pixel_value < 127:
        gray_image = cv2.bitwise_not(gray_image)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(cleaned_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if show_contours:
        contour_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #get the average contour w and h
    avg_w = 0
    avg_h = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        avg_w += w
        avg_h += h
    avg_w = avg_w / len(contours)
    avg_h = avg_h / len(contours)


    boxes = []
    # use hierarchy to remove inside of letters being identified as contours
    if hierarchy is not None:
        for idx, contour in enumerate(contours):
            if hierarchy[0][idx][3] == -1:
                # if contour is significantly smaller, continue:
                if cv2.contourArea(contour) < 0.2 * avg_w * avg_h:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w > 15 and h > 15:
                    boxes.append((x, y, w, h, contour))
    
    boxes.sort(key=lambda b: b[0])

    # merge contours to preserve is
    merged_contours = []
    i = 0
    while i < len(boxes):
        x, y, w, h, contour = boxes[i]
        if i + 1 < len(boxes):
            nx, ny, nw, nh, ncontour = boxes[i + 1]
            if abs(x - nx) < 25 and abs(y + h - ny) < 50: 
                new_x = min(x, nx)
                new_y = min(y, ny)
                new_w = max(x + w, nx + nw) - new_x
                new_h = max(y + h, ny + nh) - new_y
                merged_contours.append((new_x, new_y, new_w, new_h))
                i += 1
            else:
                merged_contours.append((x, y, w, h))
        else:
            merged_contours.append((x, y, w, h))
        i += 1
    
    # extract letters, but also make them uniform size (target size x target size) for cnn
    target_size = 28
    letters = []
    for x, y, w, h in merged_contours:
        letter_img = gray_image[y:y+h, x:x+w]
        h, w = letter_img.shape
        top = bottom = left = right = 0
        if h > w:
            left = right = (h - w) // 2
        else:
            top = bottom = (w - h) // 2
        padded_img = cv2.copyMakeBorder(letter_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        resized_img = cv2.resize(padded_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        letters.append(resized_img)
    
    return letters, merged_contours

def prepare_data(image_path, show_contours, show_data):
    img = cv2.imread(image_path)
    letters, merged_contours = segment_image(img, show_contours)
    if show_data:
        for i, letter in enumerate(letters):
            plt.subplot(1, len(letters), i+1)
            plt.imshow(letter, cmap='gray')
            plt.axis('off')
        plt.show()
    return letters, merged_contours

# parse in command line arguments: image path, show contours
# ensure we have the correct number of arguments
# if len(sys.argv) != 4:
#     print('Invalid number of arguments. Correct usage: python segmentation.py <image_path> <show_contours> <show_letters>')
#     sys.exit(1)
# image_path = sys.argv[1]
# show_contours = (sys.argv[2] == 'True')
# show_data = (sys.argv[3] == 'True')

# letters, boxes = prepare_data(image_path, show_contours, show_data)







