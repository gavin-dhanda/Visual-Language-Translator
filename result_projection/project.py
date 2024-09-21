import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sorting_x(i):
    return i[0]

def sorting_y(i):
    return i[1]


def create_rect(locations):

    by_x = sorted(locations, key=sorting_x)
    by_y = sorted(locations, key=sorting_y)

    top_left = (by_x[0][0], by_y[0][1])
    bottom_right = (by_x[-1][0] + by_x[-1][2], by_y[-1][1] + by_y[-1][3])
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    rect_coords = [top_left, top_right, bottom_left, bottom_right]

    rect_height = bottom_right[1] - top_right[1]
    rect_width = top_right[0] - top_left[0]

    return rect_height, rect_width, rect_coords


def add_text(draw, translated_text, rect_coords, font, rect_width, rect_height):

    lines = []
    words = translated_text.split()
    line = []
    total_height = 0
    height = 0
    for word in words:

        temp_line = " ".join(line + [word])
        width, height = draw.textsize(temp_line, font=font)
        if width <= rect_width:
            line.append(word)
        else:
            line = ' '.join(line)
            lines.append(line)
            total_height += (height * 1.1)
            line = [word]

    if line:
        line = ' '.join(line)
        lines.append(line)
        total_height += (height * 1.1)

    if total_height < (rect_height * 1.1) + 20:
        y = rect_coords[0][1]
        draw.rectangle(((rect_coords[0][0], rect_coords[0][1] - 10), (rect_coords[3][0], rect_coords[3][1] + 10)), fill=(255, 255, 255))
        for line in lines:
            draw.text((rect_coords[0][0], y), line, font=font, fill=(0,0,0))
            y += font.getsize(line)[1]
        return True

    else: return False


def format_lines(locations, trans_words, image):

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    locations_copy = []
    for location in locations:
        one_letter = []
        for unit in location:
            one_letter.append(unit // 5)
        location = tuple(one_letter)
        locations_copy.append(location)
        print("inner location: ", location)
    locations = locations_copy

    rect_height, rect_width, rect_coords = create_rect(locations)
    draw = ImageDraw.Draw(image_pil)
    for font_size in range(100, 10, -2):

        font = ImageFont.truetype('font.ttf', font_size)
        translated_text = " ".join(trans_words)
        if add_text(draw, translated_text, rect_coords, font, rect_width, rect_height):
            break

    final_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return final_image
