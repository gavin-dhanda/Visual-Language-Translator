from image_processing import segmentation as seg
from text_manipulation import reformatter as ref
from result_projection import project as proj
from image_processing import neural_network
import cv2 
import random
from result_projection import project as proj

def main():
    image_path = 'image_processing/images/letssee.png'
    show_contours = False
    img = cv2.imread(image_path)
    letters_old, boxes = seg.segment_image(img, show_contours)
    
    # Gets the predicted output string from the neural network.
    letters = neural_network.find_predicted_letters(True, letters_old)
      
    # Append items from letters into a sequence. 
    # Find the average distance between two adjacent letters using the boxes.
    # Use the average distance to determine the number of spaces between each letter.
    # Use the sequence and the number of spaces to generate a coherent English text.
         
    avg_distance = 0
    count = 0
    for i in range(len(boxes) - 1):
        count += 1
        avg_distance += abs(boxes[i + 1][0] - (boxes[i][0] + boxes[i][2]))
    avg_distance = avg_distance / count
    epsilon = avg_distance ** 0.5

    # Generate a sequence of characters from the letters, adding spaces as necessary.
    sequence = ""
    for i in range(len(letters) - 1):
        sequence += letters[i]
        if (boxes[i + 1][0] - (boxes[i][0] + boxes[i][2])) > avg_distance + epsilon:
            sequence += " "
    sequence += letters[len(letters) - 1]

    # Test
    language = "es"
    original, translated = ref.generate_text(sequence, ref.client, language)
    translated_image = proj.format_lines(boxes, translated.split(" "), img)
    cv2.imshow("Translation into " + language + "!", translated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Make it run main() if this script is run as the main module.
if __name__ == "__main__":
    main()
