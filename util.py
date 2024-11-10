import easyocr
import os

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
# dict_char_to_int = {'O': '0',
#                     'I': '1',
#                     'J': '3',
#                     'A': '4',
#                     'G': '6',
#                     'S': '5'}
dict_char_to_int = {'O': 'O',
                    'I': 'I',
                    'J': 'J',
                    'A': 'A',
                    'G': 'G',
                    'S': 'S'}

# dict_int_to_char = {'0': 'O',
#                     '1': 'I',
#                     '3': 'J',
#                     '4': 'A',
#                     '6': 'G',
#                     '5': 'S'}
dict_int_to_char = {'0': '0',
                    '1': '1',
                    '3': '3',
                    '4': '4',
                    '6': '6',
                    '5': '5'}

def write_csv(results, output_path):
    """
    Write the results to a CSV file. If the file already exists, append to it.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    # Check if the file already exists
    file_exists = os.path.exists(output_path)

    # Open the file in append mode if it exists, else in write mode
    with open(output_path, 'a' if file_exists else 'w') as f:
        # If the file does not exist, write the header
        if not file_exists:
            f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                    'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                    'license_number_score'))

        # Append the results to the file
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
    f.close()

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)
    print(detections)

    if detections == [] :
        return None, None

    for detection in detections:
        bbox, text, score = detection

        #text = text.upper().replace(' ', '')
        text = text.upper()
        print(text)

        if text is not None and score is not None and bbox is not None and len(text) >= 4:
        #if license_complies_format(text):
        #    return format_license(text), score
            return text, score

    return None, None

