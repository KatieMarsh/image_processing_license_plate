import customtkinter
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import os
from ultralytics import YOLO
from util import set_background, write_csv
import csv
import easyocr
import uuid
from util import *
import matplotlib.pyplot as plt

folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"
reader = easyocr.Reader(['en'], gpu=False)

my_label =  ""


coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
vehicles = [2]
threshold = 0.15

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("License Plate detection")
        self.geometry(f"{700}x{700}") # width x height
        self.filename = ""
        self.input_folder = ""
        self.output_detection_folder = ""
        self.my_label = customtkinter.CTkLabel(self, text = "Hello")


        self.create_widgets()

    def create_widgets(self):
        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=650, height=200)
        self.tabview.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Image File")

        # configure grid of individual tabs
        self.tabview.tab("Image File").grid_columnconfigure(0, weight=1)  # Allows the entry to expand
        self.tabview.tab("Image File").grid_columnconfigure(1, weight=0)  # For Browse button
        self.tabview.tab("Image File").grid_columnconfigure(2, weight=0)  # For Run button
        self.tabview.tab("Image File").grid_columnconfigure(3, weight=1)  # Empty column for centering

   

        # Input folder selection with label and browse button
        self.filename_entry = customtkinter.CTkEntry(self.tabview.tab("Image File"), width=450)
        self.filename_entry.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.filename_entry.configure(state="readonly")
        # Set the state to readonly to prevent editing
        self.button_browse_input = customtkinter.CTkButton(self.tabview.tab("Image File"), text="Browse File", command=self.selectfile, text_color = "black", fg_color = "#E4CD05",  hover_color="#BA8E23")
        self.button_browse_input.grid(row=2, column=1, padx=10, pady=10)

        # Run detection button
        self.button_run = customtkinter.CTkButton(self.tabview.tab("Image File"), text="Run", command=self.run_detection_file, hover_color = "#06402b", fg_color="green")
        self.button_run.grid(row=2, column=2, padx=10, pady=10)

    def selectfile(self):
        self.filename_entry.configure(state="normal")
        self.filename = filedialog.askopenfilename()
        self.filename_entry.delete(0, tk.END)
        self.filename_entry.insert(0, self.filename)
        self.filename_entry.configure(state="readonly")

    def run_detection_file(self):
        self.my_label.destroy()
        self.run_detection(input_type='file')
        
    
    
    
    def run_detection(self, input_type):
        barcode_list = []
        

        if input_type not in ['folder', 'file']:
            raise ValueError("Invalid input type. Must be 'folder' or 'file'.")

        if input_type == 'folder':
            if not self.input_folder:
                messagebox.showwarning("Input Required", "Please select input folder")
                return
            input_path = self.input_folder
        elif input_type == 'file':
            if not self.filename:
                messagebox.showwarning("Input Required", "Please select an input file")
                return
            if not (self.filename.endswith('.jpg') or self.filename.endswith('.png') or self.filename.endswith('.JPG') or self.filename.endswith('.PNG') or self.filename.endswith('.JPEG') or self.filename.endswith('.jpeg')):
                messagebox.showwarning("Wrong input", "Please select file ending in '.jpg' or '.png'")
                return
            input_path = self.filename

        def process_image(filepath):
            license_numbers = 0
            results = {}
            licenses_texts = []
            image = cv2.imread(filepath)
            image = cv2.resize(image, (2048, 2048))

            
            # img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            
            #-----

            # # Histograms Equalization using OpenCV
            equ = cv2.equalizeHist(gray)

            # # Apply sharpening function
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            dst = cv2.filter2D(equ, -1, kernel)
            
                 
            
            # # Apply Fourier Transform
            #dft = cv2.dft(np.float32(equ), flags=cv2.DFT_COMPLEX_OUTPUT)
            #dft_shifted = np.fft.fftshift(dft)
            
            # # Create a mask for the low-pass filter
            #rows, cols = dst.shape
            #crow, ccol = rows // 2, cols // 2
            #mask = np.zeros((rows, cols, 2), np.uint8)
            # radius = 90  # Adjust radius for the low-pass filter
            # cv2.circle(mask, (ccol, crow), radius, (1, 1), -1)

            # # Apply the mask and inverse DFT
            #filtered_dft = dft_shifted * mask
            #dft_ishift = np.fft.ifftshift(filtered_dft)
            #img_back = cv2.idft(dft_ishift)
            # img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            # # Normalize img_back to a 0-255 range and convert to 8-bit
            #img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            #img_back = np.uint8(img_back)
            
            #------------
            
            
            
            # threshold_value, binary_image = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # print(threshold_value)
            
            # # Apply Gaussian blur
            blurred = cv2.GaussianBlur(dst, (3, 3), 0)
            sharp_gaussian = cv2.addWeighted(dst, 1.5, blurred, -0.7, 0)
            
            # Apply closing operation
            kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size as needed
            # closed = cv2.morphologyEx(sharp_gaussian, cv2.MORPH_CLOSE, kernel)
            closed = cv2.morphologyEx(sharp_gaussian, cv2.MORPH_CLOSE, kernel)

            #  # # Mean filtering
            blur = cv2.blur(closed,(3,3))
            
            # Convert the grayscale image to RGB
            img_back_rgb = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.title('Original Image')
            # plt.imshow(image, cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.title('Filtered Image')
            # plt.imshow(img_back_rgb)
            # plt.show()
            object_detections = coco_model(img_back_rgb)[0]
            license_detections = license_plate_detector(img_back_rgb)[0]
            
            # _, width, _ = image.shape
            # if width > 600: 
            #     image = resize_image(image)
            
            
            if len(object_detections.boxes.cls.tolist()) != 0 :
                for detection in object_detections.boxes.data.tolist() :
                    xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

                    if int(class_id) in vehicles :
                        cv2.rectangle(img_back_rgb, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
            else :
                xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
                car_score = 0
                
            if len(license_detections.boxes.cls.tolist()) != 0 :
                license_plate_crops_total = []
                for license_plate in license_detections.boxes.data.tolist() :
                    x1, y1, x2, y2, score, class_id = license_plate

                    cv2.rectangle(img_back_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                    license_plate_crop = img_back_rgb[int(y1):int(y2), int(x1): int(x2), :]

                    img_name = '{}.jpg'.format(uuid.uuid1())
                
                    cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 

                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img_back_rgb)

                    licenses_texts.append(license_plate_text)

                    if license_plate_text is not None and license_plate_text_score is not None  :
                        license_plate_crops_total.append(license_plate_crop)
                        results[license_numbers] = {}
                        
                        results[license_numbers][license_numbers] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score}} 
                        license_numbers+=1
                os.makedirs("csv_detections", exist_ok=True)
                csv_filename = os.path.join("csv_detections", "detection_results.csv")
                write_csv(results, csv_filename)
                img_wth_box = cv2.cvtColor(img_back_rgb, cv2.COLOR_BGR2RGB)
                cv2.imwrite('temp_image.png', img_wth_box)
    
            else : 
                img_wth_box = cv2.cvtColor(img_back_rgb, cv2.COLOR_BGR2RGB)
                #return [img_wth_box]
                    
            
            
            # barcode_data = None
            # barcode_type = None
            # # Detect barcodes in the grayscale image
            # barcodes = decode(gray)
            # for barcode in barcodes:
            #     # Extract barcode data and type
            #     barcode_data = barcode.data.decode("utf-8")
            #     barcode_type = barcode.type
            #     data = (barcode_data, barcode_type)
            #     barcode_list.append(data)

            #     # Draw a rectangle around the barcode
            #     (x, y, w, h) = barcode.rect
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)

            #     # Put barcode data and type on the image
            #     cv2.putText(image, f"{barcode_data} ({barcode_type})", (x, y - 2), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 4)

            # Save the processed image temporarily for display purposes
            #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = img_wth_box
            cv2.imwrite('temp_image.png', image_rgb)
            
                      
            if not results[0][0]['license_plate']['text']:
                messagebox.showwarning("Invalid", "Could not read license plate.")
                return
            
            print('Results:', results)
            

            # Update the text of the existing label
            #self.my_label.configure(text="License Plate Number: {0}".format(results[0][0]['license_plate']['text']))

            #-----
            
            
            # Create the label for the first time if it doesn't exist
            
            self.my_label = customtkinter.CTkLabel(self, text="License Plate Number: {0}".format(results[0][0]['license_plate']['text']))
            self.my_label.grid(row=2, column=1, padx=10, pady=15)
                
            # else: 
            #     self.my_label.configure(text="License Plate Number: {0}".format(results[0][0]['license_plate']['text']))
                             
            #my_label = customtkinter.CTkLabel(self,text="")
            #my_label.grid(row=2, column=1, padx=10, pady=15)
            
            #my_label = customtkinter.CTkLabel(self,text="License Plate Number: {0}".format(results[0][0]['license_plate']['text']))
            #my_label = customtkinter.CTkLabel(self, text="Bar code data: {0}\n Bar code type: {1}".format(barcode_data, barcode_type))
            #my_label.grid(row=2, column=1, padx=10, pady=15)
            #--------------
            
            my_image = customtkinter.CTkImage(light_image=Image.open('temp_image.png'), dark_image=Image.open('temp_image.png'), size=(450, 400))
                    
            my_label_image = customtkinter.CTkLabel(self, text="", image=my_image)
            my_label_image.grid(row=3, column=1, padx=10, pady=10)

        if input_type == 'folder':
            for filename in os.listdir(input_path):
                if filename.endswith(('.jpg', '.png', '.JPG', '.PNG')):
                    filepath = os.path.join(input_path, filename)
                    process_image(filepath)
        else:
            process_image(input_path)

        # # Save barcode_list to a CSV file
        # with open('barcode_data.csv', mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Barcode Data', 'Barcode Type'])  # Column headers
        #     writer.writerows(barcode_list)
        # print("Barcode data has been saved to barcode_data.csv")


def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]
    
    if detections == [] :
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]

    plate = [] 

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)
    
    if len(plate) != 0 : 
        return " ".join(plate), scores/len(plate)
    else :
        return " ".join(plate), 0


if __name__ == "__main__":
    app = App()
    app.mainloop()
