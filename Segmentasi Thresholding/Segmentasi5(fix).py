import cv2
import numpy as np
import pandas as pd
import os

image_folder = "proyek/"
output_csv = 'data/detected_colors005.csv'

results = []

for filename in os.listdir(image_folder):
    if filename.endswith(('.jpeg', '.jpg', '.png')): 
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (500, 700))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_helm = np.array([0, 0, 200])
        upper_helm = np.array([180, 30, 255])
        lower_vest = np.array([30, 50, 50])
        upper_vest = np.array([40, 255, 255])
        

        mask_helm = cv2.inRange(hsv, lower_helm, upper_helm)
        mask_vest = cv2.inRange(hsv, lower_vest, upper_vest)

        contours_helm, _ = cv2.findContours(mask_helm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_vest, _ = cv2.findContours(mask_vest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        helm_detected = False
        vest_detected = False

        r_helm, g_helm, b_helm = 0, 0, 0
        r_vest, g_vest, b_vest = 0, 0, 0

        for contour in contours_helm:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                helm_detected = True
                
                helmet_region = img[y:y+h, x:x+w]
                avg_color_per_row = np.average(helmet_region, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                r_helm, g_helm, b_helm = avg_color[2], avg_color[1], avg_color[0]

      
        for contour in contours_vest:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                vest_detected = True
                
                # Calculate average RGB for vest
                vest_region = img[y:y+h, x:x+w]
                avg_color_per_row = np.average(vest_region, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                r_vest, g_vest, b_vest = avg_color[2], avg_color[1], avg_color[0]

        if not helm_detected:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
              
            for (x, y, w, h) in faces:
                head_top = y - int(h * 0.5)
                if head_top < 0:
                    head_top = 0
                helmet_region = img[head_top:y, x:x+w]
                if helmet_region.size > 0:
                    avg_color_per_row = np.average(helmet_region, axis=0)
                    avg_color = np.average(avg_color_per_row, axis=0)
                    r_helm, g_helm, b_helm = avg_color[2], avg_color[1], avg_color[0]
                else:
                    r_helm, g_helm, b_helm = 0, 0, 0
                # cv2.rectangle(img, (x, head_top), (x + w, y), (0, 0, 255), 2)
        if not vest_detected:
            lower_shirt = np.array([0, 50, 50])
            upper_shirt = np.array([180, 255, 255])
    
            mask_shirt = cv2.inRange(hsv, lower_shirt, upper_shirt)
            contours_shirt, _ = cv2.findContours(mask_shirt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
            for contour in contours_shirt:
                if cv2.contourArea(contour) > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    vest_region = img[y:y+h, x:x+w]
                    if vest_region.size > 0:
                        avg_color_per_row = np.average(vest_region, axis=0)
                        avg_color = np.average(avg_color_per_row, axis=0)
                        r_vest, g_vest, b_vest = avg_color[2], avg_color[1], avg_color[0]
                    else:
                        r_vest, g_vest, b_vest = 0, 0, 0
                    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

        
        # Tentukan label
        if helm_detected and vest_detected:
            label = "Lengkap"
        else:
            
            label = "Tidak Lengkap"

        # Prepare data for CSV
        results.append({
            'r_helm': r_helm,
            'g_helm': g_helm,
            'b_helm': b_helm,
            'r_vest': r_vest,
            'g_vest': g_vest,
            'b_vest': b_vest,
            'label': label  # Tambahkan label ke dalam hasil
        })

# Create a DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print(f"Hasil rata-rata warna telah disimpan ke dalam '{output_csv}'")
