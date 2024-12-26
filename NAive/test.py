import cv2
import numpy as np

path_img = "1.png"

img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("gada gambar cok!")
else:
    print("Array gambar sebelum normalisasi:")
    print(img)
    gambarNormal = img / 255.0
    print("\nArray gambar setelah normalisasi:")
    print(gambarNormal)
    cv2.imshow('Gambar', img)
    cv2.waitKey()