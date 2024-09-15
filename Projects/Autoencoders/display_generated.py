import os
import cv2


image_folders_path = 'new_results'

for file in os.listdir(image_folders_path):
    img_path = os.path.join(image_folders_path,file)
    img = cv2.imread(img_path)
    img = cv2.resize(img,(512,512))
    cv2.imshow('generated images',img)
    k = cv2.waitKey(80) & 0xff
    if k == 'q':
        break