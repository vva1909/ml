import cv2
import matplotlib.pyplot as plt
import numpy as np

def display_image(image_path):
    img = cv2.imread(image_path)

    # Create a figure with 2 subplots side by side
    plt.figure(figsize=(10,5))

    # First subplot for BGR image
    plt.subplot(2,3,1)
    plt.title('BGR Image')
    plt.imshow(img)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Second subplot for RGB image 
    plt.subplot(2,3,2)
    plt.title('RGB Image')
    plt.imshow(img_rgb)

    red = img_rgb.copy()
    red[:,:,1] = 0
    red[:,:,2] = 0

    green = img_rgb.copy()
    green[:,:,0] = 0
    green[:,:,2] = 0

    blue = img_rgb.copy()
    blue[:,:,0] = 0
    blue[:,:,1] = 0

    plt.subplot(2,3,3)
    plt.title('Red')
    plt.imshow(red)

    plt.subplot(2,3,4)
    plt.title('Green')
    plt.imshow(green)   

    plt.subplot(2,3,5)
    plt.title('Blue')
    plt.imshow(blue)

    plt.tight_layout()
    plt.show()


img = np.zeros([512,512,3], np.uint8)
plt.imshow(img)
plt.show()
