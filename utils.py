import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def batch_resize(height, width, data_path, output_dir):
    for image_path in os.listdir(data_path):
        image = cv2.imread(os.path.join(data_path, image_path))
        image = cv2.resize(image, (width, height))
        image_name = image_path.split('/')[-1]

        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        # Find the contours of the mask
        contours, hierarchy = cv2.findContours(m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw the contours in white color
        cv2.drawContours(img, contours, -1, (0, 0, 1), 6)
        ax.imshow(np.dstack((img, m * 0.6)))


def cuts_out(img, anns, save_dir):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for i, ann in enumerate(sorted_anns):
        #     for ann in sorted_anns:
        m = ann['segmentation']

        # Apply the mask to the image
        img_masked = cv2.bitwise_and(img, img, mask=m.astype(np.uint8))

        # Find the bounding box of the mask
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))

        # Crop the image to the bounding box
        img_cropped = img_masked[y:y + h, x:x + w]

        # Save the cropped image in the save directory
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)
        cv2.imwrite(os.path.join(save_dir, 'cuts_out_{}.jpg'.format(i)), img_cropped)
    print("Done")
