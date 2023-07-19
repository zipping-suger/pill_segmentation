import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import skimage.color


def normalize_list(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def rotate_image(image):
    # Get the height and width of the image
    h, w = image.shape[:2]

    # Define the desired size
    w_new = 400
    h_new = 400

    # Create a black image of the same size as w_new and h_new
    black = np.zeros((h_new, w_new, 3), dtype=np.uint8)

    # Copy the original image to the center of the black image
    x_offset = (w_new - w) // 2
    y_offset = (h_new - h) // 2
    black[y_offset:y_offset + h, x_offset:x_offset + w] = image

    image = black

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 1, 1, 0)

    # find largest contour
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # fit an ellipse to the contour and get the center, axes and angle
    (xc, yc), (d1, d2), angle = cv2.fitEllipse(big_contour)
    # create a rotation matrix using the center and angle
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    M_opp = cv2.getRotationMatrix2D((xc, yc), angle + 180, 1)

    # apply the rotation matrix to the image and crop it to the ellipse size
    rotated = cv2.warpAffine(image, M, (400, 400))
    rotated_opp = cv2.warpAffine(image, M_opp, (400, 400))

    # return the rotated image
    return rotated, rotated_opp


def batch_resize(height, width, data_path, output_dir):
    for image_path in os.listdir(data_path):
        image = cv2.imread(os.path.join(data_path, image_path))
        image = cv2.resize(image, (width, height))
        image_name = image_path.split('/')[-1]

        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image)


def compare_hist(hist1, hist2):
    # Chi-squared distance
    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-6))


# Create 3D color histogram
def get_3D_hsv_hist(image):
    # Convert to HSV color space
    hsv = skimage.color.rgb2hsv(image)
    # Create 3D color histogram
    hist, edges = np.histogramdd(hsv.reshape(-1, 3), bins=(8, 8, 8))
    # Normalize histogram
    hist /= hist.sum()
    return hist


# Calculate HoG feature
def get_hog_feature(query):
    # Compute the HOG features using 8x8 cells and 9 orientations
    hog = cv2.HOGDescriptor((64, 64), (8, 8), (8, 8), (8, 8), 9)
    query = cv2.resize(query, (64, 64))
    # Convert the query image to grayscale
    query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    # Compute the HOG features using 8x8 cells and 9 orientations
    query_features = hog.compute(query)
    # Flatten the features vector
    # query_features = query_features.ravel()
    query_features /= query_features.sum()
    return query_features


def get_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, db_mask = cv2.threshold(img, 1, 1, 0)
    contours, hierarchy = cv2.findContours(db_mask, 2, 1)
    cv2.drawContours(db_mask, contours, -1, color=(0, 0, 255), thickness=6)
    return db_mask


def get_size(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, db_mask = cv2.threshold(img, 1, 1, 0)
    return db_mask.sum()


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    fig = ax.figure
    fig.canvas.manager.full_screen_toggle()  # toggle fullscreen mode
    # ax.set_autoscale_on(False)
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
        ax.imshow(np.dstack((img, m * 0.5)))


def register_db_img():
    img_list = []
    for k in range(7):
        tmp = []
        for j in range(1):
            img, _ = rotate_image(cv2.imread('database/pill_db/p{}_{}.jpg'.format(k, j)))
            tmp.append(img)
        img_list.append(tmp)
    return img_list


def register_color_hist():
    img_hist_list = []
    for k in range(7):
        tmp = []
        for j in range(1):
            img, _ = rotate_image(cv2.imread('database/pill_db/p{}_{}.jpg'.format(k, j)))
            img_hist = get_3D_hsv_hist(img)
            tmp.append(img_hist)
        img_hist_list.append(tmp)
    return img_hist_list


def register_shape_db():
    shape_db = []
    for i in range(22):
        img = cv2.imread('database/shape/s{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
        ret, db_mask = cv2.threshold(img, 1, 1, 0)
        contours, hierarchy = cv2.findContours(db_mask, 2, 1)
        cv2.drawContours(db_mask, contours, -1, color=(0, 0, 255), thickness=6)
        shape_db.append(db_mask)
    return shape_db


def register_hog_db():
    pill_db = []
    # Compute the HOG features using 8x8 cells and 9 orientations
    hog = cv2.HOGDescriptor((64, 64), (8, 8), (8, 8), (8, 8), 9)
    for i in range(7):
        temp_l = []
        for j in range(5):
            img = cv2.imread('database/pill_db/p{}_{}.jpg'.format(i, j))
            img = cv2.resize(img, (64, 64))
            # Convert the image to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = hog.compute(img)
            # Flatten the features vector
            # features = features.ravel()
            features /= features.sum()
            temp_l.append(features)
        pill_db.append(temp_l)
    return pill_db


def register_color_db():
    color_db = []
    for i in range(8):
        temp_l = []
        for j in range(5):
            img = cv2.imread('database/pill_db/p{}_{}.jpg'.format(i, j))
            temp_l.append(get_3D_hsv_hist(img))
        color_db.append(temp_l)
    return color_db


# def register_color_db():
#     color_db = []
#     for i in range(5):
#         img = cv2.imread('database/color/c{}.jpg'.format(i))
#         color_db.append(get_3D_hsv_hist(img))
#     return color_db


def size_fileter(masks, min_area=5e2, max_area=1e4):
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    size_filtered_mask = [m for m in masks if max_area >= m['area'] >= min_area]
    return size_filtered_mask


def shape_filter(size_filtered_mask, shape_db, threshold=0.1):
    shape_score_masks = []
    for ann in size_filtered_mask:

        bbox = [int(x) for x in ann['bbox']]

        # directly use mask prediction
        m = ann['segmentation']
        cropped_mask = m[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]].copy()
        cropped_mask = cropped_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(cropped_mask.astype(np.uint8), 2, 1)
        cv2.drawContours(cropped_mask, contours, -1, color=(0, 0, 255), thickness=6)
        shape_score = 1
        for i in range(len(shape_db)):
            ret = cv2.matchShapes(cropped_mask, shape_db[i], 1, 0.0)
            # print(ret)
            shape_score = min(shape_score, ret)
        # print("--------")
        ann['shape_score'] = shape_score
        shape_score_masks.append(ann)
    shape_filtered_mask = [m for m in shape_score_masks if m['shape_score'] <= threshold]
    return shape_filtered_mask


def color_filter(size_filtered_mask, color_db, threshold):
    color_score_masks = []
    i = 0
    for ann in size_filtered_mask:
        img_cropped = cv2.imread('cuts_out/cuts_out_{}.jpg'.format(i))
        i += 1
        img_hist = get_3D_hsv_hist(img_cropped)

        # color_score_list = []
        # for j in range(len(color_db)):
        #     color_score_list.append(compare_hist(img_hist, color_db[j]))

        color_score_list = []
        for features_rot in color_db:
            tmp_list = []
            for features in features_rot:
                # Calculate the chi-square distance between the query and the image features
                tmp_list.append(compare_hist(img_hist, features))
                # Append the distance to the list
            color_score_list.append(min(tmp_list))

        color_score = min(color_score_list)
        color_index = color_score_list.index(color_score)
        ann['color_score'] = color_score
        ann['color_index'] = color_index
        color_score_masks.append(ann)
    # color_score_masks = [m for m in color_score_masks if m['color_score'] <= threshold]
    return color_score_masks


def hog_filter(size_filtered_mask, hog_db, threshold):
    hog_score_masks = []
    # Compute the HOG features using 8x8 cells and 9 orientations
    hog = cv2.HOGDescriptor((64, 64), (8, 8), (8, 8), (8, 8), 9)
    i = 0
    for ann in size_filtered_mask:
        query = cv2.imread('cuts_out/cuts_out_{}.jpg'.format(i))
        i += 1
        query = cv2.resize(query, (64, 64))
        # Convert the query image to grayscale
        query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        # Compute the HOG features using 8x8 cells and 9 orientations
        query_features = hog.compute(query)
        # Flatten the features vector
        # query_features = query_features.ravel()
        query_features /= query_features.sum()

        # Initialize a list to store the chi-square distances
        distances = []
        # Loop over the images in the database
        for features_rot in hog_db:
            chi_dist_list = []
            for features in features_rot:
                # Calculate the chi-square distance between the query and the image features
                chi_dist_list.append(compare_hist(query_features, features))
                # Append the distance to the list
            distances.append(min(chi_dist_list))
        hog_score = min(distances)
        hog_index = distances.index(hog_score)
        ann['hog_score'] = hog_score
        ann['hog_index'] = hog_index
        hog_score_masks.append(ann)
    # hog_score_masks = [m for m in hog_score_masks if m['hog_score'] <= threshold]
    return hog_score_masks


def pill_identify(size_filtered_mask, cost, img_hist_list, img_list):
    registered_masks = []
    i = 0
    for ann in size_filtered_mask:
        img_cropped = cv2.imread('cuts_out/cuts_out_{}.jpg'.format(i))
        i += 1
        temp1, temp2 = rotate_image(img_cropped)

        # get color histgram
        temp1_hist = get_3D_hsv_hist(temp1)
        temp2_hist = get_3D_hsv_hist(temp2)

        # get HoG feature
        temp1_hog = get_hog_feature(temp1)
        temp2_hog = get_hog_feature(temp2)

        # get mask for Hu momentum
        temp1_mask = get_mask(temp1)
        temp2_mask = get_mask(temp2)

        # get mask size
        temp1_size = get_size(temp1)

        # temp_ncc = []
        temp_color = []
        temp_hog = []
        temp_hu = []
        temp_size = []

        for k in range(7):
            ncc, color, hog, hu, size = 0, 0, 0, 0, 0
            for j in range(1):
                img = img_list[k][j]
                # st = time.time()
                # img, _ = rotate_image(cv2.imread('database/pill_db/p{}_{}.jpg'.format(k, j)))
                # et = time.time()
                # print("image preprocessing", et - st)

                # # NCC score
                # # st = time.time()
                # ncc_1 = cv2.matchTemplate(img, temp1, cv2.TM_CCORR_NORMED)
                # ncc_2 = cv2.matchTemplate(img, temp2, cv2.TM_CCORR_NORMED)
                # ncc += (1 - max(ncc_1, ncc_2)[0][0])
                # # et = time.time()
                # # print("NCC", et - st)

                # Color histgram difference (normarlized)
                st = time.time()
                img_hist = img_hist_list[k][j]
                color_1 = compare_hist(img_hist, temp1_hist)
                color_2 = compare_hist(img_hist, temp2_hist)
                color += (min(color_1, color_2))
                et = time.time()
                print("Color", et - st)

                # HoG
                # st = time.time()
                img_hog = get_hog_feature(img)
                hog_1 = compare_hist(img_hog, temp1_hog)
                hog_2 = compare_hist(img_hog, temp2_hog)
                hog += (min(hog_1, hog_2))
                # et = time.time()
                # print("HoG", et-st)

                # Hu momentum
                # st = time.time()
                img_mask = get_mask(img)
                hu_1 = cv2.matchShapes(img_mask, temp1_mask, 1, 0.0)
                hu_2 = cv2.matchShapes(img_mask, temp2_mask, 1, 0.0)
                hu += (min(hu_1, hu_2))
                # et = time.time()
                # print("Hu", et - st)

                # Size difference
                # st = time.time()
                img_size = get_size(img)
                size_diff = abs(int(img_size) - int(temp1_size))
                size += (size_diff)
                # et = time.time()
                # print("Size", et - st)

            # temp_ncc.append(ncc)
            temp_color.append(color)
            temp_hog.append(hog)
            temp_hu.append(hu)
            temp_size.append(size)

        # score_list = normalize_list(temp_ncc) * cost[0] + normalize_list(temp_color) * cost[1] + normalize_list(temp_hog) * cost[2] + normalize_list(temp_hu) * cost[3] + normalize_list(temp_size) * cost[4]
        score_list = normalize_list(temp_color) * cost[0] + normalize_list(
            temp_hog) * cost[1] + normalize_list(temp_hu) * cost[2] + normalize_list(temp_size) * cost[3]
        # print(score_list)
        score_list = score_list.tolist()

        score = min(score_list)
        index = score_list.index(score)
        ann['score'] = score
        ann['index'] = index
        registered_masks.append(ann)
    return registered_masks


# def color_filter(img, size_filtered_mask, color_db, threshold=1):
#     color_score_masks = []
#     for ann in size_filtered_mask:
#         # directly use mask prediction
#         m = ann['segmentation']
#         # Apply the mask to the image
#         img_masked = cv2.bitwise_and(img, img, mask=m.astype(np.uint8))
#         # Find the bounding box of the mask
#         x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
#         # Crop the image to the bounding box
#         img_cropped = img_masked[y:y + h, x:x + w]
#         # # showing the image
#         cv2.imshow('cropped_img', img_cropped)
#         # # waiting using waitKey method
#         cv2.waitKey(0)
#         img_hist = get_3D_hsv_hist(img_cropped)
#         color_score = 10
#         for i in range(len(color_db)):
#             ret = compare_hist(img_hist, color_db[i])
#             print(ret)
#             color_score = min(color_score, ret)
#         print("--------")
#         ann['color_score'] = color_score
#         color_score_masks.append(ann)
#     # color_score_masks = [m for m in color_score_masks if m['color_score'] <= threshold]
#     return color_score_masks

# Define a function to create a color list
def create_color_list(length=15):
    # Create a colormap object
    cmap = plt.get_cmap("tab10")

    # Get the number of colors in the colormap
    ncolors = cmap.N

    # Create an empty list to store the colors
    color_list = []

    # Loop through the length of the list
    for i in range(length):
        # Get the color at the index modulo the number of colors
        color = cmap(i % ncolors)
        # Convert the color to integers
        color = tuple(int(c * 255) for c in color)
        # Slice the color tuple to get only the first three elements
        color = color[:3]
        # Append the color to the list
        color_list.append(color)

    # Return the color list
    return color_list


def visualize_sam(img, anns, color_list):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    pill_id = 0
    for ann in sorted_anns:
        # color = np.random.randint(0, 256, size=3).tolist()

        # Draw the score and id on the image
        pill_index = ann['index']
        #         text = f"score: {score:.2f}"
        pill_type = {0: "Seirogan", 1: "Avandaryl", 2: "Seconal", 3: "omega-3 capsule", 4: "Sanlietong",
                     5: "Multivitamin", 6: "New_type", 7: "Silver"}
        text = "{}".format(pill_type[pill_index])
        pill_id += 1

        color = color_list[pill_index]

        # draw mask
        m = ann['segmentation']
        # Draw the mask on the image with some transparency
        img[m == 1] = img[m == 1] * 0.3 + np.array(color) * 0.7
        # Find the contours of the mask
        contours, hierarchy = cv2.findContours(m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw the contours in white color
        cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=6)

        # Draw the bounding box on the image
        bbx = ann['bbox']
        x1, y1, x2, y2 = [int(x) for x in bbx]
        cv2.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), color=color, thickness=2)

        # Get the text size and baseline
        (text_width, text_height), baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                              fontScale=0.4, thickness=1)
        # Define the rectangle coordinates
        rect_x1 = x1 + 5
        rect_y1 = y1 + 5
        rect_x2 = rect_x1 + text_width
        rect_y2 = rect_y1 + text_height + baseline
        # Create a copy of the original image for drawing the rectangle
        rect_img = img.copy()
        # Draw the rectangle on the copy
        cv2.rectangle(rect_img, (x1, y1), (rect_x2, rect_y2), color=color, thickness=-1)
        # Blend the copy with the original image with some transparency
        alpha = 0.5  # Adjust this value to change the transparency level
        img = cv2.addWeighted(rect_img, alpha, img, 1 - alpha, 0)
        # Draw the text on top of the blended image
        cv2.putText(img, text, (rect_x1, rect_y2 - baseline), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(255, 255, 255), thickness=1)

        ax.imshow(img)


def cuts_out(img, anns, save_dir):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # Save the cropped image in the save directory
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    # Clear the directory
    for f in os.listdir(save_dir):
        if f.endswith(".jpg"):  # delete files with .jpg extension
            os.remove(os.path.join(save_dir, f))
    for i, ann in enumerate(sorted_anns):
        #     for ann in sorted_anns:
        m = ann['segmentation']

        # Apply the mask to the image
        img_masked = cv2.bitwise_and(img, img, mask=m.astype(np.uint8))

        # Find the bounding box of the mask
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))

        # Crop the image to the bounding box
        img_cropped = img_masked[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(save_dir, 'cuts_out_{}.jpg'.format(i)), img_cropped)
    # print("Done")


if "__name__" == "__main__":
    raw_image_path = './dataset/pills'
    output_path = './dataset/pills_resized_256_192'

    batch_resize(256, 192, raw_image_path, output_path)
