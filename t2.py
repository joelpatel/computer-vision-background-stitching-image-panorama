# 1. Only add your code inside the function (including newly improted packages).
#  You can design a new function and call the new function in the given functions.
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


# For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
def stitch(imgmark, N=5, savepath=''):
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    from math import ceil, floor, sqrt
    overlap_arr = np.zeros((N, N), int)
    threshold_1 = 0.4  # ratio threshold for finding unique correspondences

    def resize_img(img):
        if len(img) > 1000 or len(img[0]) > 1000:
            (h, w) = img.shape[:2]
            new_h = int(h/w * 1000)
            new_w = 1000
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    imgs = [resize_img(image) for image in imgs]

    def exhaustive_overlap_finding(images, overlap):
        np.fill_diagonal(overlap, 1)
        for i_c in range(len(images)):
            for j in range(i_c, len(images)):
                if i_c == j:
                    continue
                # img i img j find sift matches and find assign values accordingly
                img1 = images[i_c]
                img2 = images[j]

                gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                sift = cv2.xfeatures2d.SIFT_create(
                    nfeatures=400)  # try different values

                keypoint1, descriptor1 = sift.detectAndCompute(gray_img1, None)
                keypoint2, descriptor2 = sift.detectAndCompute(gray_img2, None)

                matches_directional = []

                for des_index_1 in range(len(descriptor1)):
                    key_des = list()
                    for des_index_2 in range(len(descriptor2)):
                        ssd = np.sum((descriptor1[des_index_1][:] -
                                      descriptor2[des_index_2][:])**2)
                        key_des.append([des_index_2, ssd])
                    key_des = sorted(key_des, key=lambda l: l[1])

                    ratio = key_des[0][1] / key_des[1][1]
                    if ratio < threshold_1:
                        matches_directional.append([(keypoint1[des_index_1]).pt,
                                                    (keypoint2[key_des[0][0]]).pt])

                if len(matches_directional) < 20:
                    continue
                else:
                    overlap[i_c][j] = 1
                    overlap[j][i_c] = 1

        return overlap

    overlap_arr = exhaustive_overlap_finding(imgs, overlap_arr)

    starting = True

    img1 = imgs[0]
    img2 = imgs[1]
    considered = []

    for i_c in range(len(overlap_arr)):
        for j in range(len(overlap_arr[0])):
            if i_c == j or j in considered:
                continue

            if overlap_arr[i_c][j] == 1:
                if starting:
                    starting = False
                    img1 = imgs[i_c]
                    img2 = imgs[j]
                    considered.append(i_c)
                    considered.append(j)
                    # print(f"{i_c}\n{j}")
                else:
                    img1 = resize_img(img1)
                    img2 = imgs[j]
                    considered.append(j)
                    # print(f"{j}")
                    img2 = resize_img(img2)

                # steps to stitch images
                gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                sift = cv2.xfeatures2d.SIFT_create(
                    nfeatures=1000)  # try different values

                keypoint1, descriptor1 = sift.detectAndCompute(gray_img1, None)
                keypoint2, descriptor2 = sift.detectAndCompute(gray_img2, None)

                # matches = []
                matches_directional = []

                for des_index_1 in range(len(descriptor1)):
                    key_des = list()
                    for des_index_2 in range(len(descriptor2)):
                        ssd = np.sum((descriptor1[des_index_1][:] -
                                      descriptor2[des_index_2][:])**2)
                        key_des.append([des_index_2, ssd])
                    key_des = sorted(key_des, key=lambda l: l[1])

                    ratio = key_des[0][1] / key_des[1][1]
                    if ratio < 0.6:
                        matches_directional.append([(keypoint1[des_index_1]).pt,
                                                    (keypoint2[key_des[0][0]]).pt])

                if len(matches_directional) < 20:
                    print("bad match...\nSHOULD NOT HAPPEN")
                    continue

                """
                    NOTE: My implementation of something like a warp perspective. Slower than the opencv function. 
                        However professor initially adviced me not to use warpPerspective function in a Piazza post.
                        Post number: @357_f1
                        Post link: https://piazza.com/class/kyezjpl4zxb3k6?cid=357
                """

                pts_1 = np.asarray([i_m[0] for i_m in matches_directional])
                pts_2 = np.asarray([i_m[1] for i_m in matches_directional])

                H, status = cv2.findHomography(pts_2, pts_1, cv2.RANSAC, 4.0)

                top_left = np.mat([[0], [0], [1]])
                top_left_transformed = (np.dot(H, top_left)).tolist()
                top_left_transformed = [(el[0]/top_left_transformed[2][0])
                                        for el in top_left_transformed]

                bottom_left = np.mat([[0], [len(img2)-1], [1]])
                bottom_left_transformed = (np.dot(H, bottom_left)).tolist()
                bottom_left_transformed = [(el[0]/bottom_left_transformed[2][0])
                                           for el in bottom_left_transformed]

                top_right = np.mat([[len(img2[0])-1], [0], [1]])
                top_right_transformed = (np.dot(H, top_right)).tolist()
                top_right_transformed = [(el[0]/top_right_transformed[2][0])
                                         for el in top_right_transformed]

                bottom_right = np.mat([[len(img2[0])-1], [len(img2)-1], [1]])
                bottom_right_transformed = (np.dot(H, bottom_right)).tolist()
                bottom_right_transformed = [(el[0]/bottom_right_transformed[2][0])
                                            for el in bottom_right_transformed]

                edges_transformed = [top_left_transformed, bottom_left_transformed,
                                     top_right_transformed, bottom_right_transformed]

                temp1 = [i_et[0] for i_et in edges_transformed]
                temp2 = [i_et[1] for i_et in edges_transformed]
                min_x = round(min(temp1))
                max_x = round(max(temp1))
                min_y = round(min(temp2))
                max_y = round(max(temp2))

                H_inv = np.linalg.inv(H)

                col_size = max_x-min_x + 1
                row_size = max_y-min_y + 1

                col_size = max_x-min_x + 1
                row_size = max_y-min_y + 1

                final_min_x = min_x if min_x < 0 else 0
                final_max_x = max_x if max_x > len(
                    img1[0]) - 1 else len(img1[0]) - 1
                final_min_y = min_y if min_y < 0 else 0
                final_max_y = max_y if max_y > len(img1) - 1 else len(img1) - 1

                final_col_size = (final_max_x - final_min_x + 1)
                final_row_size = (final_max_y - final_min_y + 1)

                blank_image = np.zeros(
                    (final_row_size, final_col_size, 3), np.uint8)
                blend_m = np.zeros((final_row_size, final_col_size))

                i_loop = 0
                for row_index in range(final_min_y, final_max_y + 1):
                    j = 0
                    for col_index in range(final_min_x, final_max_x + 1):
                        v = np.array([[col_index, row_index, 1.0]]).T
                        result_coord = np.dot(H_inv, v).tolist()
                        result_coord = [(el[0]/result_coord[2][0])
                                        for el in result_coord]
                        r_no = result_coord[1]
                        # [r_no, c_no] is in image_2 space
                        c_no = result_coord[0]
                        if 0 < r_no and r_no < len(img2) - 1 and 0 < c_no and c_no < len(img2[0]) - 1:
                            # here get result of bilinear interpolation into the image_ matrix
                            # coordinates of image_ will be [i, j]
                            b = r_no - floor(r_no)
                            a = c_no - floor(c_no)
                            zero_channel = ((1 - a) * (1 - b) * img2[floor(r_no)][floor(c_no)][0]
                                            + a*(1 - b) *
                                            img2[floor(r_no)][floor(
                                                c_no) + 1][0]
                                            + a * b *
                                            img2[floor(r_no) +
                                                 1][floor(c_no) + 1][0]
                                            + (1 - a) * b * img2[floor(r_no) + 1][floor(c_no)][0])
                            # zero_channel = 0 if zero_channel <= 0 else zero_channel
                            one_channel = ((1 - a) * (1 - b) * img2[floor(r_no)][floor(c_no)][1]
                                           + a*(1 - b) *
                                           img2[floor(r_no)][floor(
                                               c_no) + 1][1]
                                           + a * b *
                                           img2[floor(r_no) +
                                                1][floor(c_no) + 1][1]
                                           + (1 - a) * b * img2[floor(r_no) + 1][floor(c_no)][1])
                            # one_channel = 0 if one_channel <= 0 else one_channel
                            two_channel = ((1 - a) * (1 - b) * img2[floor(r_no)][floor(c_no)][2]
                                           + a*(1 - b) * img2[floor(r_no)
                                                              ][floor(c_no) + 1][2]
                                           + a * b *
                                           img2[floor(r_no) +
                                                1][floor(c_no) + 1][2]
                                           + (1 - a) * b * img2[floor(r_no) + 1][floor(c_no)][2])
                            # two_channel = 0 if two_channel <= 0 else two_channel
                            if len(img2) >= 80:
                                left = r_no - 0
                                right = len(img2) - 1 - r_no
                                value = 0
                                if (left < right):
                                    value = left/40 + 1/40 if left < 40 and value < 1 else 1
                                else:
                                    value = right/40 + 1/40 if right < 40 and value < 1 else 1
                                up = c_no - 0
                                down = len(img2[0]) - 1 - c_no
                                v_value = 0
                                if up < down:
                                    v_value = up/40 + 1/40 if up < 40 and v_value < 1 else 1
                                else:
                                    v_value = down/40 + 1/40 if down < 40 and v_value < 1 else 1
                                final_value = v_value if v_value < value else value
                                blend_m[i_loop][j] = final_value
                            else:
                                blend_m[i_loop][j] = 1
                            blank_image[i_loop][j][0] = zero_channel
                            blank_image[i_loop][j][1] = one_channel
                            blank_image[i_loop][j][2] = two_channel
                        j += 1
                    i_loop += 1

                min_x = min_x if min_x < 0 else 0
                min_y = min_y if min_y < 0 else 0
                transformed_img = blank_image.copy()
                blank_image[0-min_y:img1.shape[0]-min_y,
                            0-min_x:img1.shape[1]-min_x] = img1

                # print("blending...")
                for row_index in range(len(blank_image)):
                    for col_index in range(len(blank_image[0])):
                        t_b = transformed_img[row_index][col_index][0]
                        t_g = transformed_img[row_index][col_index][1]
                        t_r = transformed_img[row_index][col_index][2]
                        if (t_b == 0 and t_g == 0 and t_r == 0):
                            pass
                        else:  # following pixels will have value
                            # check if blank image has pixel in it
                            # if not then following code
                            # if blank_image[row_index][col_index][0] == 0 and blank_image[row_index][col_index][1] == 0 and blank_image[row_index][col_index][2] == 0:
                            if blank_image[row_index][col_index][0] <= 120 and blank_image[row_index][col_index][1] <= 120 and blank_image[row_index][col_index][2] <= 120:
                                blank_image[row_index][col_index][0] = t_b
                                blank_image[row_index][col_index][1] = t_g
                                blank_image[row_index][col_index][2] = t_r
                            # if so then
                            else:
                                alpha = blend_m[row_index][col_index]
                                if alpha != 1 and alpha != 0:
                                    one_minus_alpha = 1 - alpha
                                    blank_image[row_index][col_index][0] = (
                                        t_b)*alpha + one_minus_alpha*blank_image[row_index][col_index][0]
                                    blank_image[row_index][col_index][1] = (
                                        t_g)*alpha + one_minus_alpha*blank_image[row_index][col_index][1]
                                    blank_image[row_index][col_index][2] = (
                                        t_r)*alpha + one_minus_alpha*blank_image[row_index][col_index][2]
                                else:
                                    blank_image[row_index][col_index][0] = t_b
                                    blank_image[row_index][col_index][1] = t_g
                                    blank_image[row_index][col_index][2] = t_r

                img1 = blank_image

    cv2.imwrite(savepath, img1)
    return overlap_arr


if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
