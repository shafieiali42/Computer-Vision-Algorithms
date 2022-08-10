import random
import time

import cv2 as cv
import numpy as np


def find_homography(image2, image1):
    scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    sift = cv.SIFT_create()
    image1 = cv.resize(image1, (0, 0), image1, 0.5, 0.5, interpolation=cv.INTER_AREA)
    image2 = cv.resize(image2, (0, 0), image2, 0.5, 0.5, interpolation=cv.INTER_AREA)
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    # images_with_interest_points = cv.drawMatches(image1, kp1, image2, kp2, None, None, None,
    #                                              singlePointColor=(0, 255, 0))
    # save_image(images_with_interest_points, "interest_points")

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # image_for_corresponding_points = np.copy(images_with_interest_points)
    # for match in good_matches:
    #     cv.drawMatches(image1, [kp1[match.queryIdx]], image2, [kp2[match.trainIdx]], None,
    #                    image_for_corresponding_points
    #                    , singlePointColor=(255, 0, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    # save_image(image_for_corresponding_points, "corresponding")

    # image_for_match_points = image_for_corresponding_points.copy()
    # cv.drawMatches(image1, kp1, image2, kp2, good_matches, image_for_match_points
    #                , matchColor=(255, 0, 0),
    #                flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS + cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    # save_image(image_for_match_points, "res15_matches")

    # random_good_match = random.sample(good_matches, 20)
    # image_for_20_match_points = image_for_corresponding_points.copy()
    # cv.drawMatches(image1, kp1, image2, kp2, random_good_match, image_for_20_match_points
    #                , matchColor=(255, 0, 0),
    #                flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    #
    # save_image(image_for_20_match_points, "res16")
    #
    image1_points = []
    image2_points = []
    for match in good_matches:
        image1_points.append(kp1[match.queryIdx].pt)
        image2_points.append(kp2[match.trainIdx].pt)

    image1_points = np.array(image1_points, dtype="float64")
    image2_points = np.array(image2_points, dtype="float64")
    homography, mask = cv.findHomography(image2_points, image1_points, method=cv.RANSAC, ransacReprojThreshold=5,
                                         maxIters=1000)

    image_for_inliers = cv.drawMatches(image1, None, image2, None, None, None)
    # for counter, match in enumerate(good_matches):
    #     if mask[counter] == 1:
    #         cv.drawMatches(image1, [kp1[match.queryIdx]], image2, [kp2[match.trainIdx]], None,
    #                        image_for_inliers
    #                        , singlePointColor=(0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    # save_image(image_for_inliers, "res17")
    inverse_scale = np.linalg.inv(scale_matrix)

    return np.matmul(inverse_scale, np.matmul(homography, scale_matrix))
    # return homography


def my_clip(image, cast):
    image = image.astype("float64")
    if np.min(image) == np.max(image):
        if np.ndim(image) == 3:
            image[:, :, :] = 0
        else:
            image[:, :] = 0
    image = image - np.min(image)
    image = (255 / (np.max(image) - np.min(image))) * image
    if cast:
        image = image.astype("uint8")
    return image


def save_image(image, name):
    image = my_clip(image, True)
    image = image.astype("uint8")
    cv.imwrite(f"{name}.jpg", image)


# def read_resized_frames(file_name):
#     cap = cv.VideoCapture(f"{file_name}.mp4")
#     frames = []
#     frames.append(0)
#
#     i = 1
#     while (cap.isOpened() and i <= 900):
#         ret, frame = cap.read()
#         if ret == False:
#             break
#         frame_copy = np.copy(frame)
#         frame_copy = cv.resize(frame_copy, (int(frame_copy.shape[1] / 5), int(frame_copy.shape[0] / 5)), frame_copy,
#                                interpolation=cv.INTER_AREA)
#         frames.append(frame_copy)
#
#         i = i + 1
#
#     cap.release()
#     cv.destroyAllWindows()
#     # return frames,reference, frame_90,frame270,frame_630,frame_810
#     return frames
#

def find_size_of_total_frame(indexes, reference_frame):
    points = []
    src_points = np.array([[0, 0], [0, reference_frame.shape[0]], [reference_frame.shape[1], reference_frame.shape[0]],
         [reference_frame.shape[1], 0]],dtype="float32")
    counter1 = 0
    for index in indexes:
        if counter1 == 0:
            counter1 = counter1 + 1
            continue
        dst = cv.perspectiveTransform(src_points.reshape(-1, 1, 2), all_homographies[index])
        dst = dst.reshape(src_points.shape)
        # print(dst.shape)
        # print(dst)
        points.append(dst[0])
        points.append(dst[1])
        points.append(dst[2])
        points.append(dst[3])
        counter1 = counter1 + 1
    min = np.min(points, axis=0)
    max = np.max(points, axis=0)
    size = max-min
    size=size.astype("int64")
    translation = np.array([[1, 0, -min[0]], [0, 1, -min[1]], [0, 0, 1]])
    return translation, size


def read_all_frames(file_name):
    cap = cv.VideoCapture(f"{file_name}.mp4")
    frames = []
    frames.append(0)

    i = 1
    while (cap.isOpened() and i <= 900):
        ret, frame = cap.read()
        if ret == False:
            break

        frames.append(frame)
        i = i + 1

    cap.release()
    cv.destroyAllWindows()
    return frames


# def get_frame_by_index(file_name, index):
#     cap = cv.VideoCapture(f"{file_name}.mp4")
#     frames = []
#     frames.append(0)
#
#     i = 1
#     while (cap.isOpened() and i <= 900):
#         ret, frame = cap.read()
#         if ret == False:
#             break
#
#         if i == index:
#             return frame
#         i = i + 1
#
#     cap.release()
#     cv.destroyAllWindows()


# def read_frames(file_name):
#     cap = cv.VideoCapture(f"{file_name}.mp4")
#     frames = []
#     frames.append(0)
#
#     i = 1
#     while (cap.isOpened() and i <= 900):
#         ret, frame = cap.read()
#         if ret == False:
#             break
#         if i == 450:
#             reference = frame
#         if i == 270:
#             frame270 = frame
#         if i == 90:
#             frame_90 = frame
#         if i == 630:
#             frame_630 = frame
#         if i == 810:
#             frame_810 = frame
#
#         frames.append(frame)
# i = i + 1
#
# cap.release()
# cv.destroyAllWindows()
# return reference, frame_90, frame270, frame_630, frame_810


def draw_rect_on_image(image):
    image_copy = np.copy(image)
    cv.rectangle(image_copy, (375, 410), (774, 933), color=(0, 0, 255), thickness=5)
    save_image(image_copy, "res01-450-rect")
    return image_copy


def transform_rect(image, homography_inverse):
    p1 = cv.perspectiveTransform(np.float32([[[375, 410]]]), homography_inverse)
    p2 = cv.perspectiveTransform(np.float32([[[375, 933]]]), homography_inverse)
    p3 = cv.perspectiveTransform(np.float32([[[774, 933]]]), homography_inverse)
    p4 = cv.perspectiveTransform(np.float32([[[774, 410]]]), homography_inverse)
    p1 = p1.astype("int32")
    p2 = p2.astype("int32")
    p3 = p3.astype("int32")
    p4 = p4.astype("int32")
    pts = np.array([p1, p2, p3, p4])
    pts = pts.reshape((-1, 1, 2))
    image2_with_rect = np.copy(image)
    cv.drawMarker(image2_with_rect, (p1[0][0][0], p1[0][0][1]), color=(0, 0, 255), thickness=8)
    cv.drawMarker(image2_with_rect, (p2[0][0][0], p2[0][0][1]), color=(0, 0, 255), thickness=8)
    cv.drawMarker(image2_with_rect, (p3[0][0][0], p3[0][0][1]), color=(0, 0, 255), thickness=8)
    cv.drawMarker(image2_with_rect, (p4[0][0][0], p4[0][0][1]), color=(0, 0, 255), thickness=8)
    cv.polylines(image2_with_rect, [pts], isClosed=True, color=(0, 255, 0))
    save_image(image2_with_rect, "res02-270-rect")


def warp_and_translate(image, homography_matrix, size):
    p1_image2 = cv.perspectiveTransform(np.float32([[[0, 0]]]), homography_matrix)
    p2_image2 = cv.perspectiveTransform(np.float32([[[image.shape[1], 0]]]), homography_matrix)
    p3_image2 = cv.perspectiveTransform(np.float32([[[image.shape[1], frame270.shape[0]]]]), homography_matrix)
    p4_image2 = cv.perspectiveTransform(np.float32([[[0, image.shape[0]]]]), homography_matrix)
    # print(p4_image2)
    min_x = min(p1_image2[0][0][0], p2_image2[0][0][0], p3_image2[0][0][0], p4_image2[0][0][0])
    max_x = max(p1_image2[0][0][0], p2_image2[0][0][0], p3_image2[0][0][0], p4_image2[0][0][0])
    min_y = min(p1_image2[0][0][1], p2_image2[0][0][1], p3_image2[0][0][1], p4_image2[0][0][1])
    max_y = max(p1_image2[0][0][1], p2_image2[0][0][1], p3_image2[0][0][1], p4_image2[0][0][1])
    m_translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    matrix = np.matmul(m_translation, homography_matrix)
    warped = cv.warpPerspective(image, matrix, size)
    return warped, m_translation, int(min_x), int(min_y)


current_time = time.time()

all_frames = read_all_frames("video")
all_homographies = np.zeros((901, 3, 3))
reference = all_frames[450]
frame_90 = all_frames[90]
frame270 = all_frames[270]
frame_630 = all_frames[630]
frame_810 = all_frames[810]

all_homographies[450] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
all_homographies[270] = find_homography(frame270, reference)
all_homographies[630] = find_homography(frame_630, reference)
homography_90_270 = find_homography(frame_90, frame270)
homography_810_630 = find_homography(frame_810, frame_630)
all_homographies[90] = np.matmul(homography_90_270, all_homographies[270])
all_homographies[810] = np.matmul(homography_810_630, all_homographies[630])

homography270_inverse = np.linalg.inv(all_homographies[270])
rect_frame = draw_rect_on_image(reference)
transform_rect(frame270, homography270_inverse)
size_part1 = (reference.shape[1] + frame270.shape[1], reference.shape[0] + frame270.shape[0])
warped, m_translation, min_x, min_y = warp_and_translate(frame270, all_homographies[270], size_part1)
warped[-min_y:-min_y + reference.shape[0], -min_x:-min_x + reference.shape[1]] = reference
indexes = np.where(warped != 0)
x_indexes = indexes[0]
y_indexes = indexes[1]
warped = warped[:np.max(x_indexes), :np.max(y_indexes)]
save_image(warped, "res03-270-450-panorama")

print(time.time() - current_time)
current_time = time.time()
print("part1 end.")

print("before all homo")

for i in range(1, 181):
    if i == 90:
        continue
    all_homographies[i] = np.matmul(find_homography(all_frames[i], all_frames[90]), all_homographies[90])
    # print(i)

print("middle all homo")
for i in range(181, 361):
    if i == 270:
        continue
    all_homographies[i] = np.matmul(find_homography(all_frames[i], all_frames[270]), all_homographies[270])

for i in range(361, 541):
    if i == 450:
        continue
    all_homographies[i] = find_homography(all_frames[i], all_frames[450])

for i in range(541, 721):
    if i == 630:
        continue
    all_homographies[i] = np.matmul(find_homography(all_frames[i], all_frames[630]), all_homographies[630])

for i in range(721, 901):
    if i == 810:
        continue
    all_homographies[i] = np.matmul(find_homography(all_frames[i], all_frames[810]), all_homographies[810])

print("end all homo")

translation, size = find_size_of_total_frame(range(0, 901), reference)
size = (size[0], size[1])

video_writer = cv.VideoWriter('res05-reference-plane.mp4', cv.VideoWriter_fourcc(*'MP4V'), 30, size)
i = 0

for homo in all_homographies:
    if i == 0:
        i = i + 1
        continue
    video_writer.write(cv.warpPerspective(all_frames[i], np.matmul(translation, homo), size))
    i = i + 1
    # print(i)
video_writer.release()
#
# print("Done")


print(time.time() - current_time)
current_time = time.time()
print("part3 end.")
print("part2 start....")


def verticalMinimumCostPath(overlap1, new_overlap, left, right):
    diff = np.zeros_like(overlap1, dtype='float64')
    diff = (overlap1 - new_overlap) ** 2
    diff = np.sum(diff, axis=2)
    move_matrix = np.zeros_like(diff, dtype='int32')
    cost_matrix = np.zeros_like(diff, dtype='float64')
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if i == 0:
                cost_matrix[i, j] = diff[i, j]
                move_matrix[i, j] = 0
            else:
                cost = []
                move = []
                if j < (diff.shape[1] - 1):
                    cost.append(int(diff[i, j]) + int(cost_matrix[i - 1, j + 1]))
                    move.append(1)  # right
                cost.append(int(diff[i, j]) + int(cost_matrix[i - 1, j]))
                move.append(0)  # same col
                if j > 0:
                    cost.append(int(diff[i, j]) + int(cost_matrix[i - 1, j - 1]))
                    move.append(-1)  # left

                min_cost = min(cost)
                min_move = move[cost.index(min_cost)]
                cost_matrix[i, j] = min_cost
                move_matrix[i, j] = min_move

    mask = np.ones((diff.shape[0], diff.shape[1], 3))
    mask = mask * right
    min_index = np.argmin(cost_matrix[diff.shape[0] - 1, :])
    mask[diff.shape[0] - 1, :min_index + 1, :] = left
    for i in range(diff.shape[0] - 2, -1, -1):
        min_index = min_index + move_matrix[i + 1, min_index]
        mask[i, :min_index + 1, :] = left
    # print(mask.shape)
    return mask


def combineVerticalOverlaps(overlap1, new_overlap):
    mask = verticalMinimumCostPath(np.copy(overlap1), np.copy(new_overlap), 1, 0)
    full_one_matrix = np.ones_like(overlap1)
    result = mask * overlap1
    result = result + (full_one_matrix - mask) * new_overlap
    return result


def add_new_image(result_image, image):
    result_mask = np.where(result_image > 0, 1, 0)
    image_mask = np.where(image > 0, 1, 0)
    overlap = np.multiply(result_mask, image_mask)
    indexes = np.where(overlap == 1)
    x_indexes = indexes[0]
    y_indexes = indexes[1]
    x_min = np.min(x_indexes)
    y_min = np.min(y_indexes)
    x_max = np.max(x_indexes)
    y_max = np.max(y_indexes)
    result_overlap = result_image[x_min:x_max + 1, y_min:y_max + 1]
    image_overlap = image[x_min:x_max + 1, y_min:y_max + 1]
    combined_overlap = combineVerticalOverlaps(result_overlap, image_overlap)
    return x_min, x_max, y_min, y_max, combined_overlap, overlap


translation_for_key_frames, size_for_key_frames = find_size_of_total_frame([90, 270, 450, 630, 810], reference)
size_for_key_frames = (size_for_key_frames[0], size_for_key_frames[1])
result_for_key_frames = cv.warpPerspective(frame_90, np.matmul(translation_for_key_frames, all_homographies[90]),
                                           size_for_key_frames)

for i in [90, 270, 450, 630, 810]:
    # print(i)
    if i == 90:
        continue
    warped_frame = cv.warpPerspective(all_frames[i], np.matmul(translation_for_key_frames, all_homographies[i]),
                                      size_for_key_frames)

    warped_frame_copy = np.copy(warped_frame)


    x_min, x_max, y_min, y_max, combined_overlap, overlap = add_new_image(result_for_key_frames, warped_frame)
    warped_frame[x_min:x_max + 1, y_min:y_max + 1] = combined_overlap

    warped_frame_copy_mask = np.where(warped_frame_copy > 0, 1, 0)
    warped_frame_mask = np.where(warped_frame > 0, 1, 0)
    diff = warped_frame_copy_mask - warped_frame_mask
    diff = np.where(diff > 0, 1, 0)
    warped_frame = warped_frame + diff * warped_frame_copy

    result_for_key_frames[x_min:x_max + 1, y_min:y_max + 1] = 0
    result_for_key_frames = result_for_key_frames + warped_frame

save_image(result_for_key_frames, "res04-key-frames-panorama")

print(time.time() - current_time)
current_time = time.time()
print("part2 end.")
print("part4 start....")

translation_for_all, size_for_all = find_size_of_total_frame(range(0, 901), reference)
size_for_all = (size_for_all[0], size_for_all[1])

import gc


def check_frame_include_block(frame, homography, start_x, end_x):
    src = np.float32([[0, 0], [0, frame.shape[0]], [frame.shape[1], frame.shape[0]], [frame.shape[1], 0]])
    dst = cv.perspectiveTransform(src.reshape(-1, 1, 2), homography)
    dst = dst.reshape(src.shape)
    top_left_x = dst[0, 0]
    down_left_x = dst[1, 0]
    top_right_x = dst[3, 0]
    down_right_x = dst[2, 0]
    max_min_dst_x = max(top_left_x, down_left_x)
    min_max_dst_x = min(top_right_x, down_right_x)
    if max_min_dst_x <= start_x and min_max_dst_x >= end_x:
        return True
    return False


def find_median_and_complete_background_image(background_image, included_frames, start_x, end_x):
    if len(included_frames) == 0:
        return background_image
    blocks = []
    print(len(included_frames))
    for index in included_frames:
        this_translation = np.matmul(np.array([[1, 0, -start_x], [0, 1, 0], [0, 0, 1]]), translation_for_all)
        block = cv.warpPerspective(all_frames[index], np.matmul(this_translation, all_homographies[index]),
                                   (end_x - start_x, size_for_all[1]))
        #         warped = cv.warpPerspective(all_frames[index], np.matmul(translation_for_all, all_homographies[index]),
        #                                     size_for_all)
        #         print(f"x start {start_x}")
        #         print(f"end x {end_x}")
        #         print(f"warp shape:{warped.shape}")
        #         block = warped[:, start_x:end_x + 1]
        #         del warped
        #         gc.collect()
        #         print(f"block shape{block.shape}")
        blocks.append(block)
        # save_image(block, f"block{index}_{start_x}")
        del block
        gc.collect()

    blocks = np.array(blocks)
    #     print(f"blocks shape{blocks.shape}")
    median_block = np.median(blocks, axis=0)
    # save_image(median_block, f"median{start_x}")
    del blocks
    gc.collect()
    #     print(f"median shape {median_block.shape}")
    background_image[:, start_x:end_x, :] = median_block
    del median_block
    gc.collect()

    return background_image


def find_background_panorama():
    background_panorama = np.zeros((size_for_all[1], size_for_all[0], 3))
    for i in range(0, background_panorama.shape[1] - 80, 80):
        start_x = i
        end_x = i + 80
        frame_counter = 0
        frame_included_current_block = []
        for homo in all_homographies:
            if frame_counter == 0:
                frame_counter = frame_counter + 1
                continue
            flag = check_frame_include_block(all_frames[frame_counter], np.matmul(translation_for_all, homo), start_x,
                                             end_x)
            if flag:
                frame_included_current_block.append(frame_counter)

            frame_counter = frame_counter + 1
        background_panorama = find_median_and_complete_background_image(background_panorama,
                                                                        frame_included_current_block, start_x, end_x)

        # save_image(background_panorama, f"back{start_x}")

    save_image(background_panorama, "res06-background-panorama")
    return background_panorama


background_panorama = find_background_panorama()
background_panorama = my_clip(background_panorama, True)
print(time.time() - current_time)
current_time = time.time()
print("part4 end.")
print("part5 start....")

translation_for_part5, size_for_part5 = find_size_of_total_frame(range(0, 901), reference)
video_writer = cv.VideoWriter("res07-background-video.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30,
                              (reference.shape[1], reference.shape[0]))
i = 0
for homo in all_homographies:
    if i == 0:
        i = i + 1
        continue
    matrix = np.matmul(translation_for_part5, homo)
    matrix = np.linalg.inv(matrix)
    video_writer.write(cv.warpPerspective(background_panorama, matrix, (reference.shape[1], reference.shape[0])))
    # print(i)
    i = i + 1

video_writer.release()

print(time.time() - current_time)
current_time = time.time()
print("part5 end.")
print("part6 start....")


def find_diff(image, homo, i):
    matrix = np.matmul(translation_for_part5, homo)
    matrix = np.linalg.inv(matrix)
    background = cv.warpPerspective(background_panorama, matrix, (reference.shape[1], reference.shape[0]))

    copy_image = np.copy(image)
    copy_image = copy_image.astype("float64")
    background_copy = np.copy(background)
    background_copy = background_copy.astype("float64")
    black_indexes = np.where(background < 10)
    #     print(len(black_indexes))
    #     print("***")
    background_copy[black_indexes[0], black_indexes[1], black_indexes[2]] = copy_image[
        black_indexes[0], black_indexes[1],
        black_indexes[2]]
    # save_image(background_copy,f"after_{i}")
    # save_image(image,f"before_{i}")
    different = copy_image - background_copy
    different = different.astype("float64")
    different = different * different
    different = np.sum(different, axis=2)
    different = np.sqrt(different)
    different = np.where(different > 75, 1, 0)
    different = different.astype("uint8")
    kernel = np.ones((9, 9), np.uint8)
    different = cv.morphologyEx(different, cv.MORPH_OPEN, kernel)
    return different


video_writer = cv.VideoWriter("res08-foreground-video.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30,
                              (reference.shape[1], reference.shape[0]))
i = 0
for homo in all_homographies:
    if i == 0:
        i = i + 1
        continue

    diff = find_diff(all_frames[i], homo, i)
    copy_frame = np.copy(all_frames[i])
    indexes = np.where(diff > 0)
    # save_image(diff,f"diff_{i}")
    #     print(indexes[0].shape)
    #     print(copy_frame[indexes[0],indexes[1]].shape)
    copy_frame[indexes[0], indexes[1], 2] = 255
    copy_frame[indexes[0], indexes[1], 0] = 0
    copy_frame[indexes[0], indexes[1], 1] = 0
    #     cv.imwrite(f"foreGround_{i}.jpg",copy_frame)
    #     save_image(copy_frame,f"foreGround_{i}")
    video_writer.write(copy_frame)
    # print(i)
    i = i + 1

video_writer.release()

print(time.time() - current_time)
current_time = time.time()
print("part6 end.")
print("part7 start....")

translation_for_part5, size_for_part5 = find_size_of_total_frame(range(0, 901), reference)
new_frame_width = 1.5 * reference.shape[1]
new_frame_width = int(new_frame_width)
frame_size = (new_frame_width, reference.shape[0])

video_writer = cv.VideoWriter("res09-background-video-wider.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, frame_size)
i = 0
for homo in all_homographies:
    if i == 0:
        i = i + 1
        continue
    matrix = np.matmul(translation_for_part5, homo)
    matrix = np.linalg.inv(matrix)
    video_writer.write(cv.warpPerspective(background_panorama, matrix, frame_size))
    # print(i)
    i = i + 1

video_writer.release()

print(time.time() - current_time)
current_time = time.time()
print("part7 end.")
print("part8 start....")

import scipy.ndimage

vec1 = all_homographies[:, 0, 0]
vec2 = all_homographies[:, 0, 1]
vec3 = all_homographies[:, 0, 2]
vec4 = all_homographies[:, 1, 0]
vec5 = all_homographies[:, 1, 1]
vec6 = all_homographies[:, 1, 2]
vec7 = all_homographies[:, 2, 0]
vec8 = all_homographies[:, 2, 1]
vec9 = all_homographies[:, 2, 2]
vectors = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9]
sigma = 11
i = 0
for vec in vectors:
    vectors[i] = scipy.ndimage.gaussian_filter1d(vec, sigma)
    i = i + 1

new_all_homographies = np.copy(all_homographies)
new_all_homographies[:, 0, 0] = vectors[0]
new_all_homographies[:, 0, 1] = vectors[1]
new_all_homographies[:, 0, 2] = vectors[2]
new_all_homographies[:, 1, 0] = vectors[3]
new_all_homographies[:, 1, 1] = vectors[4]
new_all_homographies[:, 1, 2] = vectors[5]
new_all_homographies[:, 2, 0] = vectors[6]
new_all_homographies[:, 2, 1] = vectors[7]
new_all_homographies[:, 2, 2] = vectors[8]
video_writer = cv.VideoWriter("res10-video-shakeless.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30,
                              (reference.shape[1], reference.shape[0]))
i = 0
for homo in all_homographies:
    if i == 0:
        i = i + 1
        continue
    matrix = np.matmul(np.linalg.inv(new_all_homographies[i]), homo)
    video_writer.write(cv.warpPerspective(all_frames[i], matrix, (reference.shape[1], reference.shape[0])))
    # print(i)
    i = i + 1

video_writer.release()

print(time.time() - current_time)
print("part8 end.")
