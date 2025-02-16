import time
import glob
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from features import RectangleRegion, HaarFeature
from progress.bar import Bar
import multiprocessing
from numba import cuda
import numba
import os

def imshow(img):
    Image.fromarray(img).show()


def load_image(image_path, as_numpy=False):
    pil_img = Image.open(image_path)
    if as_numpy:
        return np.array(pil_img)
    else:
        return pil_img


def rgb2gray(img):
    # Formula: https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


# Kernel pour le scan exclusif par bloc avec padding
@cuda.jit
def exclusive_scan_kernel(d_array, d_output, n):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gid = bid * bdim + tid

    shared_array = cuda.shared.array(shape=(1024,), dtype=numba.int32)

    if gid < n:
        shared_array[tid] = d_array[gid]
    else:
        shared_array[tid] = 0
    cuda.syncthreads()

    step = 1
    while step < bdim:
        temp = shared_array[tid]
        if tid >= step:
            temp += shared_array[tid - step]
        cuda.syncthreads()
        shared_array[tid] = temp
        cuda.syncthreads()
        step *= 2

    if gid < n:
        d_output[gid] = shared_array[tid]

# Scan exclusif sur GPU
def exclusive_scan(array):
    n = len(array)
    threads_per_block = min(1024, n)
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    d_array = cuda.to_device(array)
    d_output = cuda.device_array_like(d_array)

    exclusive_scan_kernel[blocks_per_grid, threads_per_block](d_array, d_output, n)

    return d_output.copy_to_host()

# Kernel pour la transposition
@cuda.jit
def transpose_kernel(input_array, output_array):
    x, y = cuda.grid(2)
    if x < input_array.shape[1] and y < input_array.shape[0]:
        output_array[x, y] = input_array[y, x]

# Transposition sur GPU
def transpose(array):
    height, width = array.shape
    output_array = cuda.device_array((width, height), dtype=array.dtype)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    
    transpose_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](cuda.to_device(array), output_array)
    return output_array.copy_to_host()

# Calcul de l'image intégrale sur GPU
def integral_image_gpu(image):
    image = image.astype(np.int32)  # S'assurer du bon type
    
    for i in range(image.shape[0]):
        image[i, :] = exclusive_scan(image[i, :])
    
    transposed_image = transpose(image)
    
    for i in range(transposed_image.shape[0]):
        transposed_image[i, :] = exclusive_scan(transposed_image[i, :])
    
    return transpose(transposed_image)

# Remplacer la fonction existante par la nouvelle fonction GPU
def integral_image(img):
    return integral_image_gpu(img)



def aggregate_images(images):
    """
    Combine all images into a single large image (sprite sheet).
    """
    num_images, img_height, img_width = images.shape
    sprite_height = int(np.ceil(np.sqrt(num_images))) * img_height
    sprite_width = int(np.ceil(np.sqrt(num_images))) * img_width

    sprite_sheet = np.zeros((sprite_height, sprite_width), dtype=images.dtype)

    for idx, img in enumerate(images):
        row = (idx // int(np.sqrt(num_images))) * img_height
        col = (idx % int(np.sqrt(num_images))) * img_width
        sprite_sheet[row:row + img_height, col:col + img_width] = img

    return sprite_sheet

# def integral_image(img):
#     """
#     Optimized version of Summed-area table
#     ii(-1, y) = 0
#     s(x, -1) = 0
#     s(x, y) = s(x, y-1) + i(x, y)  # Sum of column X at level Y
#     ii(x, y) = ii(x-1, y) + s(x, y)  # II at (X-1,Y) + Column X at Y
#     """
#     h, w = img.shape

#     s = np.zeros(img.shape, dtype=np.uint32)
#     ii = np.zeros(img.shape, dtype=np.uint32)

#     for x in range(0, w):
#         for y in range(0, h):
#             s[y][x] = s[y - 1][x] + img[y][x] if y - 1 >= 0 else img[y][x]
#             ii[y][x] = ii[y][x - 1] + s[y][x] if x - 1 >= 0 else s[y][x]
#     return ii

def integral_image_pow2(img):
    """
    Squared version of II
    """
    return integral_image(img**2)


def build_features(img_w, img_h, shift=1, scale_factor=1.25, min_w=4, min_h=4):
    """
    Generate values from Haar features

    White rectangles substract from black ones
    """
    features = []  # [Tuple(positive regions, negative regions),...]

    # Scale feature window
    for w_width in range(min_w, img_w + 1):
        for w_height in range(min_h, img_h + 1):

            # Walk through all the image
            x = 0
            while x + w_width < img_w:
                y = 0
                while y + w_height < img_h:

                    # Possible Haar regions
                    immediate = RectangleRegion(x, y, w_width, w_height)  # |X|
                    right = RectangleRegion(x + w_width, y, w_width, w_height)  # | |X|
                    right_2 = RectangleRegion(x + w_width * 2, y, w_width, w_height)  # | | |X|
                    bottom = RectangleRegion(x, y + w_height, w_width, w_height)  # | |/|X|
                    #bottom_2 = RectangleRegion(x, y + w_height * 2, w_width, w_height)  # | |/| |/|X|
                    bottom_right = RectangleRegion(x + w_width, y + w_height, w_width, w_height)  # | |/| |X|

                    # [Haar] 2 rectagles *********
                    # Horizontal (w-b)
                    if x + w_width * 2 < img_w:
                        features.append(HaarFeature([immediate], [right]))
                    # Vertical (w-b)
                    if y + w_height * 2 < img_h:
                        features.append(HaarFeature([bottom], [immediate]))

                    # [Haar] 3 rectagles *********
                    # Horizontal (w-b-w)
                    if x + w_width * 3 < img_w:
                        features.append(HaarFeature([immediate, right_2], [right]))
                    # # Vertical (w-b-w)
                    # if y + w_height * 3 < img_h:
                    #     features.append(HaarFeature([immediate, bottom_2], [bottom]))

                    # [Haar] 4 rectagles *********
                    if x + w_width * 2 < img_w and y + w_height * 2 < img_h:
                        features.append(HaarFeature([immediate, bottom_right], [bottom, right]))

                    y += shift
                x += shift
    return features  # np.array(features)


def apply_features(X_ii, features):
    """
    Apply build features (regions) to all the training data (integral images)
    """

    X = np.zeros((len(features), len(X_ii)), dtype=np.int32)
    # 'y' will be kept as it is => f0=([...], y); f1=([...], y),...

    bar = Bar('Processing features', max=len(features), suffix='%(percent)d%% - %(elapsed_td)s - %(eta_td)s')
    for j, feature in bar.iter(enumerate(features)):
    # for j, feature in enumerate(features):
    #     if (j + 1) % 1000 == 0 and j != 0:
    #         print("Applying features... ({}/{})".format(j + 1, len(features)))

        # Compute the value of feature 'j' for each image in the training set (Input of the classifier_j)
        X[j] = list(map(lambda ii: feature.compute_value(ii), X_ii))
    bar.finish()

    return X


def show_sample(x, y, y_pred):
    target = "Face" if y == 1 else "No face"
    pred = "Face" if y_pred == 1 else "No face"
    img_text = "Class: {}  - Prediction: {}".format(target, pred)
    print(img_text)

    plt.title(img_text)
    plt.imshow(x, cmap='gray')
    plt.show()


def evaluate(clf, X, y, show_samples=False):
    metrics = {}
    true_positive, true_negative = 0, 0  # Correct
    false_positive, false_negative = 0, 0  # Incorrect

    for i in range(len(y)):
        prediction = clf.classify(X[i])
        if prediction == y[i]:  # Correct
            if prediction == 1:  # Face
                true_positive += 1
            else:  # No-face
                true_negative += 1
        else:  # Incorrect
            #if show_samples: show_sample(X[i], y[i], prediction)

            if prediction == 1:  # Face
                false_positive += 1
            else:  # No-face
                false_negative += 1

    # Compute metrics
    metrics['true_positive'] = true_positive
    metrics['true_negative'] = true_negative
    metrics['false_positive'] = false_positive
    metrics['false_negative'] = false_negative

    metrics['accuracy'] = (true_positive + true_negative)/(true_positive+false_negative+true_negative+false_positive)
    if true_positive + false_positive > 0:
        metrics['precision'] = true_positive / (true_positive+false_positive)
    else:
        metrics['precision'] = 0
    if true_positive + false_negative > 0:
        metrics['recall'] = true_positive / (true_positive+false_negative)  # or Sensitivity
    else:
        metrics['recall'] = 0
    if true_negative + false_positive > 0:
        metrics['specifity'] = true_negative/(true_negative+false_positive)
    else:
        metrics['specifity'] = 0
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = (2.0 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0

    return metrics


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_images_from_dir(path, extension="*.*"):
    image_list = []
    for filename in glob.glob(path + '/' + extension):  # assuming gif
        img = Image.open(filename)
        #img = img.convert('L')  # To grayscale
        #img = img.resize((19, 19), Image.ANTIALIAS)  # Resize
        img = np.array(img)
        image_list.append(img)

    image_list = np.stack(image_list, axis=0)
    return image_list





def load_dataset(dataset_path, dataset_faces, dataset_nofaces):
    # Vérifier si les chemins sont déjà absolus, sinon les concaténer correctement
    pos_filepath = dataset_faces if os.path.isabs(dataset_faces) else os.path.abspath(os.path.join(dataset_path, dataset_faces))
    neg_filepath = dataset_nofaces if os.path.isabs(dataset_nofaces) else os.path.abspath(os.path.join(dataset_path, dataset_nofaces))

    if not os.path.exists(pos_filepath):
        raise FileNotFoundError(f"Le fichier {pos_filepath} n'a pas été trouvé.")
    if not os.path.exists(neg_filepath):
        raise FileNotFoundError(f"Le fichier {neg_filepath} n'a pas été trouvé.")

    X_faces = np.load(pos_filepath)
    y_faces = np.ones(X_faces.shape[0])

    X_nofaces = np.load(neg_filepath)
    y_nofaces = np.zeros(X_nofaces.shape[0])

    return np.concatenate((X_faces, X_nofaces)), np.concatenate((y_faces, y_nofaces))






def dir2file(folder, savefile):
    # Load images
    images = load_images_from_dir(folder, "*.pgm")
    print("{} images loaded".format(len(images)))

    # Save images
    np.save(savefile, images)
    print("Done!")


def get_pretty_time(start_time, end_time=None, s="", divisor=1.0):
    if not end_time:
        end_time = time.time()
    hours, rem = divmod((end_time - start_time)/divisor, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{}{:0>2}:{:0>2}:{:05.8f}".format(s, int(hours), int(minutes), seconds)


def draw_bounding_boxes(pil_image, regions, color="green", thickness=3):
    # Prepare image
    source_img = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for rect in regions:
        draw.rectangle(tuple(rect), outline=color, width=thickness)
    return source_img


def non_maximum_supression(regions, threshold=0.5):
    # Code from: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    boxes = np.array(regions)
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] == scores.shape[0]
    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious


def normalize_image(image):
    ii = integral_image(image)
    mean = np.mean(image)
    stdev = np.std(image)
    norm_img = (image-mean)/stdev
    return norm_img


def draw_haar_feature(np_img, haar_feature):
    pil_img = Image.fromarray(np_img).convert("RGBA")

    draw = ImageDraw.Draw(pil_img)
    for rect in haar_feature.positive_regions:
        x1, y1, x2, y2 = rect.x, rect.y, rect.x + rect.width - 1, rect.y + rect.height - 1
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, 255))

    for rect in haar_feature.negative_regions:
        x1, y1, x2, y2 = rect.x, rect.y, rect.x + rect.width - 1, rect.y + rect.height - 1
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 255))

    return pil_img
