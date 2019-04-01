"""
Viola-Jones Algorithm

"""

import random

import matplotlib.pyplot as plt
from nms import nms

from utils import *
from violajones import ViolaJones

training_set_faces = "cbcl_train_faces_19x19g.npy"
training_set_nofaces = "cbcl_train_nofaces_19x19g.npy"
test_set_faces = "cbcl_test_faces_19x19g.npy"
test_set_nofaces = "cbcl_test_nofaces_19x19g.npy"


def check_data(X, y, num2show=10):
    for i in range(num2show):
        idx = random.randint(0, len(y))
        img = X[idx]
        target = "Face" if y[idx] == 1 else "No face"
        img_text = "Index: {}  -  Target: {}  -  Image size: {}x{}".format(idx, target, *img.shape)
        print(img_text)

        plt.title(img_text)
        plt.imshow(img, cmap='gray')
        plt.show()


def data_augmentation(X, y):
    faces_idxs = np.where(y == 1)[0]
    nonfaces_idxs = np.where(y == 0)[0]

    # Horizontal flip
    X_hf = X[faces_idxs, :, ::-1]

    # Stack samples
    X = np.concatenate((X[faces_idxs], X_hf, X[nonfaces_idxs]))
    y = np.zeros(len(X))
    y[:len(faces_idxs)*2] = 1  # Add new targets

    return X, y


def train(dataset_path):
    # Load precomputed dataset
    test_size = "all_da"
    features_path = r"./weights/{}/".format(test_size)
    try:
        X = np.load(features_path + "x" + ".npy")
        y = np.load(features_path + "y" + ".npy")
        print("Dataset loaded!")

    except FileNotFoundError:
        # LOADING AND PREPROCESSING *************************************************
        # Load training data
        print("Loading new training set...")
        X, y = load_dataset(dataset_path, training_set_faces, training_set_nofaces)

        # Data augmentation
        X, y = data_augmentation(X, y)

        # Shuffle data (Not needed with the CascadeClassifier)
        X, y = unison_shuffled_copies(X, y)

        # Modify sizes
        if isinstance(test_size, int):
            X = X[:test_size]
            y = y[:test_size]

        # Save current dataset
        np.save(features_path + "x" + ".npy", X)
        np.save(features_path + "y" + ".npy", y)
        print("New dataset saved!")

    # Check data
    #check_data(X, y, num2show=20)

    # TRAINING ******************************************************************
    # Train
    print("\nTraining Viola-Jones...")
    clf = ViolaJones(layers=[1, 5], features_path=features_path)
    clf.train(X, y)  # X_f (optional, to speed-up training)
    print("Training finished!")

    # Save weights
    print("\nSaving weights...")
    clf.save(features_path + 'cvj_weights_' + str(int(time.time())))
    print("Weights saved!")

    return clf


def test(clf, dataset_path):
    # Load test set
    print("\nLoading test set...")
    #X, y = load_dataset(dataset_path, training_set_faces, training_set_nofaces)
    X, y = load_dataset(dataset_path, test_set_faces, test_set_nofaces)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(clf, X, y, show_samples=False)

    print("Metrics:")
    counter = 0
    for k, v in metrics.items():
        counter += 1
        if counter <= 4:
            print("\t- {}: {:,}".format(k, v))
        else:
            print("\t- {}: {:.3f}".format(k, v))


def train_and_test():
    # image_path = "datasets/judybats.jpg"
    # detector(image_path)
    #dataset_path = r"C:\Users\salva\Documents\Programacion\Datasets\Faces\CBCL Face Database\compressed"
    dataset_path = r"/Users/salvacarrion/Documents/Programming/Datasets/Faces/CBCL Face Database/compressed"

    # Training
    clf = ViolaJones.load("weights/all_da/cvj_weights_1554151172.pkl")
    clf.train(dataset_path)

    # Test
    test(clf, dataset_path)

    print("\nFinished!")


def find_faces():
    weight_path = r"weights/1000/cvj_weights_1554133497.pkl"
    face_path = r"./datasets/judybats.jpg"
    #face_path = r"./datasets/i1.jpg"
    face_path = r"./datasets/people.png"
    face_path = r"./datasets/clase.png"
    face_path = r"./datasets/physics.jpg"


    # Load classifier weights
    clf = ViolaJones.load(weight_path)

    # Find regions of the faces
    pil_img = load_image(face_path)
    regions = clf.find_faces(pil_img)

    # Draw bouding boxes
    # TODO: Review Non-maximum supression (fix own implementation)
    # regions = np.array([(10, 10, 50, 50), (20, 20, 60, 60), (37, 59, 199, 244), (47, 69, 209, 254)])
    regions = np.array(regions)
    scores = [1.0]*len(regions)  #np.ones(len(regions))
    indicies = nms.boxes(regions, scores)
    drawn_img = draw_bounding_boxes(pil_img, list(regions[indicies]), thickness=2)

    # Show image
    plt.imshow(drawn_img)
    plt.show()

    # pil_image = load_image(face_path, as_numpy=False)
    # pil_image = pil_image.convert('L')
    # img = np.array(pil_image)
    # norm_img = normalize_image(img)
    # # Show image
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # # Show image
    # plt.imshow(norm_img, cmap="gray")
    # plt.show()


if __name__ == "__main__":
    start_time = time.time()
    print("Starting scripting...")

    train_and_test()
    #find_faces()

    # Elapsed time
    print("\n" + get_pretty_time(start_time, s="Total time (Training+test): "))

