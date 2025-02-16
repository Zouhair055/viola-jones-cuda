"""
Viola-Jones Algorithm

"""

import random
import os
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from nms import non_max_suppression
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


def train(dataset_path, training_set_faces, training_set_nofaces, test_size=100):
    features_path = r"./weights/{}/".format(test_size)
    try:
        X = np.load(features_path + "x" + ".npy")
        y = np.load(features_path + "y" + ".npy")
        print("Dataset loaded!")

    except FileNotFoundError:
        print("Loading new training set...")
        X, y = load_dataset(dataset_path, training_set_faces, training_set_nofaces)

        sprite_sheet = aggregate_images(X)
        integral_sprite_sheet = integral_image_gpu(sprite_sheet)

        img_height, img_width = X.shape[1], X.shape[2]
        num_images = X.shape[0]
        integral_images = np.zeros_like(X)
        for idx in range(num_images):
            row = (idx // int(np.sqrt(num_images))) * img_height
            col = (idx % int(np.sqrt(num_images))) * img_width
            integral_images[idx] = integral_sprite_sheet[row:row + img_height, col:col + img_width]

        X, y = unison_shuffled_copies(integral_images, y)

        if isinstance(test_size, int):
            X = X[:test_size]
            y = y[:test_size]

        os.makedirs(features_path, exist_ok=True)
        np.save(features_path + "x" + ".npy", X)
        np.save(features_path + "y" + ".npy", y)
        print("New dataset saved!")

    print("\nTraining Viola-Jones...")
    clf = ViolaJones(layers=[1, 10, 50, 100], features_path=features_path)
    clf.train(X, y)
    print("Training finished!")

    print("\nSaving weights...")
    clf.save(features_path + 'cvj_weights_' + str(int(time.time())))
    print("Weights saved!")

    return clf


def test(clf, dataset_path, dataset_faces, dataset_nofaces, name=""):
    print("\nLoading {}...".format(name))
    X, y = load_dataset(dataset_path, dataset_faces, dataset_nofaces)

    print("\nEvaluating...")
    metrics = evaluate(clf, X, y, show_samples=False)

    print("Metrics: [{}]".format(name))
    counter = 0
    for k, v in metrics.items():
        counter += 1
        if counter <= 4:
            print("\t- {}: {:,}".format(k, v))
        else:
            print("\t- {}: {:.3f}".format(k, v))


def train_and_test(dataset_path, training_set_faces, training_set_nofaces, test_set_faces, test_set_nofaces):
    start_time = time.time()
    clf = train(dataset_path, training_set_faces, training_set_nofaces)
    train_time = time.time() - start_time

    # Vérifie si le chemin est déjà absolu avant de le modifier
    training_set_faces_path = training_set_faces if os.path.isabs(training_set_faces) else os.path.join(dataset_path, os.path.basename(training_set_faces))
    training_set_nofaces_path = training_set_nofaces if os.path.isabs(training_set_nofaces) else os.path.join(dataset_path, os.path.basename(training_set_nofaces))
    test_set_faces_path = test_set_faces if os.path.isabs(test_set_faces) else os.path.join(dataset_path, os.path.basename(test_set_faces))
    test_set_nofaces_path = test_set_nofaces if os.path.isabs(test_set_nofaces) else os.path.join(dataset_path, os.path.basename(test_set_nofaces))

    # Vérification des chemins
    print(f"Training Faces Path: {training_set_faces_path}")
    print(f"Training NoFaces Path: {training_set_nofaces_path}")
    print(f"Test Faces Path: {test_set_faces_path}")
    print(f"Test NoFaces Path: {test_set_nofaces_path}")

    start_time = time.time()
    test(clf, dataset_path, training_set_faces_path, training_set_nofaces_path, name="Training set")
    test(clf, dataset_path, test_set_faces_path, test_set_nofaces_path, name="Test set")
    test_time = time.time() - start_time

    print("\nTraining time: {:.2f} seconds".format(train_time))
    print("Testing time: {:.2f} seconds".format(test_time))





def find_faces(image_path, weight_path, output_path="detected_faces_output.jpg"):
    # Charger le classifieur pré-entraîné
    clf = ViolaJones.load(weight_path)

    # Charger l'image
    pil_img = load_image(image_path)

    # Trouver les visages dans l'image
    regions = clf.find_faces(pil_img)

    # Préparer les scores et appliquer la suppression non maximale
    regions = np.array(regions)
    scores = np.array([1.0] * len(regions))  # Les scores sont simplement de 1.0 ici
    threshold = 0.5  # Seuil pour la détection

    # Effectuer la suppression non maximale pour éviter les détections multiples
    indicies = non_max_suppression(regions, scores, threshold)

    # Dessiner les boîtes de détection
    drawn_img = draw_bounding_boxes(pil_img, list(regions[indicies]), thickness=1)

    # Convertir l'image en mode RGB si nécessaire
    if drawn_img.mode == 'RGBA':
        drawn_img = drawn_img.convert('RGB')

    # Enregistrer l'image avec les boîtes de détection
    drawn_img.save(output_path)
    print(f"Image avec détection sauvegardée sous {output_path}")



# Fonction pour générer un nom unique pour le fichier de sortie
def generate_unique_filename(output_path):
    if os.path.exists(output_path):
        timestamp = int(time.time())  # Utilise l'horodatage actuel
        name, ext = os.path.splitext(output_path)  # Sépare le nom du fichier et l'extension
        output_path = f"{name}_{timestamp}{ext}"  # Crée un nouveau nom avec le suffixe
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viola-Jones Algorithm")
    
    # Argument pour exécuter l'entraînement et le test
    parser.add_argument("-bench", nargs=4, metavar=('train_faces', 'train_nofaces', 'test_faces', 'test_nofaces'),
                        help="Effectue un entrainement puis test les données et affiche le temps d'exécution des différentes phases")
    
    # Argument pour effectuer seulement l'entraînement
    parser.add_argument("-train", nargs=2, metavar=('train_faces', 'train_nofaces'),
                        help="Effectue seulement la phase d'entrainement")
    
    # Argument pour détecter les visages dans une image
    parser.add_argument("-detect", metavar='image_path',
                        help="Détecte si l'image passée en paramètre contient des visages")

    args = parser.parse_args()

    if args.bench:
        # On déstructure les fichiers passés en paramètre pour l'entraînement et le test
        train_faces, train_nofaces, test_faces, test_nofaces = args.bench
        
        # Définit le chemin du dataset en fonction des fichiers d'entraînement
        dataset_path = os.path.abspath("viola-jones/datasets")
        print(f"Train Faces: {train_faces}")
        print(f"Train NoFaces: {train_nofaces}")
        print(f"Test Faces: {test_faces}")
        print(f"Test NoFaces: {test_nofaces}")

        # Lancer l'entraînement et les tests
        train_and_test(dataset_path, train_faces, train_nofaces, test_faces, test_nofaces)
    
    elif args.train:
        # On déstructure les fichiers pour l'entraînement uniquement
        train_faces, train_nofaces = args.train
        
        # Définit le chemin du dataset
        dataset_path = os.path.abspath("viola-jones/datasets")
        
        # Lancer seulement l'entraînement
        train(dataset_path, train_faces, train_nofaces)
    
    elif args.detect:
        # Récupère le chemin de l'image passée en paramètre
        image_path = args.detect
        
        # Définir le chemin des poids du modèle
        weight_path = f"weights/100/cvj_weights_1739703504.pkl"  # Remplace par ton propre chemin de poids
        
        # Définir le chemin de sauvegarde de l'image de sortie
        output_path = './detected_faces_output.jpg'
        
        # Générer un nom unique pour l'image de sortie, si nécessaire
        output_path = generate_unique_filename(output_path)
        
        # Lancer la détection des visages
        find_faces(image_path, weight_path, output_path)
    
    else:
        # Si aucun argument valide n'est passé, afficher l'aide
        parser.print_help()