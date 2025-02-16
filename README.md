# Viola-Jones GPU Acceleration Project

## Description
Ce projet vise à améliorer une implémentation existante de l'algorithme de détection de visages Viola-Jones en optimisant certaines parties intensives du code en les exécutant sur GPU via CUDA. L'objectif est d'accélérer les performances par rapport à la version séquentielle sur CPU.

## Améliorations apportées
### Remplacement des méthodes CPU par des versions GPU :
- **Conversion d'image en niveaux de gris** : Accélération via GPU
- **Scan inclusif** : Optimisation avec CUDA
- **Transposition de matrice** : Optimisation avec CUDA
- **Calcul de l'image intégrale** : Porté sur GPU pour une exécution plus rapide
- **Agrégation des images en Sprite Sheet** : Traitement de toutes les images en une seule passe GPU

### Optimisation de l'entraînement :
- Ajout de la possibilité de traiter toutes les images en une seule passe sur GPU

## Exécution du projet
### Dépendances
Avant d'exécuter le projet, installez les dépendances requises :
```bash
cd projet
pip install -r requirements.txt
```

### Commandes disponibles
#### 1. Entraînement et test :
```bash
python viola-jones/main.py -bench     ./viola-jones/datasets/cbcl_train_faces_19x19g.npy     ./viola-jones/datasets/cbcl_train_nofaces_19x19g.npy     ./viola-jones/datasets/cbcl_test_faces_19x19g.npy     ./viola-jones/datasets/cbcl_test_nofaces_19x19g.npy
```
Exécute l'entraînement et le test en affichant le temps d'exécution des différentes phases.
    --**Exemple d'éxécution** : Projet/viola-jones/results/result.png
    --**http://miage-gpu2.unice.fr:8001/user/dz303428/files/Projet/viola-jones/results/result.png?_xsrf=2%7C0a6cd358%7Ca4650a0ccc6a6e1ab2ff1d801dc31c69%7C1739175531**

#### 2. Entraînement uniquement :
```bash
python viola-jones/main.py -train ./viola-jones/datasets/train_faces.npy ./viola-jones/datasets/train_nofaces.npy
```
Effectue uniquement la phase d'entraînement.

#### 3. Détection de visages :
```bash
python viola-jones/main.py -detect <path_to_image>
python viola-jones/main.py -detect ./viola-jones/datasets/test.jpg  ## Exemple d'image existante
```
Détecte si l'image passée en paramètre contient des visages, et L'image avec la détection est sauvegardée automatiquement à la racine du projet. **/projet**

## Structure du projet
```
viola-jones/
│── __pycache__/
│── datasets/          # 📂 Dataset - Contient les données utilisées pour l'entraînement
│── images/            
│── results/           # 📂 Data - Stocke les résultats de détection
│── testdata/          # 📂 Data - Contient les images utilisées pour la détection
│── tests/             
│── .gitignore
│── adaboost.py        # 📜 Core - Implémentation de l'algorithme AdaBoost
│── convert-to-pny.py  # 📜 Convert - Conversion des fichiers en un format spécifique
│── features.py        # 📜 Core - Extraction des features de Haar
│── main.py            # 🚀 Main - Point d'entrée du programme
│── nms.py             # 📜 Utils - Implémentation de la suppression des non-maxima
│── README.md
│── requirements.txt
│── utils.py           # 📜 Utils - Fonctions utilitaires où ya le calcul de l'image intégral
│── version.py         # 📜 Version - Gestion des versions du projet
│── violajones.py      # 📜 Core - Algorithme de Viola-Jones pour la détection d'objets
│── weakclassifier.py  # 📜 Core - Implémentation des classifieurs faibles
weights/               # 📂 Data - Poids du modèle entraîné

```

## Résumé des modifications GPU
Les parties suivantes du code ont été adaptées pour être exécutées sur GPU :
- **Conversion d'image en NB** : `image_to_gray_gpu`
- **Scan inclusif** : `inclusive_scan_gpu`
- **Transposition de matrice** : `transpose_matrix_gpu`
- **Calcul de l'image intégrale** : `integral_image_gpu`
- **Traitement de l'ensemble des en une fois sur le GPU**

## Auteur
Projet réalisé dans le cadre d'un cours d'optimisation GPU.

## Zouhair DKHISSI


