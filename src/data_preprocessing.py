"""
FruitVision - Prétraitement des Données
=======================================

Ce module gère le chargement, le prétraitement et l'augmentation des données d'images de fruits.
Il prépare les données pour l'entraînement du modèle CNN.

Auteur: Mamadou Fall
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import random
from typing import Tuple, List, Dict


class PreprocesseurDonnees:
    """
    Classe principale pour le prétraitement des données d'images de fruits.
    
    Cette classe gère:
    - Le chargement des images depuis les dossiers
    - Le redimensionnement et la normalisation
    - L'augmentation des données
    - La création des ensembles train/validation/test
    """
    
    def __init__(self, taille_image=(100, 100), classes_fruits=None):
        """
        Initialiser le préprocesseur.
        
        Args:
            taille_image (tuple): Taille finale des images (largeur, hauteur)
            classes_fruits (dict): Mapping des fruits vers leurs catégories dataset
        """
        self.taille_image = taille_image
        self.classes_fruits = classes_fruits or {
            'Pomme': ['Apple Braeburn', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
                     'Apple Granny Smith', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
            'Banane': ['Banana', 'Banana Red'],
            'Kiwi': ['Kiwi'],
            'Citron': ['Lemon', 'Lemon Meyer'],
            'Peche': ['Peach', 'Peach 2']
        }
        
        # Encoder pour convertir noms de fruits en nombres
        self.encodeur_labels = LabelEncoder()
        self.noms_classes = list(self.classes_fruits.keys())
        self.encodeur_labels.fit(self.noms_classes)
        
        print(f"✅ Préprocesseur initialisé pour {len(self.noms_classes)} classes de fruits")
        
    
    def charger_image(self, chemin_image: str) -> np.ndarray:
        """
        Charger et préprocesser une seule image.
        
        Pourquoi ces étapes?
        - Redimensionnement: Toutes les images doivent avoir la même taille
        - Normalisation: Valeurs entre 0-1 pour un meilleur entraînement
        - Conversion RGB: Format standard pour les CNN
        
        Args:
            chemin_image (str): Chemin vers le fichier image
            
        Returns:
            np.ndarray: Image préprocessée (100, 100, 3)
        """
        try:
            # Charger l'image avec PIL (plus robuste que OpenCV)
            with Image.open(chemin_image) as img:
                # Convertir en RGB (au cas où l'image serait en RGBA ou autre)
                img_rgb = img.convert('RGB')
                
                # Redimensionner à la taille standard
                img_redimensionnee = img_rgb.resize(self.taille_image, Image.Resampling.LANCZOS)
                
                # Convertir en array numpy et normaliser (0-255 -> 0-1)
                img_array = np.array(img_redimensionnee, dtype=np.float32) / 255.0
                
                return img_array
                
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de {chemin_image}: {e}")
            # Retourner une image noire en cas d'erreur
            return np.zeros((*self.taille_image, 3), dtype=np.float32)
    
    
    def augmenter_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Créer des variations d'une image pour augmenter les données.
        
        Pourquoi l'augmentation?
        - Plus de données = meilleur apprentissage
        - Robustesse aux variations (rotation, luminosité, etc.)
        - Évite le surapprentissage
        
        Args:
            image (np.ndarray): Image originale
            
        Returns:
            List[np.ndarray]: Liste d'images augmentées
        """
        images_augmentees = []
        
        # Convertir en PIL pour les transformations
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        # 1. ROTATION (fruits peuvent être orientés différemment)
        for angle in [-15, -10, 10, 15]:  # Rotations légères
            img_rotee = img_pil.rotate(angle, fillcolor=(255, 255, 255))
            images_augmentees.append(np.array(img_rotee) / 255.0)
        
        # 2. LUMINOSITÉ (éclairage différent)
        enhancer_luminosite = ImageEnhance.Brightness(img_pil)
        for facteur in [0.8, 1.2]:  # Plus sombre, plus clair
            img_luminosite = enhancer_luminosite.enhance(facteur)
            images_augmentees.append(np.array(img_luminosite) / 255.0)
        
        # 3. CONTRASTE (différences d'appareil photo)
        enhancer_contraste = ImageEnhance.Contrast(img_pil)
        for facteur in [0.9, 1.1]:
            img_contraste = enhancer_contraste.enhance(facteur)
            images_augmentees.append(np.array(img_contraste) / 255.0)
        
        # 4. FLOU LÉGER (simule photos pas parfaitement nettes)
        img_floue = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
        images_augmentees.append(np.array(img_floue) / 255.0)
        
        # 5. MIROIR HORIZONTAL (certains fruits sont symétriques)
        img_miroir = img_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        images_augmentees.append(np.array(img_miroir) / 255.0)
        
        return images_augmentees
    
    
    def charger_donnees_dossier(self, chemin_base: str, utiliser_augmentation=True, 
                               max_images_par_classe=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charger toutes les images depuis la structure de dossiers du dataset.
        
        Structure attendue:
        chemin_base/
        ├── Apple Braeburn/
        ├── Apple Golden 1/
        ├── Banana/
        └── ...
        
        Args:
            chemin_base (str): Chemin vers le dossier Training ou Test
            utiliser_augmentation (bool): Appliquer l'augmentation de données
            max_images_par_classe (int): Limiter le nombre d'images par classe
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (images, labels)
        """
        print(f"📂 Chargement des données depuis: {chemin_base}")
        
        toutes_images = []
        tous_labels = []
        
        # Parcourir chaque fruit défini
        for nom_fruit, categories_dataset in self.classes_fruits.items():
            print(f"🍎 Traitement du fruit: {nom_fruit}")
            
            images_fruit = []
            
            # Parcourir chaque catégorie du dataset pour ce fruit
            for categorie in categories_dataset:
                chemin_categorie = os.path.join(chemin_base, categorie)
                
                if not os.path.exists(chemin_categorie):
                    print(f"⚠️ Dossier non trouvé: {chemin_categorie}")
                    continue
                
                # Lister les fichiers images
                fichiers_images = [f for f in os.listdir(chemin_categorie) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                print(f"   📁 {categorie}: {len(fichiers_images)} images")
                
                # Limiter le nombre d'images si spécifié
                if max_images_par_classe:
                    fichiers_images = fichiers_images[:max_images_par_classe]
                
                # Charger chaque image
                for nom_fichier in fichiers_images:
                    chemin_complet = os.path.join(chemin_categorie, nom_fichier)
                    image = self.charger_image(chemin_complet)
                    images_fruit.append(image)
                    
                    # Augmentation de données (seulement pour l'entraînement)
                    if utiliser_augmentation:
                        images_augmentees = self.augmenter_image(image)
                        images_fruit.extend(images_augmentees)
            
            # Ajouter toutes les images de ce fruit
            toutes_images.extend(images_fruit)
            
            # Créer les labels correspondants
            label_encode = self.encodeur_labels.transform([nom_fruit])[0]
            labels_fruit = [label_encode] * len(images_fruit)
            tous_labels.extend(labels_fruit)
            
            print(f"   ✅ {nom_fruit}: {len(images_fruit)} images totales (avec augmentation)")
        
        # Convertir en arrays numpy
        X = np.array(toutes_images, dtype=np.float32)
        y = np.array(tous_labels, dtype=np.int32)
        
        print(f"📊 Dataset chargé: {X.shape[0]} images, {len(self.noms_classes)} classes")
        
        return X, y
    
    
    def creer_ensembles_donnees(self, X: np.ndarray, y: np.ndarray, 
                               taille_validation=0.2, taille_test=0.1) -> Dict:
        """
        Diviser les données en ensembles d'entraînement, validation et test.
        
        Pourquoi cette division?
        - Entraînement (70%): Pour apprendre
        - Validation (20%): Pour ajuster les hyperparamètres
        - Test (10%): Pour évaluer la performance finale
        
        Args:
            X (np.ndarray): Images
            y (np.ndarray): Labels
            taille_validation (float): Proportion pour validation
            taille_test (float): Proportion pour test
            
        Returns:
            Dict: Dictionnaire avec les ensembles de données
        """
        print("🔄 Division des données en ensembles...")
        
        # Première division: séparer les données de test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=taille_test, random_state=42, stratify=y
        )
        
        # Deuxième division: séparer entraînement et validation
        taille_val_ajustee = taille_validation / (1 - taille_test)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=taille_val_ajustee, random_state=42, stratify=y_temp
        )
        
        # Convertir les labels en format one-hot pour le CNN
        from sklearn.preprocessing import LabelBinarizer
        binarizer = LabelBinarizer()
        y_train_onehot = binarizer.fit_transform(y_train)
        y_val_onehot = binarizer.transform(y_val)
        y_test_onehot = binarizer.transform(y_test)
        
        ensembles = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train_onehot,
            'y_val': y_val_onehot,
            'y_test': y_test_onehot,
            'y_train_raw': y_train,
            'y_val_raw': y_val,
            'y_test_raw': y_test,
            'noms_classes': self.noms_classes,
            'binarizer': binarizer
        }
        
        print(f"✅ Ensembles créés:")
        print(f"   📚 Entraînement: {X_train.shape[0]} images")
        print(f"   🔍 Validation: {X_val.shape[0]} images")
        print(f"   🧪 Test: {X_test.shape[0]} images")
        
        return ensembles
    
    
    def sauvegarder_statistiques(self, ensembles: Dict, chemin_sauvegarde: str):
        """
        Sauvegarder les statistiques des données pour référence.
        
        Args:
            ensembles (Dict): Ensembles de données
            chemin_sauvegarde (str): Où sauvegarder les stats
        """
        stats = {
            'nb_classes': len(self.noms_classes),
            'noms_classes': self.noms_classes,
            'taille_image': self.taille_image,
            'nb_train': ensembles['X_train'].shape[0],
            'nb_validation': ensembles['X_val'].shape[0],
            'nb_test': ensembles['X_test'].shape[0],
            'forme_image': ensembles['X_train'].shape[1:],
            'distribution_classes': {}
        }
        
        # Calculer la distribution des classes
        for i, nom_classe in enumerate(self.noms_classes):
            nb_train = np.sum(ensembles['y_train_raw'] == i)
            nb_val = np.sum(ensembles['y_val_raw'] == i)
            nb_test = np.sum(ensembles['y_test_raw'] == i)
            
            stats['distribution_classes'][nom_classe] = {
                'train': int(nb_train),
                'validation': int(nb_val),
                'test': int(nb_test),
                'total': int(nb_train + nb_val + nb_test)
            }
        
        # Sauvegarder en JSON
        with open(chemin_sauvegarde, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Statistiques sauvegardées dans: {chemin_sauvegarde}")


# Fonctions utilitaires
def visualiser_echantillons(X: np.ndarray, y: np.ndarray, noms_classes: List[str], 
                           nb_echantillons=5):
    """
    Visualiser quelques échantillons de chaque classe.
    
    Args:
        X (np.ndarray): Images
        y (np.ndarray): Labels
        noms_classes (List[str]): Noms des classes
        nb_echantillons (int): Nombre d'échantillons par classe
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(len(noms_classes), nb_echantillons, 
                            figsize=(15, 3*len(noms_classes)))
    
    for i, nom_classe in enumerate(noms_classes):
        # Trouver les indices de cette classe
        indices_classe = np.where(y == i)[0]
        echantillons_indices = np.random.choice(indices_classe, 
                                               min(nb_echantillons, len(indices_classe)), 
                                               replace=False)
        
        for j, idx in enumerate(echantillons_indices):
            ax = axes[i, j] if len(noms_classes) > 1 else axes[j]
            ax.imshow(X[idx])
            ax.set_title(f"{nom_classe}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Échantillons d'Images par Classe", fontsize=16, y=1.02)
    plt.show()


# Test du module
if __name__ == "__main__":
    """
    Code de test pour vérifier le fonctionnement du préprocesseur.
    """
    print("🧪 Test du Module de Prétraitement")
    print("=" * 50)
    
    # Créer le préprocesseur
    preprocesseur = PreprocesseurDonnees()
    
    # Test avec une image factice
    print("\n🔬 Test de chargement d'image factice...")
    image_test = np.random.random((100, 100, 3)).astype(np.float32)
    
    # Test d'augmentation
    print("🔬 Test d'augmentation d'image...")
    images_augmentees = preprocesseur.augmenter_image(image_test)
    print(f"✅ Image originale -> {len(images_augmentees)} images augmentées")
    
    # Test de l'encodeur de labels
    print("\n🔬 Test de l'encodeur de labels...")
    for nom_fruit in preprocesseur.noms_classes:
        label_encode = preprocesseur.encodeur_labels.transform([nom_fruit])[0]
        print(f"   {nom_fruit} -> {label_encode}")
    
    print("\n✅ Tous les tests passés!")
    print("📌 Prêt à charger les vraies données du dataset Fruits-360")