"""
Edufruits - Entraînement Optimisé
===================================

Version optimisée pour un entraînement réaliste en 1-2 heures.
Configuration ajustée pour votre PC avec 8GB RAM.

Auteur: Mamadou Fall
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
from typing import List, Dict

# Ajouter les dossiers au path
sys.path.append('src')
sys.path.append('config')

# Importer nos modules
from model_architecture import creer_modele_fruivision
from data_preprocessing import PreprocesseurDonnees
from training_utils import GestionnaireEntrainement

def configuration_entrainement_optimise():
    """Configuration optimisée pour entraînement réaliste."""
    
    config = {
        # Données - Limitation raisonnable
        'max_images_par_classe': 800,          # Limite à 800 images par fruit
        'utiliser_augmentation': True,          # Augmentation modérée
        'augmentation_factor': 3,               # Multiplier par 3 seulement
        'taille_validation': 0.2,              # 20% pour validation
        'taille_test': 0.1,                    # 10% pour test
        
        # Entraînement - Configuration réaliste
        'epochs': 30,                           # 30 époques suffisantes
        'batch_size': 64,                       # Batch plus grand = plus rapide
        'learning_rate': 0.001,                 # Taux d'apprentissage standard
        
        # Callbacks optimisés
        'early_stopping_patience': 10,         # Arrêt précoce plus agressif
        'reduce_lr_patience': 5,               # Réduction LR plus rapide
        'save_best_only': True,                # Sauver seulement le meilleur
        
        # Performance
        'verbose': 1,                          # Affichage détaillé
        'validation_freq': 1,                  # Validation à chaque époque
        'workers': 2,                          # Parallélisation
        'use_multiprocessing': False,          # Éviter surcharge RAM
    }
    
    return config

class PreprocesseurOptimise(PreprocesseurDonnees):
    """Version optimisée du preprocesseur pour entraînement rapide."""
    
    def augmenter_image(self, image: np.ndarray, factor=3) -> List[np.ndarray]:
        """
        Augmentation modérée pour éviter explosion des données.
        
        Args:
            image: Image originale
            factor: Facteur d'augmentation (3 = 3 images générées)
        """
        
        if factor <= 1:
            return []
        
        images_augmentees = []
        
        # Convertir en PIL
        from PIL import Image as PILImage, ImageEnhance
        img_pil = PILImage.fromarray((image * 255).astype(np.uint8))
        
        # Seulement les augmentations les plus utiles
        augmentations = [
            # Rotation légère
            lambda img: img.rotate(15, fillcolor=(255, 255, 255)),
            # Luminosité
            lambda img: ImageEnhance.Brightness(img).enhance(1.2),
            # Miroir horizontal
            lambda img: img.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT),
        ]
        
        # Appliquer seulement le nombre d'augmentations demandé
        for i in range(min(factor - 1, len(augmentations))):
            try:
                img_aug = augmentations[i](img_pil)
                img_array = np.array(img_aug) / 255.0
                images_augmentees.append(img_array)
            except:
                continue
        
        return images_augmentees
    
    def charger_donnees_optimise(self, chemin_base: str, config: dict):
        """
        Chargement optimisé avec contrôle précis du nombre d'images.
        
        Args:
            chemin_base: Chemin vers les données
            config: Configuration d'entraînement
        """
        
        print(f"📂 Chargement optimisé des données depuis: {chemin_base}")
        
        toutes_images = []
        tous_labels = []
        
        for nom_fruit, categories_dataset in self.classes_fruits.items():
            print(f"🍎 Traitement du fruit: {nom_fruit}")
            
            images_fruit = []
            images_chargees = 0
            max_par_fruit = config['max_images_par_classe']
            
            # Charger images originales
            for categorie in categories_dataset:
                if images_chargees >= max_par_fruit:
                    break
                    
                chemin_categorie = os.path.join(chemin_base, categorie)
                if not os.path.exists(chemin_categorie):
                    continue
                
                fichiers_images = [f for f in os.listdir(chemin_categorie) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Limiter le nombre d'images à charger
                reste_a_charger = max_par_fruit - images_chargees
                fichiers_a_charger = fichiers_images[:reste_a_charger]
                
                print(f"   📁 {categorie}: chargement de {len(fichiers_a_charger)} images")
                
                for nom_fichier in fichiers_a_charger:
                    chemin_complet = os.path.join(chemin_categorie, nom_fichier)
                    image = self.charger_image(chemin_complet)
                    images_fruit.append(image)
                    images_chargees += 1
            
            print(f"   📊 Images originales: {len(images_fruit)}")
            
            # Augmentation contrôlée
            if config['utiliser_augmentation']:
                images_augmentees_total = []
                factor = config['augmentation_factor']
                
                for image in images_fruit:
                    augmentees = self.augmenter_image(image, factor)
                    images_augmentees_total.extend(augmentees)
                
                images_fruit.extend(images_augmentees_total)
                print(f"   ✨ Avec augmentation (x{factor}): {len(images_fruit)} images")
            
            # Ajouter au dataset global
            toutes_images.extend(images_fruit)
            
            # Labels correspondants
            label_encode = self.encodeur_labels.transform([nom_fruit])[0]
            labels_fruit = [label_encode] * len(images_fruit)
            tous_labels.extend(labels_fruit)
            
            print(f"   ✅ {nom_fruit}: {len(images_fruit)} images finales")
        
        X = np.array(toutes_images, dtype=np.float32)
        y = np.array(tous_labels, dtype=np.int32)
        
        print(f"📊 Dataset optimisé chargé: {X.shape[0]:,} images, {len(self.noms_classes)} classes")
        
        return X, y

def estimer_temps_realiste(nb_images, epochs, batch_size):
    """Estimation plus réaliste basée sur les vrais tests."""
    
    # Basé sur votre test: 595 images, 5 époques = 9 secondes
    # Donc: ~0.003 secondes par image par époque
    temps_par_image_par_epoque = 0.003
    
    temps_total_secondes = nb_images * epochs * temps_par_image_par_epoque
    temps_total_minutes = temps_total_secondes / 60
    temps_total_heures = temps_total_minutes / 60
    
    return {
        'temps_total_minutes': temps_total_minutes,
        'temps_total_heures': temps_total_heures,
        'temps_par_epoque_minutes': temps_total_minutes / epochs
    }

def entrainement_optimise():
    """Entraînement optimisé pour votre configuration."""
    
    print("🚀 ENTRAÎNEMENT OPTIMISÉ FRUIVISION")
    print("=" * 60)
    
    debut_total = datetime.now()
    
    # Configuration optimisée
    config = configuration_entrainement_optimise()
    
    print("⚙️ Configuration optimisée:")
    print(f"   Max images par fruit: {config['max_images_par_classe']}")
    print(f"   Facteur d'augmentation: {config['augmentation_factor']}x")
    print(f"   Époques: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    
    # Classes du dataset
    classes_fruits_reelles = {
        'Pomme': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Braeburn 1',
                 'Apple Granny Smith 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Red 1'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    # Préprocesseur optimisé
    preprocesseur = PreprocesseurOptimise(classes_fruits=classes_fruits_reelles)
    
    # Chargement des données
    print(f"\n📥 Chargement des données (estimation: 2-3 minutes)...")
    debut_chargement = datetime.now()
    
    X, y = preprocesseur.charger_donnees_optimise("data/fruits-360/Training", config)
    
    fin_chargement = datetime.now()
    temps_chargement = (fin_chargement - debut_chargement).seconds
    
    print(f"✅ Données chargées en {temps_chargement // 60}m {temps_chargement % 60}s")
    
    # Estimation réaliste du temps
    estimation = estimer_temps_realiste(X.shape[0], config['epochs'], config['batch_size'])
    
    print(f"\n⏱️ ESTIMATION RÉALISTE:")
    print(f"   📊 Images totales: {X.shape[0]:,}")
    print(f"   🔄 Époques: {config['epochs']}")
    print(f"   ⏱️ Temps par époque: ~{estimation['temps_par_epoque_minutes']:.1f} minutes")
    print(f"   🕐 Temps total estimé: ~{estimation['temps_total_heures']:.1f} heures")
    
    # Validation de l'estimation
    if estimation['temps_total_heures'] > 4:
        print(f"\n⚠️ ATTENTION: Temps estimé trop long ({estimation['temps_total_heures']:.1f}h)")
        print(f"💡 Suggestions:")
        print(f"   - Réduire max_images_par_classe à 500")
        print(f"   - Réduire les époques à 20")
        print(f"   - Augmenter batch_size à 128")
        
        continuer = input("\n   Continuer quand même? (o/n): ").lower().strip()
        if continuer not in ['o', 'oui', 'y', 'yes']:
            print("❌ Entraînement annulé.")
            return False
    
    # Division des données
    print(f"\n🔄 Division des données...")
    ensembles = preprocesseur.creer_ensembles_donnees(
        X, y, 
        taille_validation=config['taille_validation'],
        taille_test=config['taille_test']
    )
    
    # Création du modèle
    print(f"\n🧠 Création du modèle...")
    modele = creer_modele_fruivision()
    
    # Configuration des callbacks
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=config['reduce_lr_patience'],
                verbose=1
            ),
            ModelCheckpoint(
                filepath='models/fruivision_optimise.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        os.makedirs('models', exist_ok=True)
        print(f"✅ Callbacks configurés")
        
    except Exception as e:
        print(f"⚠️ Callbacks simplifiés: {e}")
        callbacks = []
    
    # ENTRAÎNEMENT
    print(f"\n🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print(f"🕐 Heure de début: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    debut_entrainement = datetime.now()
    
    try:
        historique = modele.fit(
            ensembles['X_train'], ensembles['y_train'],
            validation_data=(ensembles['X_val'], ensembles['y_val']),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=config['verbose'],
            shuffle=True
        )
        
        fin_entrainement = datetime.now()
        temps_entrainement = fin_entrainement - debut_entrainement
        
        print(f"\n🎉 ENTRAÎNEMENT TERMINÉ!")
        print(f"⏱️ Temps réel: {temps_entrainement}")
        
    except KeyboardInterrupt:
        print(f"\n⛔ Entraînement interrompu")
        modele.save('models/fruivision_interrompu.h5')
        return False
    
    # Évaluation finale
    print(f"\n📊 ÉVALUATION FINALE")
    print("-" * 40)
    
    try:
        # Charger le meilleur modèle
        import tensorflow as tf
        meilleur_modele = tf.keras.models.load_model('models/fruivision_optimise.h5')
        print("✅ Meilleur modèle chargé")
    except:
        meilleur_modele = modele
        print("⚠️ Utilisation du modèle final")
    
    # Test final
    resultats = meilleur_modele.evaluate(ensembles['X_test'], ensembles['y_test'], verbose=0)
    accuracy_finale = resultats[1]
    
    # Sauvegarde finale
    meilleur_modele.save('models/fruivision_final_optimise.h5')
    
    # Résultats
    resultats_finaux = {
        'accuracy_finale': float(accuracy_finale),
        'temps_entrainement': str(temps_entrainement),
        'epochs_completees': len(historique.history['accuracy']),
        'nb_images_entrainement': int(ensembles['X_train'].shape[0]),
        'config': config
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/entrainement_optimise_resultats.json', 'w') as f:
        json.dump(resultats_finaux, f, indent=2)
    
    # Résumé final
    print(f"\n🎉 RÉSUMÉ FINAL")
    print("=" * 50)
    print(f"🎯 Précision finale: {accuracy_finale:.4f} ({accuracy_finale*100:.2f}%)")
    print(f"⏱️ Temps d'entraînement: {temps_entrainement}")
    print(f"🔄 Époques completées: {len(historique.history['accuracy'])}")
    print(f"💾 Modèle sauvegardé: models/fruivision_final_optimise.h5")
    
    if accuracy_finale > 0.90:
        print(f"🎉 EXCELLENT! Modèle prêt pour production!")
    elif accuracy_finale > 0.85:
        print(f"✅ TRÈS BON! Objectif atteint!")
    else:
        print(f"👍 BON début, peut être amélioré")
    
    return True

if __name__ == "__main__":
    """Lancer l'entraînement optimisé."""
    
    print("🚀 ENTRAÎNEMENT OPTIMISÉ FRUIVISION")
    print("Configuration adaptée à votre PC (8GB RAM)")
    print("Temps estimé: 1-2 heures maximum")
    print("=" * 60)
    
    success = entrainement_optimise()
    
    if success:
        print("\n✅ ENTRAÎNEMENT RÉUSSI!")
        print("🚀 Modèle prêt à être testé dans l'application Streamlit!")
    else:
        print("\n❌ ENTRAÎNEMENT INTERROMPU")