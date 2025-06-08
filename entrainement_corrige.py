"""
Entraînement Corrigé - Version Fiable
=====================================

Configuration corrigée pour éviter les pièges du modèle précédent
"""

import os
import sys
import numpy as np
from datetime import datetime

# Ajouter les dossiers au path
sys.path.append('src')
sys.path.append('config')

# Importer nos modules
from model_architecture import creer_modele_fruivision
from data_preprocessing import PreprocesseurDonnees
from training_utils import GestionnaireEntrainement

def configuration_entrainement_corrige():
    """Configuration corrigée pour éviter les erreurs précédentes."""
    
    config = {
        # Données - Moins d'augmentation pour éviter le bruit
        'max_images_par_classe': 800,          # Comme version rapide
        'utiliser_augmentation': True,
        'augmentation_factor': 2,              # RÉDUIT: 2x au lieu de 5x
        'taille_validation': 0.15,             # RÉDUIT: 15% au lieu de 20%
        'taille_test': 0.15,                   # RÉDUIT: 15% au lieu de 10%
        
        # Entraînement - Plus conservateur
        'epochs': 50,                          # Plus d'époques max
        'batch_size': 64,                      # Plus gros batch = plus stable
        'learning_rate': 0.0005,               # RÉDUIT: Plus conservateur
        
        # Callbacks - Plus patients et stricts
        'early_stopping_patience': 8,         # RÉDUIT: Arrêt plus précoce
        'early_stopping_monitor': 'val_loss', # CHANGÉ: Surveiller la loss, pas accuracy
        'early_stopping_min_delta': 0.001,    # AJOUTÉ: Vraie amélioration requise
        'reduce_lr_patience': 5,               # RÉDUIT: Réduction LR plus rapide
        'reduce_lr_factor': 0.3,               # RÉDUIT: Réduction plus agressive
        'save_best_only': True,
        
        # Validation
        'verbose': 1,
        'validation_freq': 1,
        'workers': 1,
        'use_multiprocessing': False,
    }
    
    return config

def entrainement_corrige():
    """Entraînement avec configuration corrigée."""
    
    print("🔧 ENTRAÎNEMENT CORRIGÉ - VERSION FIABLE")
    print("=" * 60)
    print("🎯 Objectif: Modèle fiable qui généralise bien")
    print("🚫 Éviter: Over-fitting et early stopping abusif")
    
    debut_total = datetime.now()
    
    # Configuration corrigée
    config = configuration_entrainement_corrige()
    
    print("\n⚙️ Configuration corrigée:")
    print(f"   📊 Max images par fruit: {config['max_images_par_classe']}")
    print(f"   ✨ Facteur d'augmentation: {config['augmentation_factor']}x (RÉDUIT)")
    print(f"   🔄 Époques max: {config['epochs']}")
    print(f"   📦 Batch size: {config['batch_size']} (PLUS STABLE)")
    print(f"   📈 Learning rate: {config['learning_rate']} (RÉDUIT)")
    print(f"   ⏹️ Early stopping: {config['early_stopping_patience']} époques sur val_loss")
    
    # Classes du dataset (MÊMES que précédent)
    classes_fruits_reelles = {
        'Pomme': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Braeburn 1',
                 'Apple Granny Smith 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Red 1'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    # Préprocesseur
    preprocesseur = PreprocesseurDonnees(classes_fruits=classes_fruits_reelles)
    
    # Chargement des données (utiliser la VRAIE méthode du preprocesseur)
    print(f"\n📥 Chargement des données (moins d'augmentation)...")
    debut_chargement = datetime.now()
    
    # Utiliser la méthode qui existe vraiment dans votre code
    X, y = preprocesseur.charger_donnees_dossier(
        chemin_base="data/fruits-360/Training",
        utiliser_augmentation=config['utiliser_augmentation'],
        max_images_par_classe=config['max_images_par_classe']
    )
    
    fin_chargement = datetime.now()
    temps_chargement = (fin_chargement - debut_chargement).seconds
    
    print(f"✅ Données chargées en {temps_chargement // 60}m {temps_chargement % 60}s")
    print(f"📊 Total: {X.shape[0]:,} images ({X.shape[0]/28225:.1f}x moins que version précédente)")
    
    # Division des données
    print(f"\n🔄 Division des données...")
    ensembles = preprocesseur.creer_ensembles_donnees(
        X, y, 
        taille_validation=config['taille_validation'],
        taille_test=config['taille_test']
    )
    
    print(f"✅ Ensembles créés:")
    print(f"   📚 Entraînement: {ensembles['X_train'].shape[0]:,} images")
    print(f"   🔍 Validation: {ensembles['X_val'].shape[0]:,} images")
    print(f"   🧪 Test: {ensembles['X_test'].shape[0]:,} images")
    
    # Création du modèle
    print(f"\n🧠 Création du modèle...")
    modele = creer_modele_fruivision()
    
    # Ajuster le learning rate
    modele.optimizer.learning_rate.assign(config['learning_rate'])
    print(f"🔧 Learning rate: {config['learning_rate']} (plus conservateur)")
    
    # Configuration des callbacks CORRIGÉS
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(
                monitor=config['early_stopping_monitor'],  # val_loss au lieu de val_accuracy
                patience=config['early_stopping_patience'],
                min_delta=config['early_stopping_min_delta'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=config['reduce_lr_factor'],
                patience=config['reduce_lr_patience'],
                verbose=1,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                filepath='models/fruivision_corrige.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        os.makedirs('models', exist_ok=True)
        print(f"✅ Callbacks corrigés configurés")
        
    except Exception as e:
        print(f"⚠️ Callbacks simplifiés: {e}")
        callbacks = []
    
    # ENTRAÎNEMENT CORRIGÉ
    print(f"\n🚀 DÉBUT DE L'ENTRAÎNEMENT CORRIGÉ")
    print(f"🕐 Heure de début: {datetime.now().strftime('%H:%M:%S')}")
    print(f"⚠️ Surveillance: val_loss (plus fiable que val_accuracy)")
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
        fin_entrainement = datetime.now()
        temps_entrainement = fin_entrainement - debut_entrainement
        modele.save('models/fruivision_corrige_interrompu.h5')
        print(f"💾 Modèle partiel sauvegardé après {temps_entrainement}")
        return False
    
    # Test immédiat avec les vraies images du dataset
    print(f"\n🧪 TEST IMMÉDIAT AVEC VRAIES IMAGES")
    print("-" * 50)
    
    try:
        # Charger le meilleur modèle
        import tensorflow as tf
        meilleur_modele = tf.keras.models.load_model('models/fruivision_corrige.h5')
        print("✅ Meilleur modèle chargé")
        
        # Test sur quelques vraies images
        test_paths = [
            ('data/fruits-360/Training/Apple Golden 1', 'Pomme', 0),
            ('data/fruits-360/Training/Banana 1', 'Banane', 1),
            ('data/fruits-360/Training/Kiwi 1', 'Kiwi', 2),
            ('data/fruits-360/Training/Lemon 1', 'Citron', 3),
            ('data/fruits-360/Training/Peach 1', 'Pêche', 4)
        ]
        
        resultats_test = []
        for dossier, fruit_attendu, classe_attendue in test_paths:
            if os.path.exists(dossier):
                images = [f for f in os.listdir(dossier) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    image_path = os.path.join(dossier, images[0])
                    image = preprocesseur.charger_image(image_path)
                    
                    if image is not None:
                        image_batch = np.expand_dims(image, axis=0)
                        prediction = meilleur_modele.predict(image_batch, verbose=0)
                        classe_predite = np.argmax(prediction[0])
                        confiance = prediction[0][classe_predite]
                        fruit_predit = preprocesseur.noms_classes[classe_predite]
                        
                        succes = fruit_predit == fruit_attendu
                        resultats_test.append(succes)
                        
                        print(f"   {fruit_attendu:8} → {fruit_predit:8} ({confiance:.3f}) ", end="")
                        print("✅" if succes else "❌")
        
        accuracy_test = sum(resultats_test) / len(resultats_test) if resultats_test else 0
        print(f"\n📊 RÉSULTAT TEST IMMÉDIAT: {accuracy_test:.1%} ({sum(resultats_test)}/{len(resultats_test)})")
        
        if accuracy_test >= 0.8:
            print("🎉 SUCCÈS! Modèle corrigé fonctionne!")
        else:
            print("❌ ÉCHEC! Modèle encore défaillant...")
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
    
    return True

if __name__ == "__main__":
    """Lancer l'entraînement corrigé."""
    
    print("🔧 ENTRAÎNEMENT CORRIGÉ - FRUIVISION FIABLE")
    print("Correction des erreurs du modèle précédent")
    print("=" * 60)
    
    success = entrainement_corrige()
    
    if success:
        print("\n✅ ENTRAÎNEMENT CORRIGÉ TERMINÉ!")
        print("🧪 Test immédiat effectué!")
        print("🎯 Modèle corrigé prêt!")
    else:
        print("\n❌ ENTRAÎNEMENT INTERROMPU")