"""
Chargement Manuel des Données - Bypass du Bug d'Augmentation
"""

import os
import sys
import numpy as np
from datetime import datetime

sys.path.append('src')
from data_preprocessing import PreprocesseurDonnees
from model_architecture import creer_modele_fruivision

def charger_donnees_manuellement():
    """Charger les données en bypassant le bug d'augmentation."""
    
    print("🔧 CHARGEMENT MANUEL - BYPASS DU BUG")
    print("=" * 50)
    
    # Classes et preprocesseur
    classes_fruits = {
        'Pomme': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Braeburn 1',
                 'Apple Granny Smith 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Red 1'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    preprocesseur = PreprocesseurDonnees(classes_fruits=classes_fruits)
    
    # CHARGEMENT MANUEL ÉQUILIBRÉ
    print("\n📥 Chargement équilibré (200 images par fruit)...")
    
    toutes_images = []
    tous_labels = []
    
    for i, (nom_fruit, categories) in enumerate(classes_fruits.items()):
        print(f"🍎 {nom_fruit}:")
        
        images_fruit = []
        
        # Charger exactement 200 images par fruit
        for categorie in categories:
            chemin_categorie = os.path.join("data/fruits-360/Training", categorie)
            
            if os.path.exists(chemin_categorie):
                fichiers = [f for f in os.listdir(chemin_categorie) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Limiter strictement
                nb_a_prendre = min(200 // len(categories), len(fichiers))
                fichiers_pris = fichiers[:nb_a_prendre]
                
                print(f"   📁 {categorie}: {len(fichiers_pris)} images")
                
                for fichier in fichiers_pris:
                    chemin_complet = os.path.join(chemin_categorie, fichier)
                    image = preprocesseur.charger_image(chemin_complet)
                    if image is not None:
                        images_fruit.append(image)
        
        # Limiter à exactement 200 par fruit
        if len(images_fruit) > 200:
            images_fruit = images_fruit[:200]
        
        print(f"   ✅ Total {nom_fruit}: {len(images_fruit)} images")
        
        # Ajouter au dataset global
        toutes_images.extend(images_fruit)
        tous_labels.extend([i] * len(images_fruit))
    
    X = np.array(toutes_images, dtype=np.float32)
    y = np.array(tous_labels, dtype=np.int32)
    
    print(f"\n📊 Dataset équilibré: {X.shape[0]} images total")
    
    # Vérifier la distribution
    print("📊 Distribution finale:")
    for i, fruit in enumerate(preprocesseur.noms_classes):
        nb = np.sum(y == i)
        print(f"   {fruit:8}: {nb} images")
    
    return X, y, preprocesseur

def entrainement_equilibre():
    """Entraînement avec données parfaitement équilibrées."""
    
    print("🎯 ENTRAÎNEMENT ÉQUILIBRÉ - VRAIE BASELINE")
    print("=" * 60)
    
    # Chargement manuel
    X, y, preprocesseur = charger_donnees_manuellement()
    
    # Division équitable
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelBinarizer
    
    # Test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train et validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # One-hot encoding
    binarizer = LabelBinarizer()
    y_train_onehot = binarizer.fit_transform(y_train)
    y_val_onehot = binarizer.transform(y_val)
    y_test_onehot = binarizer.transform(y_test)
    
    print(f"\n✅ Division équilibrée:")
    print(f"   📚 Train: {X_train.shape[0]} images")
    print(f"   🔍 Val: {X_val.shape[0]} images")
    print(f"   🧪 Test: {X_test.shape[0]} images")
    
    # Modèle
    modele = creer_modele_fruivision()
    modele.optimizer.learning_rate.assign(0.001)
    
    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('models/fruivision_equilibre.h5', save_best_only=True)
    ]
    
    # Entraînement
    print(f"\n🚀 Entraînement sur données équilibrées...")
    
    debut = datetime.now()
    
    historique = modele.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    fin = datetime.now()
    print(f"\n⏱️ Temps d'entraînement: {fin - debut}")
    
    # Test final
    print(f"\n🧪 ÉVALUATION FINALE:")
    
    # Charger le meilleur modèle
    import tensorflow as tf
    meilleur_modele = tf.keras.models.load_model('models/fruivision_equilibre.h5')
    
    # Test sur données de test
    loss_test, acc_test = meilleur_modele.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"   🎯 Test Accuracy: {acc_test:.1%}")
    
    # Test sur vraies images DIFFÉRENTES
    print(f"\n🍎 Test sur nouvelles images du dataset:")
    
    test_images = [
        ('data/fruits-360/Training/Apple Golden 1', 'Pomme', 0),
        ('data/fruits-360/Training/Banana 1', 'Banane', 1),
        ('data/fruits-360/Training/Kiwi 1', 'Kiwi', 2),
        ('data/fruits-360/Training/Lemon 1', 'Citron', 3),
        ('data/fruits-360/Training/Peach 1', 'Pêche', 4)
    ]
    
    resultats_vrais = []
    
    for dossier, fruit_attendu, classe_attendue in test_images:
        if os.path.exists(dossier):
            fichiers = [f for f in os.listdir(dossier) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(fichiers) > 250:  # Prendre une image NON utilisée en training
                fichier_test = fichiers[250]  # Image 250 = pas dans les 200 premières
                chemin = os.path.join(dossier, fichier_test)
                
                image = preprocesseur.charger_image(chemin)
                if image is not None:
                    pred = meilleur_modele.predict(np.expand_dims(image, axis=0), verbose=0)
                    classe_pred = np.argmax(pred[0])
                    confiance = pred[0][classe_pred]
                    fruit_pred = preprocesseur.noms_classes[classe_pred]
                    
                    succes = fruit_pred == fruit_attendu
                    resultats_vrais.append(succes)
                    
                    print(f"   {fruit_attendu:8} → {fruit_pred:8} ({confiance:.3f}) {'✅' if succes else '❌'}")
    
    acc_vrais = sum(resultats_vrais) / len(resultats_vrais) if resultats_vrais else 0
    
    print(f"\n📊 RÉSULTATS FINAUX HONNÊTES:")
    print(f"   🧪 Test set: {acc_test:.1%}")
    print(f"   🍎 Nouvelles images: {acc_vrais:.1%}")
    
    if acc_test > 0.8 and acc_vrais > 0.6:
        print(f"\n🎉 SUCCÈS! Modèle équilibré fonctionne!")
    else:
        print(f"\n✅ Performances honnêtes (pas de faux 100%)")
    
    return True

if __name__ == "__main__":
    print("🎯 ENTRAÎNEMENT ÉQUILIBRÉ - FRUIVISION HONNÊTE")
    print("Données parfaitement équilibrées, pas de bug d'augmentation")
    print("=" * 60)
    
    entrainement_equilibre()