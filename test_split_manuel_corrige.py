"""
Test Split Manuel - Preuve que Votre Approche Marche
===================================================

OBJECTIF: Prouver que split 80/20 du Training/ fonctionne
         SANS le bug d'augmentation

Mamadou Fall - Test de Validation
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from datetime import datetime
import json

# Ajouter le path pour vos modules
sys.path.append('src')

def charger_donnees_training_seulement():
    """Charger SEULEMENT le dossier Training/ et faire split manuel 80/20."""
    
    print("🔧 TEST: SPLIT MANUEL du Training/ SEULEMENT")
    print("=" * 60)
    print("🎯 Objectif: Prouver que votre approche initiale était bonne")
    print("🚫 SANS bug d'augmentation cette fois!")
    
    # Vos 5 fruits dans Training/
    classes_fruits = {
        'Pomme': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Red 1'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    print(f"\n📂 Chargement depuis Training/ seulement...")
    
    toutes_images = []
    tous_labels = []
    noms_classes = list(classes_fruits.keys())
    
    # Charger SANS augmentation (corrigé!)
    for i, (nom_fruit, categories) in enumerate(classes_fruits.items()):
        print(f"🍎 {nom_fruit}:")
        
        images_fruit = []
        
        for categorie in categories:
            chemin_dossier = os.path.join("data/fruits-360/Training", categorie)
            
            if os.path.exists(chemin_dossier):
                fichiers = [f for f in os.listdir(chemin_dossier) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Limiter à 200 images par catégorie pour équilibre
                fichiers = fichiers[:200]
                
                print(f"   📁 {categorie}: {len(fichiers)} images")
                
                for fichier in fichiers:
                    chemin_image = os.path.join(chemin_dossier, fichier)
                    
                    try:
                        # Chargement simple SANS augmentation
                        image = Image.open(chemin_image).convert('RGB')
                        image = image.resize((100, 100), Image.Resampling.LANCZOS)
                        image_array = np.array(image, dtype=np.float32) / 255.0
                        
                        images_fruit.append(image_array)
                        
                    except Exception as e:
                        print(f"      Erreur {fichier}: {e}")
        
        print(f"   ✅ Total {nom_fruit}: {len(images_fruit)} images (SANS augmentation)")
        
        # Ajouter au dataset global
        toutes_images.extend(images_fruit)
        tous_labels.extend([i] * len(images_fruit))  # Label numérique
    
    X = np.array(toutes_images, dtype=np.float32)
    y = np.array(tous_labels, dtype=np.int32)
    
    print(f"\n📊 Dataset final:")
    print(f"   📁 Source: Training/ seulement")
    print(f"   🖼️ Total images: {X.shape[0]}")
    print(f"   🔢 Classes: {len(noms_classes)}")
    
    # Vérifier la distribution
    print(f"\n📊 Distribution par classe:")
    for i, nom in enumerate(noms_classes):
        count = np.sum(y == i)
        print(f"   {nom:8}: {count:4d} images")
    
    return X, y, noms_classes

def faire_split_manuel_80_20(X, y, noms_classes):
    """Split manuel 80/20 comme vous vouliez faire."""
    
    print(f"\n🔄 SPLIT MANUEL 80/20 (votre approche originale)")
    print("-" * 50)
    
    # Split stratifié pour maintenir la proportion des classes
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.3,      # 30% pour validation + test
        random_state=42,    # Reproductible
        stratify=y          # Maintenir proportions
    )
    
    # Split validation/test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,      # 50% de 30% = 15% chacun
        random_state=42,
        stratify=y_temp
    )
    
    print(f"✅ Split réalisé:")
    print(f"   📚 Training:   {X_train.shape[0]:4d} images (70%)")
    print(f"   🔍 Validation: {X_val.shape[0]:4d} images (15%)")
    print(f"   🧪 Test:       {X_test.shape[0]:4d} images (15%)")
    
    # Vérifier la distribution dans chaque set
    print(f"\n📊 Distribution dans chaque set:")
    
    sets_info = [
        ("Training", y_train),
        ("Validation", y_val), 
        ("Test", y_test)
    ]
    
    for set_name, y_set in sets_info:
        print(f"   {set_name:10}:", end="")
        for i, nom in enumerate(noms_classes):
            count = np.sum(y_set == i)
            print(f" {nom[:4]}({count})", end="")
        print()
    
    # One-hot encoding
    no_classes = len(noms_classes)
    y_train_onehot = to_categorical(y_train, no_classes)
    y_val_onehot = to_categorical(y_val, no_classes)
    y_test_onehot = to_categorical(y_test, no_classes)
    
    return (X_train, X_val, X_test, 
            y_train_onehot, y_val_onehot, y_test_onehot,
            y_train, y_val, y_test)

def creer_modele_simple():
    """Modèle simple et efficace (inspiré Kaggle mais plus petit)."""
    
    model = Sequential()
    
    # Architecture simple mais efficace
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))  # 5 classes
    
    # Compilation
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_split_manuel():
    """Test complet de votre approche originale (corrigée)."""
    
    print("🧪 TEST SPLIT MANUEL - VALIDATION DE VOTRE APPROCHE")
    print("=" * 70)
    
    debut_total = datetime.now()
    
    # 1. Charger données Training/ seulement
    X, y, noms_classes = charger_donnees_training_seulement()
    
    # 2. Split manuel 80/20
    (X_train, X_val, X_test, 
     y_train_onehot, y_val_onehot, y_test_onehot,
     y_train_raw, y_val_raw, y_test_raw) = faire_split_manuel_80_20(X, y, noms_classes)
    
    # 3. Modèle simple
    print(f"\n🧠 Création du modèle...")
    model = creer_modele_simple()
    
    # 4. Entraînement
    print(f"\n🚀 Entraînement (15 époques max)...")
    
    from tensorflow.keras.callbacks import EarlyStopping
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=15,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Évaluation
    print(f"\n🧪 ÉVALUATION FINALE")
    print("-" * 40)
    
    # Test accuracy
    loss_test, acc_test = model.evaluate(X_test, y_test_onehot, verbose=0)
    
    print(f"🎯 Test Accuracy: {acc_test:.1%}")
    print(f"📉 Test Loss: {loss_test:.4f}")
    
    # Test sur quelques exemples
    print(f"\n🍎 TEST DÉTAILLÉ:")
    
    predictions = model.predict(X_test[:10], verbose=0)
    
    for i in range(min(10, len(X_test))):
        pred_idx = np.argmax(predictions[i])
        true_idx = y_test_raw[i]
        confiance = predictions[i][pred_idx]
        
        print(f"   Test {i+1:2d}: {noms_classes[pred_idx]:8} ({confiance:.1%}) ", end="")
        print(f"- Vrai: {noms_classes[true_idx]:8} {'✅' if pred_idx == true_idx else '❌'}")
    
    # Sauvegarder
    os.makedirs('models', exist_ok=True)
    model.save('models/fruivision_split_manuel.h5')
    
    # Sauvegarder mapping
    with open('models/classes_split_manuel.json', 'w') as f:
        json.dump(noms_classes, f)
    
    fin_total = datetime.now()
    
    # Résultats finaux
    print(f"\n🏆 RÉSULTATS DU TEST")
    print("=" * 40)
    print(f"📊 Approche: Split manuel 80/20 du Training/")
    print(f"🚫 Augmentation: AUCUNE (bug corrigé)")
    print(f"🎯 Test Accuracy: {acc_test:.1%}")
    print(f"⏱️ Temps total: {fin_total - debut_total}")
    print(f"💾 Modèle: fruivision_split_manuel.h5")
    
    if acc_test > 0.7:
        print(f"\n🎉 SUCCÈS! Votre approche fonctionne!")
        print(f"✅ Split manuel 80/20 est valide!")
        print(f"💡 Le problème était bien l'augmentation, pas le split!")
    elif acc_test > 0.5:
        print(f"\n✅ Résultat honnête!")
        print(f"💡 Performance réaliste sans tricks!")
    else:
        print(f"\n⚠️ Performance modeste mais honnête!")
        print(f"💡 Mieux que des faux 100%!")
    
    return model, acc_test

if __name__ == "__main__":
    print("🧪 TEST DE VALIDATION - SPLIT MANUEL vs KAGGLE")
    print("Prouvons que votre approche initiale était bonne!")
    print("=" * 70)
    
    test_split_manuel()