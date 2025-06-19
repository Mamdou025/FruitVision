#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test adapté à votre structure de dossiers
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

def test_avec_votre_structure():
    """Test avec votre structure de dossiers"""
    
    print("🔍 TEST SUR VOTRE DATASET FRUITS-360")
    print("="*50)
    
    # Charger le modèle
    try:
        model = tf.keras.models.load_model('models/fruivision_split_manuel.h5')
        classes = ['Pomme', 'Banane', 'Kiwi', 'Citron', 'Peche']
        print("✅ Modèle chargé")
        print(f"✅ Classes du modèle: {classes}")
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return
    
    # Trouver les dossiers disponibles
    class_mappings = {
        'Pomme': ['Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Golden 1', 'Apple 10'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    available_mappings = {}
    base_test_dir = "data/fruits-360/test"
    
    print(f"\n🔍 Vérification des correspondances:")
    for our_class, dataset_folders in class_mappings.items():
        found_folder = None
        for folder in dataset_folders:
            folder_path = os.path.join(base_test_dir, folder)
            if os.path.exists(folder_path):
                found_folder = folder
                break
        
        if found_folder:
            print(f"   {our_class}: ✅ {found_folder}")
            available_mappings[our_class] = found_folder
        else:
            print(f"   {our_class}: ❌ Aucun dossier trouvé parmi {dataset_folders}")
    
    if not available_mappings:
        print("❌ Aucune classe correspondante trouvée")
        return
    
    print(f"\n🧪 TEST EN COURS:")
    print("="*30)
    
    total_correct = 0
    total_tested = 0
    base_test_dir = "data/fruits-360/test"
    
    for our_class, dataset_folder in available_mappings.items():
        folder_path = os.path.join(base_test_dir, dataset_folder)
        
        print(f"\n🍎 Test {dataset_folder} -> {our_class}:")
        
        # Prendre quelques images
        images = glob.glob(os.path.join(folder_path, "*.jpg"))[:5]
        
        if not images:
            print("   ❌ Aucune image .jpg trouvée")
            continue
        
        class_correct = 0
        for img_path in images:
            try:
                # Charger et préprocesser l'image
                image = Image.open(img_path).convert('RGB')
                image = image.resize((100, 100))
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                
                # Prédiction
                pred = model.predict(image_array, verbose=0)
                pred_idx = np.argmax(pred[0])
                confidence = pred[0][pred_idx]
                predicted_class = classes[pred_idx]
                
                # Vérifier si correct
                is_correct = (predicted_class == our_class)
                status = "✅" if is_correct else "❌"
                
                filename = os.path.basename(img_path)
                print(f"   {filename:<20} → {predicted_class:<8} ({confidence:.1%}) {status}")
                
                if is_correct:
                    total_correct += 1
                    class_correct += 1
                total_tested += 1
                
            except Exception as e:
                print(f"   ❌ Erreur avec {os.path.basename(img_path)}: {e}")
        
        if len(images) > 0:
            class_accuracy = class_correct / len(images)
            print(f"   📊 Accuracy pour {our_class}: {class_accuracy:.1%} ({class_correct}/{len(images)})")
    
    # Résultat final
    print(f"\n" + "="*50)
    print("🎯 RÉSULTAT FINAL")
    print("="*50)
    
    if total_tested > 0:
        accuracy = total_correct / total_tested
        print(f"🎯 Accuracy globale: {accuracy:.1%} ({total_correct}/{total_tested})")
        
        if accuracy > 0.9:
            print("✅ EXCELLENT! Le modèle fonctionne parfaitement sur le dataset officiel!")
            print("💡 Ceci confirme que le problème est bien la généralisation externe")
        elif accuracy > 0.7:
            print("⚠️  CORRECT! Performance acceptable sur le dataset officiel")
        else:
            print("❌ Performance plus faible qu'attendu")
            
        print(f"\n💭 INTERPRÉTATION:")
        print(f"   - Performance élevée ici = Le modèle fonctionne dans son environnement")
        print(f"   - Performance faible sur images externes = Problème de généralisation")
        print(f"   - C'est exactement ce qu'on attendait avec Fruits-360!")
            
    else:
        print("❌ Aucune image testée - vérifiez la structure des dossiers")

if __name__ == "__main__":
    test_avec_votre_structure()