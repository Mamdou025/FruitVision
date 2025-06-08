#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du modèle FruitVision sur le dataset officiel Fruits-360
Objectif: Vérifier les performances sur données du même environnement
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_classes():
    """Charger le modèle et les classes"""
    try:
        model = tf.keras.models.load_model('models/fruivision_split_manuel.h5')
        classes = ['Pomme', 'Banane', 'Kiwi', 'Citron', 'Peche']
        print("✅ Modèle chargé avec succès")
        print(f"✅ Classes: {classes}")
        return model, classes
    except:
        print("❌ Erreur: Impossible de charger le modèle")
        return None, None

def preprocess_image(image_path, target_size=(100, 100)):
    """Préprocesser une image comme pendant l'entraînement"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"❌ Erreur avec {image_path}: {e}")
        return None

def test_on_official_dataset(model, classes, test_dir="Test"):
    """Tester sur le dataset officiel Fruits-360"""
    
    print("\n" + "="*60)
    print("🧪 TEST SUR DATASET OFFICIEL FRUITS-360")
    print("="*60)
    
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    results_by_class = {}
    
    # Mapper les noms de classes du dataset vers nos classes
    class_mapping = {
        'Apple Red 1': 'Pomme',
        'Apple Red 2': 'Pomme', 
        'Apple Red 3': 'Pomme',
        'Apple Red Delicious': 'Pomme',
        'Banana': 'Banane',
        'Banana Red': 'Banane',
        'Kiwi': 'Kiwi',
        'Lemon': 'Citron',
        'Lemon Meyer': 'Citron',
        'Peach': 'Peche',
        'Peach 2': 'Peche'
    }
    
    if not os.path.exists(test_dir):
        print(f"❌ Dossier {test_dir} introuvable")
        return
    
    # Parcourir chaque classe dans le dossier Test
    for class_folder in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_folder)
        
        if not os.path.isdir(class_path):
            continue
            
        # Vérifier si cette classe nous intéresse
        if class_folder not in class_mapping:
            continue
            
        true_class = class_mapping[class_folder]
        
        print(f"\n🍎 Analyse de {class_folder} -> {true_class}:")
        
        class_predictions = []
        class_confidences = []
        correct_predictions = 0
        total_images = 0
        
        # Tester toutes les images de cette classe
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))
        
        for image_path in image_files[:20]:  # Limiter à 20 images par classe
            image_array = preprocess_image(image_path)
            
            if image_array is None:
                continue
                
            # Prédiction
            prediction = model.predict(image_array, verbose=0)
            pred_idx = np.argmax(prediction[0])
            confidence = prediction[0][pred_idx]
            predicted_class = classes[pred_idx]
            
            # Stocker les résultats
            all_predictions.append(predicted_class)
            all_true_labels.append(true_class)
            all_confidences.append(confidence)
            
            class_predictions.append(predicted_class)
            class_confidences.append(confidence)
            
            if predicted_class == true_class:
                correct_predictions += 1
                
            total_images += 1
            
            # Afficher quelques exemples
            if total_images <= 5:
                status = "✅" if predicted_class == true_class else "❌"
                filename = os.path.basename(image_path)
                print(f"   {filename:<15} → {predicted_class:<8} ({confidence:.1%}) {status}")
        
        # Statistiques par classe
        if total_images > 0:
            class_accuracy = correct_predictions / total_images
            avg_confidence = np.mean(class_confidences)
            
            results_by_class[true_class] = {
                'accuracy': class_accuracy,
                'confidence': avg_confidence,
                'total': total_images,
                'correct': correct_predictions
            }
            
            print(f"   📊 Accuracy: {class_accuracy:.1%} ({correct_predictions}/{total_images})")
            print(f"   📊 Confiance moyenne: {avg_confidence:.1%}")
    
    # Statistiques globales
    print("\n" + "="*60)
    print("📊 RÉSULTATS GLOBAUX")
    print("="*60)
    
    if all_predictions:
        overall_accuracy = accuracy_score(all_true_labels, all_predictions)
        avg_confidence = np.mean(all_confidences)
        
        print(f"🎯 Accuracy globale: {overall_accuracy:.1%}")
        print(f"🎯 Confiance moyenne: {avg_confidence:.1%}")
        print(f"🎯 Total d'images testées: {len(all_predictions)}")
        
        # Détail par classe
        print(f"\n📋 DÉTAIL PAR CLASSE:")
        for class_name, stats in results_by_class.items():
            print(f"   {class_name:<8}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
        
        # Matrice de confusion
        print(f"\n🔍 MATRICE DE CONFUSION:")
        unique_classes = list(set(all_true_labels))
        cm = confusion_matrix(all_true_labels, all_predictions, labels=unique_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_classes, yticklabels=unique_classes)
        plt.title('Matrice de Confusion - Dataset Officiel')
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies Classes')
        plt.tight_layout()
        plt.show()
        
        # Rapport de classification
        print(f"\n📝 RAPPORT DÉTAILLÉ:")
        print(classification_report(all_true_labels, all_predictions))
        
        return overall_accuracy
    else:
        print("❌ Aucune image trouvée à tester")
        return 0

def main():
    """Fonction principale"""
    print("🔍 TEST SUR DATASET OFFICIEL FRUITS-360")
    print("Objectif: Vérifier les performances sur le même environnement")
    print("="*70)
    
    # Charger le modèle
    model, classes = load_model_and_classes()
    if model is None:
        return
    
    # Tester sur le dataset officiel
    accuracy = test_on_official_dataset(model, classes)
    
    print(f"\n🎯 CONCLUSION:")
    if accuracy > 0.95:
        print(f"✅ Excellente performance: {accuracy:.1%}")
        print("✅ Le modèle fonctionne parfaitement sur le dataset officiel")
        print("💡 Ceci confirme que le problème vient de la généralisation")
        print("💡 Le dataset Fruits-360 est trop artificiel pour le monde réel")
    elif accuracy > 0.80:
        print(f"⚠️  Performance correcte: {accuracy:.1%}")
        print("⚠️  Il pourrait y avoir des problèmes de préprocessing")
    else:
        print(f"❌ Performance faible: {accuracy:.1%}")
        print("❌ Il y a un problème avec le modèle ou les données")

if __name__ == "__main__":
    main()