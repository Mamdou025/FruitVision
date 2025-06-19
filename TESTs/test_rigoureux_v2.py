#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Rigoureux FruitVision V2
Évaluation complète pour vérifier si le biais "Pomme" est corrigé
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

class RigorousTestV2:
    def __init__(self, model_path=None):
        """Initialiser le testeur rigoureux"""
        self.classes = ['Pomme', 'Banane', 'Kiwi', 'Citron', 'Peche']
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Charger le modèle V2"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Modèle V2 chargé: {os.path.basename(model_path)}")
            return True
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def find_latest_model(self):
        """Trouver le dernier modèle V2 sauvegardé"""
        model_files = []
        
        # Chercher les modèles V2
        for pattern in ['fruivision_v2_best_*.h5', 'fruivision_v2_final_*.h5']:
            model_files.extend(glob.glob(f'models/{pattern}'))
        
        if not model_files:
            print("❌ Aucun modèle V2 trouvé")
            return None
        
        # Prendre le plus récent
        latest_model = sorted(model_files)[-1]
        print(f"🔍 Modèle le plus récent trouvé: {latest_model}")
        
        return latest_model
    
    def preprocess_image(self, image_path):
        """Préprocesser une image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((100, 100))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
        except Exception as e:
            print(f"❌ Erreur préprocessing {image_path}: {e}")
            return None
    
    def test_dataset_official(self, num_per_class=20):
        """Test rigoureux sur le dataset officiel"""
        
        print("🧪 TEST RIGOUREUX SUR DATASET OFFICIEL")
        print("="*60)
        
        if self.model is None:
            print("❌ Aucun modèle chargé")
            return None
        
        # Mapping des classes
        class_mappings = {
            'Pomme': ['Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
            'Banane': ['Banana 1', 'Banana 3', 'Banana 4'],
            'Kiwi': ['Kiwi 1'],
            'Citron': ['Lemon 1', 'Lemon Meyer 1'],
            'Peche': ['Peach 1', 'Peach 2']
        }
        
        all_predictions = []
        all_true_labels = []
        all_confidences = []
        detailed_results = []
        
        base_dir = "data/fruits-360/test"
        
        print(f"🎯 Test de {num_per_class} images par classe")
        print("-" * 40)
        
        for our_class, dataset_folders in class_mappings.items():
            print(f"\n🍎 Test classe: {our_class}")
            
            # Trouver le dossier disponible
            available_folder = None
            for folder in dataset_folders:
                folder_path = os.path.join(base_dir, folder)
                if os.path.exists(folder_path):
                    available_folder = folder
                    break
            
            if not available_folder:
                print(f"❌ Aucun dossier trouvé pour {our_class}")
                continue
            
            # Récupérer les images
            folder_path = os.path.join(base_dir, available_folder)
            image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
            
            if len(image_files) < num_per_class:
                print(f"⚠️  Seulement {len(image_files)} images disponibles")
                test_images = image_files
            else:
                test_images = random.sample(image_files, num_per_class)
            
            print(f"📁 Dossier: {available_folder}")
            print(f"📊 Images testées: {len(test_images)}")
            
            class_correct = 0
            class_predictions = []
            
            for img_path in test_images:
                image_array = self.preprocess_image(img_path)
                if image_array is None:
                    continue
                
                # Prédiction
                pred = self.model.predict(image_array, verbose=0)
                pred_idx = np.argmax(pred[0])
                confidence = pred[0][pred_idx]
                predicted_class = self.classes[pred_idx]
                
                # Stocker les résultats
                all_predictions.append(predicted_class)
                all_true_labels.append(our_class)
                all_confidences.append(confidence)
                class_predictions.append(predicted_class)
                
                is_correct = predicted_class == our_class
                if is_correct:
                    class_correct += 1
                
                detailed_results.append({
                    'image': os.path.basename(img_path),
                    'true_class': our_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'correct': is_correct
                })
            
            # Statistiques par classe
            if len(test_images) > 0:
                class_accuracy = class_correct / len(test_images)
                class_predictions_count = Counter(class_predictions)
                
                print(f"✅ Accuracy: {class_accuracy:.1%} ({class_correct}/{len(test_images)})")
                print(f"📈 Prédictions: {dict(class_predictions_count)}")
                
                # Vérifier le biais "Pomme"
                pomme_predictions = class_predictions_count.get('Pomme', 0)
                pomme_rate = pomme_predictions / len(test_images)
                
                if our_class != 'Pomme' and pomme_rate > 0.5:
                    print(f"⚠️  BIAIS DÉTECTÉ: {pomme_rate:.1%} prédit comme Pomme")
                elif our_class != 'Pomme' and pomme_rate < 0.2:
                    print(f"✅ Biais réduit: seulement {pomme_rate:.1%} prédit comme Pomme")
        
        return self.analyze_overall_results(all_predictions, all_true_labels, all_confidences, detailed_results)
    
    def analyze_overall_results(self, predictions, true_labels, confidences, detailed_results):
        """Analyser les résultats globaux"""
        
        print("\n" + "="*60)
        print("📊 ANALYSE GLOBALE DES RÉSULTATS")
        print("="*60)
        
        if not predictions:
            print("❌ Aucune prédiction à analyser")
            return None
        
        # Accuracy globale
        overall_accuracy = accuracy_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        print(f"🎯 Accuracy globale: {overall_accuracy:.1%}")
        print(f"🎯 Confiance moyenne: {avg_confidence:.1%}")
        print(f"🎯 Total d'images: {len(predictions)}")
        
        # Analyse du biais "Pomme"
        prediction_counts = Counter(predictions)
        pomme_ratio = prediction_counts.get('Pomme', 0) / len(predictions)
        
        print(f"\n🔍 ANALYSE DU BIAIS:")
        print(f"📈 Distribution des prédictions:")
        for class_name in self.classes:
            count = prediction_counts.get(class_name, 0)
            percentage = count / len(predictions) * 100
            print(f"   {class_name:<8}: {count:3d} ({percentage:5.1f}%)")
        
        # Diagnostic du biais
        print(f"\n🎯 DIAGNOSTIC:")
        if pomme_ratio > 0.6:
            print(f"❌ BIAIS POMME SÉVÈRE: {pomme_ratio:.1%} des prédictions")
            bias_status = "SEVERE"
        elif pomme_ratio > 0.4:
            print(f"⚠️  BIAIS POMME MODÉRÉ: {pomme_ratio:.1%} des prédictions")
            bias_status = "MODERATE"
        elif pomme_ratio > 0.25:
            print(f"✅ BIAIS POMME LÉGER: {pomme_ratio:.1%} des prédictions (acceptable)")
            bias_status = "LIGHT"
        else:
            print(f"✅ AUCUN BIAIS POMME: {pomme_ratio:.1%} des prédictions")
            bias_status = "NONE"
        
        # Matrice de confusion
        self.plot_confusion_matrix(true_labels, predictions)
        
        # Rapport détaillé par classe
        print(f"\n📋 RAPPORT DÉTAILLÉ PAR CLASSE:")
        report = classification_report(true_labels, predictions, 
                                     target_names=self.classes, 
                                     output_dict=True)
        
        for class_name in self.classes:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                
                print(f"   {class_name:<8}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={support})")
        
        # Comparaison avec V1
        print(f"\n📈 COMPARAISON AVEC V1:")
        print(f"   V1: 40% accuracy, biais Pomme sévère")
        print(f"   V2: {overall_accuracy:.1%} accuracy, biais {bias_status}")
        
        if overall_accuracy > 0.8 and bias_status in ["LIGHT", "NONE"]:
            print(f"🎉 AMÉLIORATION SIGNIFICATIVE!")
        elif overall_accuracy > 0.6:
            print(f"✅ Amélioration modérée")
        else:
            print(f"❌ Amélioration insuffisante")
        
        return {
            'accuracy': overall_accuracy,
            'avg_confidence': avg_confidence,
            'bias_status': bias_status,
            'pomme_ratio': pomme_ratio,
            'predictions_distribution': prediction_counts,
            'detailed_results': detailed_results
        }
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Créer la matrice de confusion"""
        
        cm = confusion_matrix(true_labels, predictions, labels=self.classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Matrice de Confusion - FruitVision V2', fontsize=16)
        plt.xlabel('Prédictions', fontsize=12)
        plt.ylabel('Vraies Classes', fontsize=12)
        
        # Sauvegarder
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/confusion_matrix_test_v2_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Matrice de confusion sauvegardée dans plots/")
    
    def test_synthetic_images(self):
        """Test sur images synthétiques pour vérifier la robustesse"""
        
        print("\n🎨 TEST SUR IMAGES SYNTHÉTIQUES")
        print("="*40)
        
        if self.model is None:
            print("❌ Aucun modèle chargé")
            return
        
        # Créer des images colorées simples
        synthetic_tests = {
            'Rouge (Pomme)': (255, 0, 0),
            'Jaune (Banane)': (255, 255, 0),
            'Vert (Kiwi)': (0, 255, 0),
            'Jaune (Citron)': (255, 255, 100),
            'Orange (Pêche)': (255, 165, 0)
        }
        
        print("🔍 Test de généralisation sur couleurs pures:")
        
        for description, color in synthetic_tests.items():
            # Créer une image colorée
            image = Image.new('RGB', (100, 100), color)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Prédiction
            pred = self.model.predict(image_array, verbose=0)
            pred_idx = np.argmax(pred[0])
            confidence = pred[0][pred_idx]
            predicted_class = self.classes[pred_idx]
            
            print(f"   {description:<18} → {predicted_class:<8} ({confidence:.1%})")
        
        print("\n💡 Note: Les images synthétiques testent la robustesse du modèle")
        print("   Un bon modèle ne devrait pas avoir de confiance excessive sur ces images")

def main():
    """Fonction principale de test rigoureux"""
    
    print("🔬 FRUIVISION V2 - TEST RIGOUREUX")
    print("="*70)
    print("🎯 Objectif: Vérifier si le biais 'Pomme' a été corrigé")
    print("📋 Tests: Dataset officiel + Images synthétiques")
    print("")
    
    # Créer le testeur
    tester = RigorousTestV2()
    
    # Trouver le modèle le plus récent
    latest_model = tester.find_latest_model()
    
    if latest_model is None:
        print("❌ Aucun modèle V2 trouvé pour les tests")
        return
    
    # Charger le modèle
    if not tester.load_model(latest_model):
        return
    
    # Test principal sur dataset officiel
    results = tester.test_dataset_official(num_per_class=15)
    
    # Test sur images synthétiques
    tester.test_synthetic_images()
    
    # Résumé final
    if results:
        print("\n" + "="*70)
        print("🎯 RÉSUMÉ FINAL")
        print("="*70)
        print(f"✅ Accuracy: {results['accuracy']:.1%}")
        print(f"✅ Confiance moyenne: {results['avg_confidence']:.1%}")
        print(f"✅ Statut biais: {results['bias_status']}")
        
        if results['accuracy'] > 0.8 and results['bias_status'] in ["LIGHT", "NONE"]:
            print("🎉 SUCCÈS: Le modèle V2 est significativement amélioré!")
        else:
            print("⚠️  Le modèle V2 nécessite encore des améliorations")

if __name__ == "__main__":
    main()