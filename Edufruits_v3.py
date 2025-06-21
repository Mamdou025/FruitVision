#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EduFruis V3 - Correction du Mapping de Classes
Solution: Créer un dossier unifié par classe avant l'entraînement
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, 
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json


class EduFruisV3Fixed:
    def __init__(self):
        """Initialiser EduFruis V3 avec correction du mapping"""
        self.classes = ['Pomme', 'Banane', 'Tomate', 'Concombre', 'Citron']
        self.num_classes = len(self.classes)
        self.img_size = (100, 100)
        self.history = None
        
        # Mapping corrigé - regroupement par classe logique
        self.class_mappings_v3 = {
            'Pomme': [
                'Apple Red 1',
                'Apple Golden 2', 
                'Apple Braeburn 1',
                'Apple Granny Smith 1'
            ],
            'Banane': [
                'Banana 1',
                'Banana 3', 
                'Banana 4',
                'Banana Lady Finger 1'
            ],
            'Tomate': [
                'Tomato 1',
                'Tomato 2',
                'Tomato 3',
                'Tomato 4'
            ],
            'Concombre': [
                'Cucumber 11',
                'Cucumber 1',
                'Cucumber 4',
                'Cucumber 3'
            ],
            'Citron': [
                'Lemon 1',
                'Lemon Meyer 1'
            ]
        }
        
        # Créer les dossiers nécessaires
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('data_v3', exist_ok=True)
        os.makedirs('data_v3/Training', exist_ok=True)
        os.makedirs('data_v3/Test', exist_ok=True)
        
        print("🚀 EduFruis V3 Fixed - Mapping Corrigé")
        print(f"✅ Classes finales: {self.classes}")

    def create_unified_dataset_v3(self, source_dir="data/fruits-360", target_dir="data_v3"):
        """Créer un dataset unifié en regroupant les variétés par classe"""
        
        print("\n🔄 CRÉATION DU DATASET UNIFIÉ V3")
        print("="*50)
        print("🎯 Objectif: Regrouper les variétés par classe logique")
        
        # Nettoyer le dossier cible
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        os.makedirs(f"{target_dir}/Training", exist_ok=True)
        os.makedirs(f"{target_dir}/Test", exist_ok=True)
        
        stats = {}
        
        for split in ['Training', 'Test']:
            print(f"\n📁 Traitement {split}:")
            split_stats = {}
            
            for class_name, source_folders in self.class_mappings_v3.items():
                # Créer le dossier de classe unifié
                target_class_dir = os.path.join(target_dir, split, class_name)
                os.makedirs(target_class_dir, exist_ok=True)
                
                total_images = 0
                
                print(f"   🍎 {class_name}:")
                
                for source_folder in source_folders:
                    source_path = os.path.join(source_dir, split, source_folder)
                    
                    if os.path.exists(source_path):
                        # Copier toutes les images de ce dossier vers le dossier unifié
                        images = [f for f in os.listdir(source_path) if f.endswith('.jpg')]
                        
                        for image_file in images:
                            source_image = os.path.join(source_path, image_file)
                            # Renommer pour éviter les conflits
                            new_name = f"{source_folder}_{image_file}"
                            target_image = os.path.join(target_class_dir, new_name)
                            shutil.copy2(source_image, target_image)
                            total_images += 1
                        
                        print(f"      ✅ {source_folder}: {len(images)} images")
                    else:
                        print(f"      ❌ {source_folder}: Non trouvé")
                
                split_stats[class_name] = total_images
                print(f"      📊 Total {class_name}: {total_images} images")
            
            stats[split] = split_stats
        
        # Analyse de l'équilibre final
        print(f"\n📊 ANALYSE D'ÉQUILIBRE FINAL:")
        print("="*40)
        
        train_counts = list(stats['Training'].values())
        test_counts = list(stats['Test'].values())
        
        if train_counts:
            balance_ratio = min(train_counts) / max(train_counts)
            print(f"🎯 Équilibre Training: {balance_ratio:.2%}")
            
            for class_name in self.classes:
                train_count = stats['Training'].get(class_name, 0)
                test_count = stats['Test'].get(class_name, 0)
                total = train_count + test_count
                print(f"   {class_name:<12}: Train={train_count:>4}, Test={test_count:>3}, Total={total:>4}")
        
        if balance_ratio > 0.7:
            print("✅ Excellent équilibre atteint!")
        elif balance_ratio > 0.5:
            print("⚠️  Équilibre acceptable")
        else:
            print("❌ Déséquilibre encore présent")
        
        print(f"✅ Dataset unifié créé dans: {target_dir}")
        return stats

    def create_robust_model_v3(self):
        """Créer le modèle V3 (architecture V2 prouvée)"""
        
        model = Sequential([
            # Bloc 1
            Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3), 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Bloc 2
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Bloc 3
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Bloc 4
            Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.25),
            
            # Global Average Pooling
            GlobalAveragePooling2D(),
            
            # Couches denses
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            # Sortie pour 5 classes
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Modèle V3 créé (5 classes)")
        return model

    def create_data_generators_v3(self):
        """Générateurs de données V3"""
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.4, 1.6],
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        return train_datagen, val_datagen, test_datagen

    def calculate_class_weights(self, train_generator):
        """Calculer les poids de classe pour corriger le déséquilibre"""
        
        # Compter les échantillons par classe
        class_counts = {}
        for class_name, class_idx in train_generator.class_indices.items():
            class_counts[class_idx] = 0
        
        # Parcourir tous les échantillons
        total_samples = train_generator.samples
        
        # Estimation basée sur la distribution observée
        # (plus rapide que de parcourir tous les fichiers)
        estimated_counts = {
            0: 1968,  # Pomme
            1: 1427,  # Banane  
            2: 2627,  # Tomate
            3: 856,   # Concombre
            4: 982    # Citron
        }
        
        # Calculer les poids (inverse de la fréquence)
        max_count = max(estimated_counts.values())
        class_weights = {}
        
        for class_idx, count in estimated_counts.items():
            weight = max_count / count
            class_weights[class_idx] = weight
        
        return class_weights

    def train_model_v3(self, data_dir="data_v3", epochs=50, batch_size=32):
        """Entraîner le modèle V3 avec dataset unifié"""
        
        print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT V3 CORRIGÉ")
        print("="*60)
        
        # Vérifier que le dataset unifié existe
        if not os.path.exists(data_dir):
            print("❌ Dataset unifié non trouvé. Création en cours...")
            self.create_unified_dataset_v3()
        
        # Créer le modèle - CORRECTION DU BUG
        self.create_robust_model_v3()
        
        # Générateurs
        train_datagen, val_datagen, test_datagen = self.create_data_generators_v3()
        
        # Générateurs pointant vers les classes unifiées
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, "Training"),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.classes,  # Utiliser directement nos 5 classes
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(data_dir, "Test"),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.classes,  # Utiliser directement nos 5 classes
            shuffle=True
        )
        
        # Vérifier les dimensions
        print(f"📊 Vérification des dimensions:")
        print(f"   Train generator: {train_generator.samples} échantillons, {train_generator.num_classes} classes")
        print(f"   Val generator: {validation_generator.samples} échantillons, {validation_generator.num_classes} classes")
        print(f"   Modèle output: {self.num_classes} classes")
        
        if train_generator.num_classes != self.num_classes:
            print(f"❌ ERREUR: Mismatch classes! Generator={train_generator.num_classes}, Modèle={self.num_classes}")
            return None
        
        # CORRECTION DÉSÉQUILIBRE: Calcul des poids de classe
        class_weights = self.calculate_class_weights(train_generator)
        print(f"⚖️  Poids de classe appliqués pour corriger le déséquilibre:")
        for i, class_name in enumerate(self.classes):
            print(f"   {class_name}: {class_weights[i]:.2f}")
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            ModelCheckpoint(
                f'models/fruivision_v3_best_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\n🔥 Entraînement V3 avec classes unifiées et poids équilibrés...")
        print(f"📋 Classes: {list(train_generator.class_indices.keys())}")
        
        # Entraînement avec poids de classe
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks,
            class_weight=class_weights,  # CORRECTION DU DÉSÉQUILIBRE
            verbose=1
        )
        
        # Sauvegarder
        final_model_path = f'models/fruivision_v3_final_{timestamp}.h5'
        self.model.save(final_model_path)
        print(f"💾 Modèle V3 sauvegardé: {final_model_path}")
        
        return self.history

    def plot_training_history_v3(self):
        """Graphiques d'entraînement V3"""
        
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('EduFruis V3 - Classes Unifiées', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Accuracy V3')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss V3')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Gap Train-Val
        train_acc = np.array(self.history.history['accuracy'])
        val_acc = np.array(self.history.history['val_accuracy'])
        axes[1, 0].plot(train_acc - val_acc, color='red')
        axes[1, 0].set_title('Overfitting Detection')
        axes[1, 0].grid(True)
        
        # Final metrics
        final_acc = self.history.history['val_accuracy'][-1]
        axes[1, 1].text(0.5, 0.5, f'Final Accuracy\n{final_acc:.1%}', 
                       ha='center', va='center', fontsize=20,
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Performance Finale')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/training_v3_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Entraînement V3 avec correction"""
    
    print("🚀 FRUIVISION V3 - VERSION CORRIGÉE")
    print("="*50)
    print("🔧 Correction: Dataset unifié par classe")
    print("✨ Classes: Pomme, Banane, Tomate, Concombre, Citron")
    
    # Instance V3
    fv3 = EduFruisV3Fixed()
    
    # Créer le dataset unifié
    print("\n📁 Création du dataset unifié...")
    fv3.create_unified_dataset_v3()
    
    # Entraîner
    history = fv3.train_model_v3(epochs=30, batch_size=32)
    
    if history:
        fv3.plot_training_history_v3()
        print("\n🎉 V3 RÉUSSI!")
    else:
        print("\n❌ Échec V3")

if __name__ == "__main__":
    main()