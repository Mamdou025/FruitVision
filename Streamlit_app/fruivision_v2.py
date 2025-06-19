#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FruitVision V2 - Modèle Robuste Amélioré
Incorporation de toutes les leçons apprises pour créer un modèle plus robuste
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
from datetime import datetime
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

class FruitVisionV2:
    def __init__(self):
        """Initialiser le modèle FruitVision V2"""
        self.model = None
        self.classes = ['Pomme', 'Banane', 'Kiwi', 'Citron', 'Peche']
        self.num_classes = len(self.classes)
        self.img_size = (100, 100)
        self.history = None
        
        # Créer les dossiers nécessaires
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        print("🍎 FruitVision V2 - Modèle Robuste Initialisé")
        print(f"✅ Classes: {self.classes}")
        print(f"✅ Taille d'image: {self.img_size}")

    def create_robust_model(self):
        """Créer un modèle CNN robuste avec toutes les bonnes pratiques"""
        
        model = Sequential([
            # Premier bloc convolutionnel avec plus de robustesse
            Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3), 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Deuxième bloc convolutionnel
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Troisième bloc convolutionnel
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Quatrième bloc convolutionnel
            Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.25),
            
            # Global Average Pooling au lieu de Flatten (plus robuste)
            GlobalAveragePooling2D(),
            
            # Couches denses avec régularisation forte
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            # Couche de sortie
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compilation avec un optimiseur plus sophistiqué
        model.compile(
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("✅ Modèle robuste créé avec succès!")
        print(f"📊 Paramètres totaux: {model.count_params():,}")
        
        return model

    def create_advanced_data_generators(self):
        """Créer des générateurs de données avec augmentation agressive"""
        
        # Générateur d'entraînement avec augmentation TRÈS agressive
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,           # Plus de rotation
            width_shift_range=0.3,       # Plus de décalage
            height_shift_range=0.3,
            shear_range=0.3,             # Cisaillement
            zoom_range=0.3,              # Plus de zoom
            horizontal_flip=True,
            vertical_flip=True,          # Ajout du flip vertical
            brightness_range=[0.4, 1.6], # Variation de luminosité
            channel_shift_range=0.2,     # Variation de couleur
            fill_mode='nearest'
        )
        
        # Générateur de validation (seulement normalisation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Générateur de test (seulement normalisation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        print("✅ Générateurs de données avancés créés")
        print("🔄 Augmentation agressive appliquée à l'entraînement")
        
        return train_datagen, val_datagen, test_datagen

    def create_callbacks(self):
        """Créer des callbacks avancés pour l'entraînement"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Sauvegarde du meilleur modèle
            ModelCheckpoint(
                f'models/fruivision_v2_best_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Arrêt précoce pour éviter le surapprentissage
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Réduction du taux d'apprentissage
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard pour le monitoring
            TensorBoard(
                log_dir=f'logs/fruivision_v2_{timestamp}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        print("✅ Callbacks avancés configurés")
        return callbacks

    def prepare_data(self, data_dir="data/fruits-360"):
        """Préparer les données avec mapping correct des classes"""
        
        # Mapping des dossiers du dataset vers nos classes
        class_mappings = {
            'Pomme': ['Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Golden 1', 'Apple 10'],
            'Banane': ['Banana 1', 'Banana 3', 'Banana 4'],
            'Kiwi': ['Kiwi 1'],
            'Citron': ['Lemon 1', 'Lemon Meyer 1'],
            'Peche': ['Peach 1', 'Peach 2']
        }
        
        print("🔍 Vérification des données disponibles...")
        
        train_dir = os.path.join(data_dir, "Training")
        test_dir = os.path.join(data_dir, "Test")
        
        # Vérifier la disponibilité des données
        available_classes = {}
        for our_class, dataset_folders in class_mappings.items():
            for folder in dataset_folders:
                train_path = os.path.join(train_dir, folder)
                test_path = os.path.join(test_dir, folder)
                
                if os.path.exists(train_path) and os.path.exists(test_path):
                    available_classes[our_class] = folder
                    print(f"✅ {our_class}: {folder}")
                    break
            
            if our_class not in available_classes:
                print(f"❌ {our_class}: Aucun dossier trouvé")
        
        if len(available_classes) < len(self.classes):
            print(f"⚠️  Attention: Seulement {len(available_classes)}/{len(self.classes)} classes disponibles")
        
        return available_classes

    def train_model(self, data_dir="data/fruits-360", epochs=50, batch_size=32):
        """Entraîner le modèle avec toutes les améliorations"""
        
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT - FRUIVISION V2")
        print("="*60)
        
        # Préparer les données
        available_classes = self.prepare_data(data_dir)
        
        if len(available_classes) == 0:
            print("❌ Aucune donnée disponible pour l'entraînement")
            return None
        
        # Créer le modèle
        if self.model is None:
            self.create_robust_model()
        
        # Créer les générateurs de données
        train_datagen, val_datagen, test_datagen = self.create_advanced_data_generators()
        
        # Générateurs de données
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, "Training"),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=list(available_classes.values()),  # Utiliser les dossiers disponibles
            shuffle=True
        )
        
        # Créer un split validation à partir des données de test
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(data_dir, "Test"),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=list(available_classes.values()),
            shuffle=True
        )
        
        # Créer les callbacks
        callbacks = self.create_callbacks()
        
        # Afficher les informations d'entraînement
        print(f"📊 Échantillons d'entraînement: {train_generator.samples}")
        print(f"📊 Échantillons de validation: {validation_generator.samples}")
        print(f"📊 Classes mappées: {available_classes}")
        print(f"🎯 Epochs: {epochs}")
        print(f"🎯 Batch size: {batch_size}")
        print(f"🎯 Augmentation de données: Agressive")
        
        # Entraînement
        print("\n🔥 Début de l'entraînement...")
        
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Entraînement terminé!")
        
        # Sauvegarder le modèle final
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f'models/fruivision_v2_final_{timestamp}.h5'
        self.model.save(final_model_path)
        print(f"💾 Modèle final sauvegardé: {final_model_path}")
        
        return self.history

    def plot_training_history(self):
        """Créer des graphiques détaillés de l'entraînement"""
        
        if self.history is None:
            print("❌ Aucun historique d'entraînement disponible")
            return
        
        # Créer une figure avec subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FruitVision V2 - Historique d\'Entraînement', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Accuracy Evolution')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Loss Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-2 Accuracy - Remplacé par une métrique alternative
        if 'val_loss' in self.history.history:
            # Graphique de la différence train/val pour détecter l'overfitting
            train_acc = np.array(self.history.history['accuracy'])
            val_acc = np.array(self.history.history['val_accuracy'])
            acc_diff = train_acc - val_acc
            
            axes[1, 0].plot(acc_diff, label='Train-Val Accuracy Gap', color='red')
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Overfitting Detection (Train-Val Gap)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy Difference')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate Evolution')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/training_history_v2_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Graphiques sauvegardés dans plots/")

    def evaluate_model(self, test_dir="data/fruits-360/Test"):
        """Évaluation complète du modèle"""
        
        if self.model is None:
            print("❌ Aucun modèle chargé")
            return
        
        print("🧪 ÉVALUATION COMPLÈTE DU MODÈLE")
        print("="*40)
        
        # Générateur de test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Évaluation
        print("📊 Évaluation sur données de test...")
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
        
        print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"🎯 Test Loss: {test_loss:.4f}")
        
        # Prédictions pour matrice de confusion
        print("🔍 Génération des prédictions...")
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Rapport de classification
        print("\n📋 Rapport de Classification:")
        class_names = list(test_generator.class_indices.keys())
        print(classification_report(true_classes, predicted_classes, target_names=class_names))
        
        # Matrice de confusion
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Matrice de Confusion - FruitVision V2')
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies Classes')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/confusion_matrix_v2_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        }

    def save_training_config(self):
        """Sauvegarder la configuration d'entraînement"""
        
        config = {
            'model_version': 'FruitVision V2',
            'classes': self.classes,
            'image_size': self.img_size,
            'architecture': 'Advanced CNN with BatchNorm + Dropout + L2',
            'data_augmentation': 'Aggressive',
            'optimizer': 'Adam',
            'regularization': 'L2 + Dropout + BatchNorm',
            'timestamp': datetime.now().isoformat()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = f'models/config_v2_{timestamp}.json'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"⚙️  Configuration sauvegardée: {config_path}")

def main():
    """Fonction principale pour entraîner FruitVision V2"""
    
    print("🍎 FRUIVISION V2 - MODÈLE ROBUSTE")
    print("="*50)
    print("🎯 Objectif: Créer un modèle plus robuste et généralisable")
    print("✨ Améliorations: Architecture avancée + Augmentation agressive")
    print("")
    
    # Créer l'instance
    fv2 = FruitVisionV2()
    
    # Créer le modèle
    fv2.create_robust_model()
    
    # Afficher l'architecture
    print("\n📋 ARCHITECTURE DU MODÈLE:")
    fv2.model.summary()
    
    # Entraîner le modèle
    print("\n🚀 Démarrage de l'entraînement...")
    history = fv2.train_model(epochs=50, batch_size=32)
    
    if history is not None:
        # Créer les graphiques
        fv2.plot_training_history()
        
        # Évaluer le modèle
        fv2.evaluate_model()
        
        # Sauvegarder la configuration
        fv2.save_training_config()
        
        print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("📁 Fichiers générés:")
        print("   - Modèle: models/fruivision_v2_final_*.h5")
        print("   - Graphiques: plots/")
        print("   - Logs: logs/")
        print("   - Config: models/config_v2_*.json")
    else:
        print("❌ Échec de l'entraînement")

if __name__ == "__main__":
    main()