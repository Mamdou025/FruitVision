"""
EduFruis - Entraînement Complet
==================================

Script pour entraîner le modèle final de production avec toutes les données.
Utilise l'augmentation de données et l'entraînement robuste.

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

# Ajouter les dossiers au path
sys.path.append('src')
sys.path.append('config')

# Importer nos modules
from model_architecture import creer_modele_fruivision
from data_preprocessing import PreprocesseurDonnees
from training_utils import GestionnaireEntrainement
from model_config import CONFIG_DONNEES, CONFIG_ENTRAINEMENT

def configuration_entrainement_complet():
    """Configuration pour l'entraînement complet."""
    
    config = {
        # Données - TOUTES les images
        'max_images_par_classe': None,          # Pas de limite (toutes les images)
        'utiliser_augmentation': True,          # Augmentation de données active
        'taille_validation': 0.2,              # 20% pour validation
        'taille_test': 0.1,                    # 10% pour test
        
        # Entraînement - Configuration robuste
        'epochs': 50,                           # 50 époques pour modèle robuste
        'batch_size': 32,                       # Batch optimal pour 8GB RAM
        'learning_rate': 0.001,                 # Taux d'apprentissage initial
        
        # Callbacks pour optimisation
        'early_stopping_patience': 15,         # Plus patient pour gros dataset
        'reduce_lr_patience': 7,               # Réduire LR si stagnation
        'save_best_only': True,                # Sauver seulement le meilleur
        
        # Verbosité et monitoring
        'verbose': 1,                          # Affichage détaillé
        'validation_freq': 1,                  # Validation à chaque époque
    }
    
    return config

def creer_callbacks_avances(config):
    """Créer des callbacks avancés pour l'entraînement."""
    
    try:
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
            CSVLogger, TensorBoard
        )
        
        callbacks = []
        
        # 1. Arrêt précoce intelligent
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=config['early_stopping_patience'],
            min_delta=0.001,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # 2. Réduction du taux d'apprentissage
        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,                        # Diviser par 2
            patience=config['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # 3. Sauvegarde du meilleur modèle
        os.makedirs('models', exist_ok=True)
        checkpoint = ModelCheckpoint(
            filepath='models/fruivision_best_model.h5',
            monitor='val_accuracy',
            save_best_only=config['save_best_only'],
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # 4. Log CSV pour analyse
        os.makedirs('results', exist_ok=True)
        csv_logger = CSVLogger('results/training_history.csv')
        callbacks.append(csv_logger)
        
        # 5. TensorBoard (optionnel)
        try:
            tensorboard = TensorBoard(
                log_dir='results/tensorboard_logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
            callbacks.append(tensorboard)
            print("✅ TensorBoard activé - dossier: results/tensorboard_logs")
        except:
            print("⚠️ TensorBoard non disponible")
        
        print(f"✅ {len(callbacks)} callbacks configurés")
        return callbacks
        
    except ImportError as e:
        print(f"❌ Erreur lors de l'importation des callbacks: {e}")
        return []

def estimer_temps_entrainement(nb_images, epochs, batch_size):
    """Estimer le temps d'entraînement basé sur les paramètres."""
    
    # Estimation basée sur test rapide: ~2s par batch de 16
    temps_par_batch = 2.0  # secondes
    batches_par_epoque = nb_images // batch_size
    temps_par_epoque = batches_par_epoque * temps_par_batch
    temps_total = temps_par_epoque * epochs
    
    return {
        'batches_par_epoque': batches_par_epoque,
        'temps_par_epoque_min': temps_par_epoque / 60,
        'temps_total_min': temps_total / 60,
        'temps_total_h': temps_total / 3600
    }

def entrainement_complet():
    """Fonction principale d'entraînement complet."""
    
    print("🏋️ ENTRAÎNEMENT COMPLET FRUIVISION")
    print("=" * 70)
    
    debut_total = datetime.now()
    
    # Configuration
    config = configuration_entrainement_complet()
    print("⚙️ Configuration d'entraînement:")
    for cle, valeur in config.items():
        print(f"   {cle}: {valeur}")
    
    # ========================================
    # ÉTAPE 1: PRÉPARATION DES DONNÉES
    # ========================================
    
    print(f"\n📂 ÉTAPE 1: Chargement des Données Complètes")
    print("-" * 50)
    
    # Classes réelles du dataset
    classes_fruits_reelles = {
        'Pomme': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Braeburn 1',
                 'Apple Granny Smith 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Red 1'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    # Créer le préprocesseur
    preprocesseur = PreprocesseurDonnees(classes_fruits=classes_fruits_reelles)
    print(f"✅ Préprocesseur créé pour {len(preprocesseur.noms_classes)} classes")
    
    # Charger TOUTES les données avec augmentation
    print(f"\n📥 Chargement de toutes les données (peut prendre 5-10 minutes)...")
    debut_chargement = datetime.now()
    
    X, y = preprocesseur.charger_donnees_dossier(
        "data/fruits-360/Training",
        utiliser_augmentation=config['utiliser_augmentation'],
        max_images_par_classe=config['max_images_par_classe']  # None = toutes
    )
    
    fin_chargement = datetime.now()
    temps_chargement = (fin_chargement - debut_chargement).seconds
    
    print(f"✅ Données chargées en {temps_chargement // 60}m {temps_chargement % 60}s")
    print(f"📊 Dataset final: {X.shape[0]:,} images total")
    print(f"📐 Forme des données: {X.shape}")
    
    # Analyser la distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"🎯 Distribution par classe:")
    for i, (classe, count) in enumerate(zip(preprocesseur.noms_classes, counts)):
        print(f"   {classe}: {count:,} images")
    
    # Division des données
    print(f"\n🔄 Division des données...")
    ensembles = preprocesseur.creer_ensembles_donnees(
        X, y, 
        taille_validation=config['taille_validation'],
        taille_test=config['taille_test']
    )
    
    # Estimation du temps
    estimation = estimer_temps_entrainement(
        ensembles['X_train'].shape[0], 
        config['epochs'], 
        config['batch_size']
    )
    
    print(f"\n⏱️ ESTIMATION DU TEMPS D'ENTRAÎNEMENT:")
    print(f"   📊 Images d'entraînement: {ensembles['X_train'].shape[0]:,}")
    print(f"   📦 Batches par époque: {estimation['batches_par_epoque']}")
    print(f"   ⏱️ Temps par époque: ~{estimation['temps_par_epoque_min']:.1f} minutes")
    print(f"   🕐 Temps total estimé: ~{estimation['temps_total_h']:.1f} heures")
    
    # Demander confirmation
    print(f"\n❓ CONTINUER L'ENTRAÎNEMENT?")
    print(f"   Cela va prendre environ {estimation['temps_total_h']:.1f} heures sur votre PC.")
    confirmation = input("   Continuer? (o/n): ").lower().strip()
    
    if confirmation not in ['o', 'oui', 'y', 'yes']:
        print("❌ Entraînement annulé par l'utilisateur.")
        return False
    
    # ========================================
    # ÉTAPE 2: CRÉATION DU MODÈLE
    # ========================================
    
    print(f"\n🧠 ÉTAPE 2: Création du Modèle de Production")
    print("-" * 50)
    
    modele = creer_modele_fruivision()
    print("✅ Modèle CNN créé et compilé")
    
    # Afficher le résumé du modèle
    print(f"\n📋 Résumé du modèle:")
    modele.summary()
    
    # ========================================
    # ÉTAPE 3: CONFIGURATION DES CALLBACKS
    # ========================================
    
    print(f"\n⚙️ ÉTAPE 3: Configuration des Callbacks")
    print("-" * 50)
    
    callbacks = creer_callbacks_avances(config)
    
    # ========================================
    # ÉTAPE 4: ENTRAÎNEMENT COMPLET
    # ========================================
    
    print(f"\n🚀 ÉTAPE 4: Entraînement Complet")
    print("-" * 50)
    print(f"🏁 Début: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📊 Configuration finale:")
    print(f"   Époques: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Images d'entraînement: {ensembles['X_train'].shape[0]:,}")
    print(f"   Images de validation: {ensembles['X_val'].shape[0]:,}")
    
    debut_entrainement = datetime.now()
    
    try:
        # ENTRAÎNEMENT PRINCIPAL
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
        print(f"⏱️ Temps total: {temps_entrainement}")
        print(f"🏁 Fin: {fin_entrainement.strftime('%H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n⛔ Entraînement interrompu par l'utilisateur")
        fin_entrainement = datetime.now()
        temps_entrainement = fin_entrainement - debut_entrainement
        print(f"⏱️ Temps partiel: {temps_entrainement}")
        
        # Sauvegarder le modèle actuel
        modele.save('models/fruivision_interrupted.h5')
        print(f"💾 Modèle partiel sauvegardé: models/fruivision_interrupted.h5")
        return False
        
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {e}")
        return False
    
    # ========================================
    # ÉTAPE 5: ÉVALUATION FINALE
    # ========================================
    
    print(f"\n📊 ÉTAPE 5: Évaluation du Modèle Final")
    print("-" * 50)
    
    # Charger le meilleur modèle
    try:
        if os.path.exists('models/fruivision_best_model.h5'):
            import tensorflow as tf
            meilleur_modele = tf.keras.models.load_model('models/fruivision_best_model.h5')
            print("✅ Meilleur modèle chargé pour évaluation")
        else:
            meilleur_modele = modele
            print("⚠️ Utilisation du modèle final (pas de sauvegarde intermédiaire)")
    except:
        meilleur_modele = modele
        print("⚠️ Erreur de chargement - utilisation du modèle actuel")
    
    # Évaluation sur l'ensemble de test
    print(f"🧪 Évaluation sur l'ensemble de test...")
    
    resultats_test = meilleur_modele.evaluate(
        ensembles['X_test'], ensembles['y_test'], 
        verbose=1
    )
    
    accuracy_finale = resultats_test[1]
    loss_finale = resultats_test[0]
    
    # Prédictions pour analyse détaillée
    predictions = meilleur_modele.predict(ensembles['X_test'], verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(ensembles['y_test'], axis=1)
    
    # Calculer métriques par classe
    from sklearn.metrics import classification_report, confusion_matrix
    
    rapport = classification_report(
        y_true, y_pred, 
        target_names=preprocesseur.noms_classes,
        output_dict=True
    )
    
    # ========================================
    # ÉTAPE 6: SAUVEGARDE FINALE
    # ========================================
    
    print(f"\n💾 ÉTAPE 6: Sauvegarde des Résultats")
    print("-" * 50)
    
    # Sauvegarder le modèle final
    modele.save('models/fruivision_final.h5')
    print(f"✅ Modèle final sauvegardé: models/fruivision_final.h5")
    
    # Sauvegarder les résultats complets
    resultats_complets = {
        'entrainement': {
            'debut': debut_total.isoformat(),
            'fin': fin_entrainement.isoformat(),
            'duree_secondes': temps_entrainement.total_seconds(),
            'duree_humaine': str(temps_entrainement),
            'epochs_completees': len(historique.history['accuracy']),
            'config': config
        },
        'donnees': {
            'nb_images_total': X.shape[0],
            'nb_train': ensembles['X_train'].shape[0],
            'nb_validation': ensembles['X_val'].shape[0],
            'nb_test': ensembles['X_test'].shape[0],
            'distribution_classes': {
                preprocesseur.noms_classes[i]: int(count) 
                for i, count in enumerate(counts)
            }
        },
        'performance': {
            'accuracy_finale': float(accuracy_finale),
            'loss_finale': float(loss_finale),
            'meilleure_val_accuracy': float(max(historique.history['val_accuracy'])),
            'meilleure_epoque': int(np.argmax(historique.history['val_accuracy']) + 1),
            'rapport_classification': rapport
        },
        'historique': {
            'accuracy': [float(x) for x in historique.history['accuracy']],
            'val_accuracy': [float(x) for x in historique.history['val_accuracy']],
            'loss': [float(x) for x in historique.history['loss']],
            'val_loss': [float(x) for x in historique.history['val_loss']]
        }
    }
    
    with open('results/entrainement_complet_resultats.json', 'w') as f:
        json.dump(resultats_complets, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Résultats détaillés sauvegardés: results/entrainement_complet_resultats.json")
    
    # Créer des visualisations
    creer_visualisations_finales(historique, y_true, y_pred, preprocesseur.noms_classes)
    
    # ========================================
    # ÉTAPE 7: RÉSUMÉ FINAL
    # ========================================
    
    print(f"\n🎉 RÉSUMÉ FINAL")
    print("=" * 70)
    
    print(f"⏱️ TEMPS D'ENTRAÎNEMENT:")
    print(f"   Durée totale: {temps_entrainement}")
    print(f"   Début: {debut_total.strftime('%H:%M:%S')}")
    print(f"   Fin: {fin_entrainement.strftime('%H:%M:%S')}")
    
    print(f"\n📊 PERFORMANCE FINALE:")
    print(f"   🎯 Précision sur test: {accuracy_finale:.4f} ({accuracy_finale*100:.2f}%)")
    print(f"   📉 Perte finale: {loss_finale:.4f}")
    print(f"   🏆 Meilleure val_accuracy: {resultats_complets['performance']['meilleure_val_accuracy']:.4f}")
    print(f"   📈 Atteinte à l'époque: {resultats_complets['performance']['meilleure_epoque']}")
    
    print(f"\n📂 FICHIERS GÉNÉRÉS:")
    print(f"   🤖 models/fruivision_final.h5 ({os.path.getsize('models/fruivision_final.h5')/1024/1024:.1f} MB)")
    print(f"   📊 results/entrainement_complet_resultats.json")
    print(f"   📈 results/training_history.csv")
    print(f"   🖼️ results/courbes_apprentissage.png")
    print(f"   📊 results/matrice_confusion.png")
    
    # Interprétation des résultats
    print(f"\n🤔 INTERPRÉTATION:")
    if accuracy_finale > 0.95:
        print(f"   🎉 EXCELLENT! >95% - Modèle prêt pour production")
    elif accuracy_finale > 0.90:
        print(f"   ✅ TRÈS BON! >90% - Performance solide")
    elif accuracy_finale > 0.85:
        print(f"   👍 BON! >85% - Objectif atteint")
    else:
        print(f"   ⚠️ MOYEN. <85% - Peut nécessiter des améliorations")
    
    print(f"\n🚀 PROCHAINES ÉTAPES SUGGÉRÉES:")
    print(f"   1. Tester le modèle avec vos propres images")
    print(f"   2. Déployer l'application Streamlit")
    print(f"   3. Mettre en ligne sur Streamlit Cloud")
    print(f"   4. Analyser les courbes d'apprentissage")
    
    fin_total = datetime.now()
    print(f"\n✅ Script terminé à {fin_total.strftime('%H:%M:%S')}")
    print(f"🕐 Durée totale du processus: {fin_total - debut_total}")
    
    return True

def creer_visualisations_finales(historique, y_true, y_pred, noms_classes):
    """Créer les visualisations finales des résultats."""
    
    print(f"🎨 Création des visualisations...")
    
    # 1. Courbes d'apprentissage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    epochs = range(1, len(historique.history['accuracy']) + 1)
    ax1.plot(epochs, historique.history['accuracy'], 'b-', label='Entraînement', linewidth=2)
    ax1.plot(epochs, historique.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    ax1.set_title('Évolution de la Précision', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Précision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(epochs, historique.history['loss'], 'b-', label='Entraînement', linewidth=2)
    ax2.plot(epochs, historique.history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax2.set_title('Évolution de la Perte', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Perte')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/courbes_apprentissage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Matrice de confusion
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=noms_classes, yticklabels=noms_classes,
                square=True, linewidths=0.5)
    
    plt.title('Matrice de Confusion - Modèle Final', fontsize=16, fontweight='bold')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Étiquettes')
    plt.tight_layout()
    plt.savefig('results/matrice_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualisations sauvegardées:")
    print(f"   📈 results/courbes_apprentissage.png")
    print(f"   📊 results/matrice_confusion.png")

if __name__ == "__main__":
    """
    Lancer l'entraînement complet.
    """
    
    print("🏋️ LANCEMENT DE L'ENTRAÎNEMENT COMPLET FRUIVISION")
    print("=" * 70)
    print(f"🕐 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = entrainement_complet()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 ENTRAÎNEMENT COMPLET RÉUSSI!")
        print("🚀 Votre modèle EduFruis est prêt pour la production!")
    else:
        print("\n" + "=" * 70)
        print("❌ ENTRAÎNEMENT INTERROMPU OU ÉCHOUÉ")
        print("🔧 Vérifiez les erreurs et relancez si nécessaire")