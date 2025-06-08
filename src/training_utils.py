"""
FruitVision - Utilitaires d'Entraînement
========================================

Ce module contient des fonctions utilitaires pour l'entraînement, l'évaluation et la visualisation
des performances du modèle CNN.

Auteur: Mamadou Fall
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd


class GestionnaireEntrainement:
    """
    Classe pour gérer l'entraînement et l'évaluation du modèle.
    
    Cette classe centralise:
    - La configuration des callbacks
    - La sauvegarde de l'historique
    - L'évaluation des performances
    - La génération de visualisations
    """
    
    def __init__(self, dossier_resultats: str = "results"):
        """
        Initialiser le gestionnaire d'entraînement.
        
        Args:
            dossier_resultats (str): Dossier pour sauvegarder les résultats
        """
        self.dossier_resultats = dossier_resultats
        self.historique = None
        self.modele = None
        self.noms_classes = None
        
        # Créer le dossier de résultats
        os.makedirs(dossier_resultats, exist_ok=True)
        
        print(f"📁 Gestionnaire d'entraînement initialisé")
        print(f"   Dossier de résultats: {dossier_resultats}")
    
    
    def configurer_callbacks(self, config_arret_precoce: Dict, config_reduction_lr: Dict):
        """
        Configurer les callbacks pour l'entraînement.
        
        Pourquoi ces callbacks?
        - EarlyStopping: Évite le surapprentissage en arrêtant quand la performance stagne
        - ReduceLROnPlateau: Réduit le taux d'apprentissage pour affiner l'entraînement
        - ModelCheckpoint: Sauvegarde les meilleurs modèles
        
        Args:
            config_arret_precoce (Dict): Configuration pour l'arrêt précoce
            config_reduction_lr (Dict): Configuration pour la réduction du LR
            
        Returns:
            List: Liste des callbacks configurés
        """
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        except ImportError:
            print("⚠️ TensorFlow non disponible - callbacks simulés")
            return []
        
        callbacks = []
        
        # 1. ARRÊT PRÉCOCE (Early Stopping)
        early_stopping = EarlyStopping(
            monitor=config_arret_precoce['surveiller'],
            patience=config_arret_precoce['patience'],
            min_delta=config_arret_precoce['delta_min'],
            mode=config_arret_precoce['mode'],
            restore_best_weights=config_arret_precoce['restaurer_meilleurs_poids'],
            verbose=config_arret_precoce['verbose']
        )
        callbacks.append(early_stopping)
        
        # 2. RÉDUCTION DU TAUX D'APPRENTISSAGE
        reduce_lr = ReduceLROnPlateau(
            monitor=config_reduction_lr['surveiller'],
            factor=config_reduction_lr['facteur'],
            patience=config_reduction_lr['patience'],
            min_lr=config_reduction_lr['lr_min'],
            verbose=config_reduction_lr['verbose']
        )
        callbacks.append(reduce_lr)
        
        # 3. SAUVEGARDE DU MEILLEUR MODÈLE
        chemin_checkpoint = os.path.join(self.dossier_resultats, 'meilleur_modele.h5')
        model_checkpoint = ModelCheckpoint(
            filepath=chemin_checkpoint,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        print("✅ Callbacks configurés:")
        print(f"   🛑 Arrêt précoce: patience={config_arret_precoce['patience']}")
        print(f"   📉 Réduction LR: facteur={config_reduction_lr['facteur']}")
        print(f"   💾 Sauvegarde: {chemin_checkpoint}")
        
        return callbacks
    
    
    def sauvegarder_historique(self, historique, nom_fichier: str = "historique_entrainement.json"):
        """
        Sauvegarder l'historique d'entraînement.
        
        Args:
            historique: Historique retourné par model.fit()
            nom_fichier (str): Nom du fichier de sauvegarde
        """
        chemin_complet = os.path.join(self.dossier_resultats, nom_fichier)
        
        # Convertir l'historique en format JSON sérialisable
        historique_dict = {}
        for cle, valeurs in historique.history.items():
            # Convertir les valeurs numpy en listes Python
            historique_dict[cle] = [float(v) for v in valeurs]
        
        # Ajouter des métadonnées
        historique_dict['metadata'] = {
            'date_entrainement': datetime.now().isoformat(),
            'nb_epoques_completees': len(historique_dict['loss']),
            'meilleure_val_accuracy': max(historique_dict.get('val_accuracy', [0])),
            'epoque_meilleure_accuracy': historique_dict.get('val_accuracy', [0]).index(
                max(historique_dict.get('val_accuracy', [0]))
            ) + 1 if 'val_accuracy' in historique_dict else 0
        }
        
        # Sauvegarder en JSON
        with open(chemin_complet, 'w', encoding='utf-8') as f:
            json.dump(historique_dict, f, indent=2, ensure_ascii=False)
        
        self.historique = historique_dict
        print(f"📊 Historique sauvegardé: {chemin_complet}")
        print(f"   🏆 Meilleure val_accuracy: {historique_dict['metadata']['meilleure_val_accuracy']:.4f}")
    
    
    def tracer_courbes_apprentissage(self, historique=None, nom_fichier: str = "courbes_apprentissage.png"):
        """
        Créer et sauvegarder les courbes d'apprentissage.
        
        Args:
            historique: Historique d'entraînement (utilise self.historique si None)
            nom_fichier (str): Nom du fichier image
        """
        if historique is None:
            historique = self.historique
        
        if historique is None:
            print("❌ Pas d'historique disponible pour tracer les courbes")
            return
        
        # Configuration du graphique
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Courbe 1: Accuracy (Précision)
        epoques = range(1, len(historique['accuracy']) + 1)
        
        ax1.plot(epoques, historique['accuracy'], 'b-', label='Précision Entraînement', linewidth=2)
        if 'val_accuracy' in historique:
            ax1.plot(epoques, historique['val_accuracy'], 'r-', label='Précision Validation', linewidth=2)
        
        ax1.set_title('Évolution de la Précision', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Précision')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Courbe 2: Loss (Perte)
        ax2.plot(epoques, historique['loss'], 'b-', label='Perte Entraînement', linewidth=2)
        if 'val_loss' in historique:
            ax2.plot(epoques, historique['val_loss'], 'r-', label='Perte Validation', linewidth=2)
        
        ax2.set_title('Évolution de la Perte', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Perte')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Ajouter des annotations pour les meilleurs résultats
        if 'val_accuracy' in historique:
            meilleure_val_acc = max(historique['val_accuracy'])
            meilleure_epoque = historique['val_accuracy'].index(meilleure_val_acc) + 1
            
            ax1.annotate(f'Meilleur: {meilleure_val_acc:.3f}\n(Époque {meilleure_epoque})',
                        xy=(meilleure_epoque, meilleure_val_acc),
                        xytext=(meilleure_epoque + len(epoques)*0.1, meilleure_val_acc),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                        fontsize=10)
        
        plt.tight_layout()
        
        # Sauvegarder
        chemin_complet = os.path.join(self.dossier_resultats, nom_fichier)
        plt.savefig(chemin_complet, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Courbes d'apprentissage sauvegardées: {chemin_complet}")
    
    
    def evaluer_modele(self, modele, X_test: np.ndarray, y_test: np.ndarray, 
                      noms_classes: List[str]) -> Dict:
        """
        Évaluer le modèle sur l'ensemble de test.
        
        Args:
            modele: Modèle entraîné
            X_test (np.ndarray): Images de test
            y_test (np.ndarray): Labels de test (one-hot)
            noms_classes (List[str]): Noms des classes
            
        Returns:
            Dict: Métriques d'évaluation
        """
        print("🧪 Évaluation du modèle sur l'ensemble de test...")
        
        # Prédictions
        y_pred_proba = modele.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Métriques globales
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Métriques par classe
        rapport_classification = classification_report(
            y_true, y_pred, target_names=noms_classes, output_dict=True
        )
        
        # Préparer les résultats
        resultats = {
            'accuracy_globale': float(accuracy),
            'precision_moyenne': float(precision),
            'recall_moyen': float(recall),
            'f1_score_moyen': float(f1),
            'rapport_par_classe': rapport_classification,
            'nb_echantillons_test': len(y_test),
            'predictions_correctes': int(np.sum(y_pred == y_true)),
            'predictions_incorrectes': int(np.sum(y_pred != y_true))
        }
        
        # Sauvegarder les métriques
        self.noms_classes = noms_classes
        self._sauvegarder_metriques(resultats)
        
        # Afficher un résumé
        print(f"✅ Évaluation terminée:")
        print(f"   🎯 Précision globale: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📊 F1-Score moyen: {f1:.4f}")
        print(f"   ✅ Prédictions correctes: {resultats['predictions_correctes']}/{resultats['nb_echantillons_test']}")
        
        return resultats
    
    
    def creer_matrice_confusion(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               noms_classes: List[str], nom_fichier: str = "matrice_confusion.png"):
        """
        Créer et sauvegarder la matrice de confusion.
        
        Args:
            y_true (np.ndarray): Vraies étiquettes
            y_pred (np.ndarray): Étiquettes prédites
            noms_classes (List[str]): Noms des classes
            nom_fichier (str): Nom du fichier de sauvegarde
        """
        # Calculer la matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        # Créer le graphique
        plt.figure(figsize=(10, 8))
        
        # Utiliser seaborn pour une matrice plus jolie
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=noms_classes, yticklabels=noms_classes,
                   square=True, linewidths=0.5)
        
        plt.title('Matrice de Confusion - FruitVision', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Prédictions du Modèle', fontsize=12)
        plt.ylabel('Vraies Étiquettes', fontsize=12)
        
        # Rotation des labels pour meilleure lisibilité
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Sauvegarder
        chemin_complet = os.path.join(self.dossier_resultats, nom_fichier)
        plt.savefig(chemin_complet, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Matrice de confusion sauvegardée: {chemin_complet}")
        
        # Analyser les erreurs communes
        self._analyser_erreurs_confusion(cm, noms_classes)
    
    
    def _analyser_erreurs_confusion(self, cm: np.ndarray, noms_classes: List[str]):
        """
        Analyser les erreurs les plus communes dans la matrice de confusion.
        
        Args:
            cm (np.ndarray): Matrice de confusion
            noms_classes (List[str]): Noms des classes
        """
        print("\n🔍 Analyse des erreurs les plus fréquentes:")
        
        # Trouver les erreurs (éléments hors diagonale)
        erreurs = []
        for i in range(len(noms_classes)):
            for j in range(len(noms_classes)):
                if i != j and cm[i, j] > 0:  # Erreur (pas sur la diagonale)
                    erreurs.append((cm[i, j], noms_classes[i], noms_classes[j]))
        
        # Trier par nombre d'erreurs décroissant
        erreurs.sort(reverse=True)
        
        # Afficher les 3 erreurs les plus fréquentes
        print("   Top 3 des confusions:")
        for i, (nb_erreurs, vraie_classe, classe_predite) in enumerate(erreurs[:3]):
            print(f"   {i+1}. {vraie_classe} → {classe_predite}: {nb_erreurs} erreurs")
        
        if not erreurs:
            print("   🎉 Aucune erreur détectée! Modèle parfait!")
    
    
    def _sauvegarder_metriques(self, resultats: Dict, nom_fichier: str = "metriques_evaluation.json"):
        """
        Sauvegarder les métriques d'évaluation.
        
        Args:
            resultats (Dict): Résultats de l'évaluation
            nom_fichier (str): Nom du fichier
        """
        chemin_complet = os.path.join(self.dossier_resultats, nom_fichier)
        
        # Ajouter timestamp
        resultats['metadata'] = {
            'date_evaluation': datetime.now().isoformat(),
            'modele_utilise': 'FruitVision CNN'
        }
        
        with open(chemin_complet, 'w', encoding='utf-8') as f:
            json.dump(resultats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Métriques sauvegardées: {chemin_complet}")


# Fonctions utilitaires indépendantes
def calculer_temps_entrainement(debut: datetime, fin: datetime) -> str:
    """
    Calculer et formater le temps d'entraînement.
    
    Args:
        debut (datetime): Heure de début
        fin (datetime): Heure de fin
        
    Returns:
        str: Temps formaté
    """
    duree = fin - debut
    heures = duree.seconds // 3600
    minutes = (duree.seconds % 3600) // 60
    secondes = duree.seconds % 60
    
    if heures > 0:
        return f"{heures}h {minutes}m {secondes}s"
    elif minutes > 0:
        return f"{minutes}m {secondes}s"
    else:
        return f"{secondes}s"


def afficher_resume_modele(modele, noms_classes: List[str]):
    """
    Afficher un résumé du modèle avec des informations utiles.
    
    Args:
        modele: Modèle Keras
        noms_classes (List[str]): Noms des classes
    """
    print("🧠 Résumé du Modèle FruitVision")
    print("=" * 50)
    
    # Informations générales
    try:
        total_params = modele.count_params()
        print(f"📊 Paramètres totaux: {total_params:,}")
        print(f"💾 Taille estimée: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    except:
        print("📊 Informations sur les paramètres non disponibles")
    
    print(f"🍎 Classes à reconnaître: {len(noms_classes)}")
    for i, nom in enumerate(noms_classes):
        print(f"   {i}: {nom}")
    
    print(f"📐 Forme d'entrée: {modele.input_shape}")
    print(f"📤 Forme de sortie: {model.output_shape}")


# Test du module
if __name__ == "__main__":
    """
    Code de test pour les utilitaires d'entraînement.
    """
    print("🧪 Test des Utilitaires d'Entraînement")
    print("=" * 50)
    
    # Test du gestionnaire
    gestionnaire = GestionnaireEntrainement("test_results")
    
    # Test avec des données factices
    print("\n🔬 Test avec données factices...")
    
    # Simuler un historique d'entraînement
    historique_factice = {
        'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_accuracy': [0.5, 0.6, 0.75, 0.8, 0.82],
        'loss': [1.5, 1.2, 0.8, 0.5, 0.3],
        'val_loss': [1.7, 1.3, 0.9, 0.6, 0.4]
    }
    
    gestionnaire.historique = historique_factice
    
    # Test des courbes d'apprentissage
    print("📈 Test des courbes d'apprentissage...")
    gestionnaire.tracer_courbes_apprentissage()
    
    # Test de la matrice de confusion avec données factices
    print("📊 Test de la matrice de confusion...")
    y_true_factice = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred_factice = np.array([0, 1, 1, 0, 2, 2, 1, 1])
    noms_classes_test = ['Pomme', 'Banane', 'Kiwi']
    
    gestionnaire.creer_matrice_confusion(y_true_factice, y_pred_factice, noms_classes_test)
    
    print("\n✅ Tous les tests des utilitaires passés!")
    print("📌 Prêt pour l'entraînement réel du modèle")