"""
EduFruis - Configuration du Modèle
=====================================

Ce fichier contient toutes les configurations et hyperparamètres pour le projet EduFruis.
Centraliser la configuration facilite les expérimentations et la reproductibilité.

Auteur: Mamadou Fall
Date: 2025
"""

import os
from pathlib import Path

# ============================================================================
# CONFIGURATION GÉNÉRALE
# ============================================================================

# Chemins des dossiers
DOSSIER_PROJET = Path(__file__).parent.parent  
DOSSIER_DONNEES = DOSSIER_PROJET / "data" / "fruits-360_100x100"
DOSSIER_MODELES = DOSSIER_PROJET / "models"
DOSSIER_RESULTATS = DOSSIER_PROJET / "results"

# Classes de fruits (correspondance avec le dataset Fruits-360)
CLASSES_FRUITS = {
    'Pomme': ['Apple Braeburn', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
              'Apple Granny Smith', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
    'Banane': ['Banana', 'Banana Red'],
    'Kiwi': ['Kiwi'],
    'Citron': ['Lemon', 'Lemon Meyer'],
    'Peche': ['Peach', 'Peach 2']
}

# Noms des classes pour affichage
NOMS_CLASSES = list(CLASSES_FRUITS.keys())
NOMBRE_CLASSES = len(NOMS_CLASSES)

# ============================================================================
# CONFIGURATION DES IMAGES
# ============================================================================

# Dimensions des images
TAILLE_IMAGE = (100, 100)  # (largeur, hauteur)
CANAUX_IMAGE = 3           # RGB
FORME_ENTREE = (*TAILLE_IMAGE, CANAUX_IMAGE)  # (100, 100, 3)

# Prétraitement
NORMALISATION_MIN = 0.0
NORMALISATION_MAX = 1.0

# ============================================================================
# CONFIGURATION DU MODÈLE CNN
# ============================================================================

# Architecture
CONFIG_MODELE = {
    'forme_entree': FORME_ENTREE,
    'nb_classes': NOMBRE_CLASSES,
    
    # Couches convolutionnelles
    'filtres_conv': [32, 64, 128],  # Nombre de filtres par couche
    'taille_noyau': (3, 3),         # Taille des filtres de convolution
    'activation_conv': 'relu',       # Fonction d'activation
    'padding': 'same',               # Padding pour garder la taille
    
    # Pooling
    'taille_pool': (2, 2),
    
    # Dropout (régularisation)
    'taux_dropout_conv': 0.25,      # Dropout après convolution
    'taux_dropout_dense': 0.5,      # Dropout avant sortie
    
    # Couche dense
    'unites_dense': 128,             # Nombre de neurones dans la couche dense
    'activation_dense': 'relu',
    
    # Couche de sortie
    'activation_sortie': 'softmax'   # Pour classification multi-classes
}

# ============================================================================
# CONFIGURATION DE L'ENTRAÎNEMENT
# ============================================================================

CONFIG_ENTRAINEMENT = {
    # Optimiseur
    'optimiseur': 'adam',
    'taux_apprentissage': 0.001,
    'beta_1': 0.9,                   # Paramètre momentum Adam
    'beta_2': 0.999,                 # Paramètre RMSprop Adam
    'epsilon': 1e-07,                # Petite valeur pour stabilité
    
    # Fonction de perte et métriques
    'fonction_perte': 'categorical_crossentropy',
    'metriques': ['accuracy', 'top_2_accuracy'],
    
    # Paramètres d'entraînement
    'nb_epoques': 50,                # Nombre maximum d'époques
    'taille_batch': 32,              # Nombre d'images par batch
    'melanger_donnees': True,        # Mélanger les données à chaque époque
    
    # Validation
    'frequence_validation': 1,       # Valider à chaque époque
    'verbose_entrainement': 1,       # Affichage détaillé
}

# ============================================================================
# CONFIGURATION DE L'ARRÊT PRÉCOCE (EARLY STOPPING)
# ============================================================================

CONFIG_ARRET_PRECOCE = {
    'surveiller': 'val_accuracy',    # Métrique à surveiller
    'patience': 10,                  # Nombre d'époques sans amélioration
    'delta_min': 0.001,             # Amélioration minimale considérée
    'mode': 'max',                   # Maximiser l'accuracy
    'restaurer_meilleurs_poids': True,
    'verbose': 1
}

# ============================================================================
# CONFIGURATION DE LA RÉDUCTION DU TAUX D'APPRENTISSAGE
# ============================================================================

CONFIG_REDUCTION_LR = {
    'surveiller': 'val_loss',        # Surveiller la perte de validation
    'facteur': 0.5,                  # Multiplier le LR par ce facteur
    'patience': 5,                   # Époques avant réduction
    'lr_min': 1e-6,                  # Taux d'apprentissage minimum
    'verbose': 1
}

# ============================================================================
# CONFIGURATION DES DONNÉES
# ============================================================================

CONFIG_DONNEES = {
    # Division des données
    'taille_validation': 0.2,        # 20% pour validation
    'taille_test': 0.1,              # 10% pour test
    'graine_aleatoire': 42,          # Pour reproductibilité
    
    # Augmentation de données
    'utiliser_augmentation': True,
    'augmentations': {
        'rotation_max': 15,          # Rotation max en degrés
        'variation_luminosite': [0.8, 1.2],  # Facteurs de luminosité
        'variation_contraste': [0.9, 1.1],   # Facteurs de contraste
        'flou_gaussien': 0.5,        # Rayon du flou
        'miroir_horizontal': True,    # Appliquer miroir horizontal
    },
    
    # Limitations (pour tests rapides)
    'max_images_par_classe': None,   # Limiter pour tests (None = toutes)
    'classes_a_utiliser': None,      # Utiliser seulement certaines classes (None = toutes)
}

# ============================================================================
# CONFIGURATION DE SAUVEGARDE
# ============================================================================

CONFIG_SAUVEGARDE = {
    # Noms des fichiers
    'nom_modele': 'fruivision_model.h5',
    'nom_historique': 'historique_entrainement.json',
    'nom_metriques': 'metriques_evaluation.json',
    'nom_matrice_confusion': 'matrice_confusion.png',
    'nom_courbes_apprentissage': 'courbes_apprentissage.png',
    
    # Options de sauvegarde
    'sauvegarder_a_chaque_epoque': False,
    'sauvegarder_meilleur_seulement': True,
    'format_modele': 'h5',           # Format de sauvegarde du modèle
    
    # Checkpoints
    'dossier_checkpoints': 'checkpoints',
    'nom_checkpoint': 'checkpoint_epoque_{epoch:02d}_val_acc_{val_accuracy:.3f}.h5'
}

# ============================================================================
# CONFIGURATION DE L'APPLICATION STREAMLIT
# ============================================================================

CONFIG_APP = {
    'titre': 'EduFruis - Reconnaissance de Fruits par IA',
    'description': 'Téléversez une image de fruit et laissez l\'IA le reconnaître!',
    'auteur': 'Mamadou Fall',
    
    # Interface
    'largeur_max_image': 400,        # Largeur max pour affichage
    'formats_acceptes': ['jpg', 'jpeg', 'png'],
    'taille_max_fichier': 5,         # MB
    
    # Prédiction
    'seuil_confiance': 0.3,          # Seuil minimum de confiance
    'afficher_top_k': 3,             # Afficher les 3 meilleures prédictions
    
    # Couleurs et style
    'couleur_principale': '#FF6B6B',
    'couleur_secondaire': '#4ECDC4',
    'couleur_succes': '#45B7D1',
}

# ============================================================================
# CONFIGURATION DU DÉPLOIEMENT
# ============================================================================

CONFIG_DEPLOIEMENT = {
    'plateforme': 'streamlit_cloud',  # streamlit_cloud, heroku, etc.
    'region': 'us-east-1',
    'instance_type': 'small',
    
    # Optimisations pour le déploiement
    'compression_modele': True,       # Compresser le modèle
    'cache_predictions': True,        # Mettre en cache les prédictions
    'limite_requetes': 100,           # Requêtes par heure
}

# ============================================================================
# FONCTIONS UTILITAIRES DE CONFIGURATION
# ============================================================================

def creer_dossiers():
    """Créer tous les dossiers nécessaires pour le projet."""
    dossiers = [
        DOSSIER_DONNEES,
        DOSSIER_MODELES,
        DOSSIER_RESULTATS,
        DOSSIER_RESULTATS / CONFIG_SAUVEGARDE['dossier_checkpoints'],
    ]
    
    for dossier in dossiers:
        dossier.mkdir(parents=True, exist_ok=True)
        print(f"📁 Dossier créé/vérifié: {dossier}")


def afficher_config():
    """Afficher un résumé de la configuration actuelle."""
    print("⚙️ Configuration EduFruis")
    print("=" * 50)
    print(f"🍎 Nombre de classes: {NOMBRE_CLASSES}")
    print(f"📐 Taille des images: {TAILLE_IMAGE}")
    print(f"🧠 Architecture CNN: {CONFIG_MODELE['filtres_conv']} filtres")
    print(f"📚 Époques max: {CONFIG_ENTRAINEMENT['nb_epoques']}")
    print(f"📊 Taille batch: {CONFIG_ENTRAINEMENT['taille_batch']}")
    print(f"🎯 Objectif accuracy: >85%")
    print(f"💾 Modèle sauvegardé: {CONFIG_SAUVEGARDE['nom_modele']}")


def obtenir_chemin_modele():
    """Obtenir le chemin complet vers le modèle sauvegardé."""
    return DOSSIER_MODELES / CONFIG_SAUVEGARDE['nom_modele']


def obtenir_config_complete():
    """Retourner toute la configuration sous forme de dictionnaire."""
    return {
        'classes_fruits': CLASSES_FRUITS,
        'noms_classes': NOMS_CLASSES,
        'nombre_classes': NOMBRE_CLASSES,
        'taille_image': TAILLE_IMAGE,
        'forme_entree': FORME_ENTREE,
        'config_modele': CONFIG_MODELE,
        'config_entrainement': CONFIG_ENTRAINEMENT,
        'config_arret_precoce': CONFIG_ARRET_PRECOCE,
        'config_reduction_lr': CONFIG_REDUCTION_LR,
        'config_donnees': CONFIG_DONNEES,
        'config_sauvegarde': CONFIG_SAUVEGARDE,
        'config_app': CONFIG_APP,
        'config_deploiement': CONFIG_DEPLOIEMENT,
    }


# ============================================================================
# VALIDATION DE LA CONFIGURATION
# ============================================================================

def valider_configuration():
    """Valider que la configuration est cohérente."""
    erreurs = []
    
    # Vérifier les dimensions
    if len(TAILLE_IMAGE) != 2:
        erreurs.append("TAILLE_IMAGE doit avoir 2 dimensions (largeur, hauteur)")
    
    if NOMBRE_CLASSES != len(NOMS_CLASSES):
        erreurs.append("NOMBRE_CLASSES ne correspond pas à len(NOMS_CLASSES)")
    
    # Vérifier les paramètres d'entraînement
    if CONFIG_ENTRAINEMENT['taux_apprentissage'] <= 0:
        erreurs.append("Le taux d'apprentissage doit être positif")
    
    if CONFIG_ENTRAINEMENT['nb_epoques'] <= 0:
        erreurs.append("Le nombre d'époques doit être positif")
    
    if CONFIG_ENTRAINEMENT['taille_batch'] <= 0:
        erreurs.append("La taille de batch doit être positive")
    
    # Vérifier les taux de dropout
    if not (0 <= CONFIG_MODELE['taux_dropout_conv'] <= 1):
        erreurs.append("Le taux de dropout convolutionnel doit être entre 0 et 1")
    
    if not (0 <= CONFIG_MODELE['taux_dropout_dense'] <= 1):
        erreurs.append("Le taux de dropout dense doit être entre 0 et 1")
    
    # Vérifier les paramètres de données
    total_split = CONFIG_DONNEES['taille_validation'] + CONFIG_DONNEES['taille_test']
    if total_split >= 1.0:
        erreurs.append("La somme des tailles de validation et test doit être < 1.0")
    
    if erreurs:
        print("❌ Erreurs de configuration détectées:")
        for erreur in erreurs:
            print(f"   • {erreur}")
        return False
    else:
        print("✅ Configuration validée avec succès!")
        return True


# Test de la configuration
if __name__ == "__main__":
    """
    Code de test pour vérifier la configuration.
    """
    print("🧪 Test de la Configuration EduFruis")
    print("=" * 50)
    
    # Afficher la configuration
    afficher_config()
    
    print("\n🔍 Validation de la configuration...")
    config_valide = valider_configuration()
    
    if config_valide:
        print("\n📁 Création des dossiers...")
        creer_dossiers()
        
        print("\n📋 Détails des classes de fruits:")
        for fruit, categories in CLASSES_FRUITS.items():
            print(f"   {fruit}: {len(categories)} catégories")
            for cat in categories:
                print(f"      - {cat}")
        
        print(f"\n💾 Chemin du modèle: {obtenir_chemin_modele()}")
        
        print("\n✅ Configuration prête pour l'entraînement!")
    else:
        print("\n❌ Corrigez les erreurs avant de continuer.")