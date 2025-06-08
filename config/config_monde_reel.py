"""
Configuration d'Entraînement pour Images du Monde Réel
======================================================

Configuration spécialisée pour entraîner un modèle robuste capable de gérer
les images uploadées par les utilisateurs dans des conditions réelles.

Auteur: Mamadou Fall
Date: 2025
"""

# Configuration d'augmentation de données aggressive
CONFIG_AUGMENTATION_ROBUSTE = {
    # Rotations plus importantes
    'rotation_range': 30,                    # Au lieu de 15°
    
    # Variations de luminosité et contraste
    'brightness_range': [0.6, 1.4],         # 40% variation de luminosité
    'contrast_range': [0.7, 1.3],           # 30% variation de contraste
    
    # Variations géométriques
    'width_shift_range': 0.2,               # Décalage horizontal 20%
    'height_shift_range': 0.2,              # Décalage vertical 20%
    'zoom_range': [0.8, 1.2],               # Zoom 80%-120%
    'shear_range': 10,                       # Cisaillement 10°
    
    # Miroirs et retournements
    'horizontal_flip': True,                 # Miroir horizontal
    'vertical_flip': False,                  # Pas de miroir vertical (fruits)
    
    # Variations de couleur
    'channel_shift_range': 30,               # Décalage des canaux couleur
    'hue_shift_range': 15,                   # Décalage de teinte
    'saturation_range': [0.8, 1.2],         # Variation de saturation
    
    # Ajout de bruit
    'noise_factor': 0.1,                     # Bruit gaussien léger
    
    # Variations de flou
    'blur_range': [0, 2],                    # Flou 0-2 pixels
    
    # Simulation de conditions réelles
    'shadow_probability': 0.3,               # 30% chance d'ombres
    'lighting_variations': True,             # Variations d'éclairage
    'background_variations': True            # Variations de fond
}

# Configuration d'entraînement robuste
CONFIG_ENTRAINEMENT_ROBUSTE = {
    # Paramètres généraux
    'epochs': 100,                           # Plus d'époques pour robustesse
    'batch_size': 32,
    'learning_rate_initial': 0.001,
    
    # Planification du taux d'apprentissage
    'learning_rate_schedule': {
        'type': 'reduce_on_plateau',
        'factor': 0.5,                       # Réduire par 2
        'patience': 8,                       # Attendre 8 époques
        'min_lr': 1e-6                       # LR minimum
    },
    
    # Arrêt précoce plus patient
    'early_stopping': {
        'patience': 20,                      # Plus patient pour images réelles
        'monitor': 'val_accuracy',
        'min_delta': 0.001,
        'restore_best_weights': True
    },
    
    # Régularisation renforcée
    'dropout_rates': {
        'conv_layers': 0.3,                  # Dropout convolutionnel
        'dense_layers': 0.6                  # Dropout dense plus élevé
    },
    
    # Validation croisée
    'validation_split': 0.25,               # 25% pour validation
    'test_split': 0.15,                     # 15% pour test final
    
    # Métriques à surveiller
    'metrics': ['accuracy', 'precision', 'recall'],
    
    # Sauvegardes multiples
    'save_best_only': True,
    'save_checkpoints': True,
    'checkpoint_frequency': 10               # Tous les 10 époques
}

# Configuration de l'architecture pour robustesse
CONFIG_ARCHITECTURE_ROBUSTE = {
    # Couches convolutionnelles avec plus de régularisation
    'conv_layers': [
        {'filters': 32, 'kernel_size': (3, 3), 'dropout': 0.25},
        {'filters': 64, 'kernel_size': (3, 3), 'dropout': 0.25},
        {'filters': 128, 'kernel_size': (3, 3), 'dropout': 0.3},
        {'filters': 256, 'kernel_size': (3, 3), 'dropout': 0.3}  # Couche supplémentaire
    ],
    
    # Couches denses avec plus de régularisation
    'dense_layers': [
        {'units': 256, 'dropout': 0.5},     # Plus de neurones
        {'units': 128, 'dropout': 0.6}      # Dropout plus élevé
    ],
    
    # Normalisation par batch
    'batch_normalization': True,
    
    # Régularisation L2
    'l2_regularization': 0.001,
    
    # Activation
    'activation': 'relu',
    'output_activation': 'softmax'
}

# Configuration de test sur images réelles
CONFIG_TEST_MONDE_REEL = {
    # Seuils de confiance
    'seuil_confiance_eleve': 0.8,           # Prédiction fiable
    'seuil_confiance_moyen': 0.5,           # Prédiction acceptable
    'seuil_confiance_faible': 0.3,          # Prédiction incertaine
    
    # Gestion des prédictions incertaines
    'action_confiance_faible': 'warn_user', # Avertir l'utilisateur
    'suggestions_amelioration': True,        # Donner des conseils
    
    # Métriques de qualité d'image
    'seuils_qualite': {
        'luminosite_min': 30,               # Luminosité minimale
        'luminosite_max': 230,              # Luminosité maximale
        'contraste_min': 20,                # Contraste minimal
        'nettete_min': 100                  # Netteté minimale (variance Laplacien)
    },
    
    # Post-processing des résultats
    'smooth_predictions': True,             # Lisser les prédictions
    'ensemble_predictions': False,          # Pas d'ensemble pour l'instant
    
    # Logging et analytics
    'log_predictions': True,                # Logger les prédictions
    'track_performance': True,              # Suivre les performances
    'save_failed_cases': True               # Sauver les cas d'échec
}

# Configuration de l'application Streamlit
CONFIG_APPLICATION = {
    # Interface utilisateur
    'titre': 'FruitVision - IA de Reconnaissance de Fruits',
    'description': 'Uploadez votre photo et laissez l\'IA reconnaître le fruit!',
    
    # Limitations d'upload
    'taille_max_fichier': 10,               # MB
    'formats_acceptes': ['jpg', 'jpeg', 'png'],
    'resolution_max': (2048, 2048),         # Pixels
    'resolution_min': (50, 50),             # Pixels
    
    # Affichage des résultats
    'afficher_top_k': 3,                    # Top 3 prédictions
    'afficher_confiance': True,             # Barres de confiance
    'afficher_conseils': True,              # Conseils d'amélioration
    
    # Mode développeur
    'mode_debug': False,                    # Infos de debug
    'afficher_temps_traitement': True,      # Temps de prédiction
    'sauver_historique': True,              # Historique des prédictions
    
    # Personnalisation visuelle
    'theme_couleurs': {
        'primaire': '#FF6B6B',
        'secondaire': '#4ECDC4',
        'succes': '#45B7D1',
        'avertissement': '#FFA726',
        'erreur': '#EF5350'
    }
}

# Fonction de validation de configuration
def valider_configuration():
    """
    Valider que toutes les configurations sont cohérentes.
    
    Returns:
        bool: True si la configuration est valide
    """
    
    erreurs = []
    
    # Vérifier les paramètres d'entraînement
    if CONFIG_ENTRAINEMENT_ROBUSTE['epochs'] <= 0:
        erreurs.append("Le nombre d'époques doit être positif")
    
    if CONFIG_ENTRAINEMENT_ROBUSTE['batch_size'] <= 0:
        erreurs.append("La taille de batch doit être positive")
    
    # Vérifier les taux de dropout
    for couche, taux in CONFIG_ENTRAINEMENT_ROBUSTE['dropout_rates'].items():
        if not (0 <= taux <= 1):
            erreurs.append(f"Taux de dropout invalide pour {couche}: {taux}")
    
    # Vérifier les seuils de confiance
    seuils = [
        CONFIG_TEST_MONDE_REEL['seuil_confiance_eleve'],
        CONFIG_TEST_MONDE_REEL['seuil_confiance_moyen'],
        CONFIG_TEST_MONDE_REEL['seuil_confiance_faible']
    ]
    
    if not all(0 <= s <= 1 for s in seuils):
        erreurs.append("Les seuils de confiance doivent être entre 0 et 1")
    
    if not (seuils[0] > seuils[1] > seuils[2]):
        erreurs.append("Les seuils de confiance doivent être décroissants")
    
    # Afficher les erreurs
    if erreurs:
        print("❌ Erreurs de configuration:")
        for erreur in erreurs:
            print(f"   - {erreur}")
        return False
    else:
        print("✅ Configuration validée avec succès!")
        return True

# Classes de fruits mises à jour avec variantes réelles
CLASSES_FRUITS_MONDE_REEL = {
    'Pomme': {
        'variantes_dataset': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 
                             'Apple Braeburn 1', 'Apple Granny Smith 1', 
                             'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
        'couleurs_typiques': ['rouge', 'vert', 'jaune', 'bicolore'],
        'formes_typiques': ['ronde', 'légèrement aplatie'],
        'confusions_possibles': ['Pêche', 'Tomate (si ajoutée)'],
        'conseils_photo': [
            'Montrer la forme caractéristique',
            'Éviter les reflets sur la peau',
            'Bien distinguer du fond'
        ]
    },
    
    'Banane': {
        'variantes_dataset': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Red 1'],
        'couleurs_typiques': ['jaune', 'vert (pas mûre)', 'rouge'],
        'formes_typiques': ['allongée', 'courbée'],
        'confusions_possibles': ['Très peu - forme unique'],
        'conseils_photo': [
            'Montrer la courbure caractéristique',
            'Éviter les bananes trop vertes ou trop mûres',
            'Photographier sur la longueur'
        ]
    },
    
    'Kiwi': {
        'variantes_dataset': ['Kiwi 1'],
        'couleurs_typiques': ['brun (peau)', 'vert (chair)'],
        'formes_typiques': ['ovale', 'légèrement aplatie'],
        'confusions_possibles': ['Objets ronds bruns'],
        'conseils_photo': [
            'Montrer la texture velue caractéristique',
            'Éviter les objets ronds similaires en arrière-plan',
            'Bonne lumière pour voir la texture'
        ]
    },
    
    'Citron': {
        'variantes_dataset': ['Lemon 1', 'Lemon Meyer 1'],
        'couleurs_typiques': ['jaune vif', 'jaune-vert'],
        'formes_typiques': ['ovale', 'extrémités pointues'],
        'confusions_possibles': ['Lime', 'Objets jaunes'],
        'conseils_photo': [
            'Montrer la forme caractéristique (plus allongé que rond)',
            'Éviter l\'éclairage qui change la couleur',
            'Distinguer du fond jaune'
        ]
    },
    
    'Peche': {
        'variantes_dataset': ['Peach 1', 'Peach 2'],
        'couleurs_typiques': ['orange-rosé', 'jaune-orange'],
        'formes_typiques': ['ronde', 'légèrement aplatie', 'sillon caractéristique'],
        'confusions_possibles': ['Pomme', 'Abricot'],
        'conseils_photo': [
            'Montrer le duvet caractéristique',
            'Mettre en évidence le sillon',
            'Éviter la confusion avec les pommes rouges'
        ]
    }
}

if __name__ == "__main__":
    """
    Test et validation de la configuration.
    """
    
    print("🔧 Validation de la Configuration Monde Réel")
    print("=" * 60)
    
    # Valider la configuration
    config_valide = valider_configuration()
    
    if config_valide:
        print("\n📊 Résumé de la Configuration:")
        print(f"   Époques d'entraînement: {CONFIG_ENTRAINEMENT_ROBUSTE['epochs']}")
        print(f"   Augmentation de données: {len(CONFIG_AUGMENTATION_ROBUSTE)} paramètres")
        print(f"   Architecture: {len(CONFIG_ARCHITECTURE_ROBUSTE['conv_layers'])} couches conv")
        print(f"   Seuils de confiance: {CONFIG_TEST_MONDE_REEL['seuil_confiance_eleve']:.1f} / {CONFIG_TEST_MONDE_REEL['seuil_confiance_moyen']:.1f} / {CONFIG_TEST_MONDE_REEL['seuil_confiance_faible']:.1f}")
        
        print("\n🍎 Classes de fruits configurées:")
        for fruit, info in CLASSES_FRUITS_MONDE_REEL.items():
            print(f"   {fruit}: {len(info['variantes_dataset'])} variantes")
        
        print("\n✅ Configuration prête pour l'entraînement robuste!")
    else:
        print("\n❌ Configuration invalide - corrigez les erreurs avant de continuer")