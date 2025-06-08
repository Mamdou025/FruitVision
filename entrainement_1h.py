"""
Entraînement Simple Corrigé - Copie de entrainement_1h.py avec corrections
"""

# Copiez EXACTEMENT le code de votre entrainement_1h.py
# et modifiez seulement ces paramètres :

def configuration_entrainement_corrige():
    """Configuration corrigée pour éviter l'overfitting."""
    
    config = {
        # Données - RÉDUITES pour éviter l'overfitting
        'max_images_par_classe': 500,          # RÉDUIT de 1500 → 500
        'utiliser_augmentation': True,          
        'augmentation_factor': 2,               # RÉDUIT de 5 → 2
        'taille_validation': 0.3,              # AUGMENTÉ de 0.2 → 0.3 (plus de validation)
        'taille_test': 0.15,                   
        
        # Entraînement - Plus conservateur  
        'epochs': 30,                          # RÉDUIT de 40 → 30
        'batch_size': 32,                      # Gardé identique
        'learning_rate': 0.0003,               # RÉDUIT de 0.0008 → 0.0003
        
        # Callbacks - Plus stricts
        'early_stopping_patience': 5,         # RÉDUIT de 15 → 5
        'reduce_lr_patience': 3,               # RÉDUIT de 6 → 3
        'save_best_only': True,
        
        # Performance
        'verbose': 1,
        'validation_freq': 1,
        'workers': 1,
        'use_multiprocessing': False,
    }
    
    return config

# COPIEZ LE RESTE DE VOTRE entrainement_1h.py ICI
# mais changez la ligne :
# config = configuration_entrainement_1h()
# par :
# config = configuration_entrainement_corrige()

print("🔧 INSTRUCTIONS POUR CORRECTION RAPIDE:")
print("=" * 50)
print("1. Copiez votre entrainement_1h.py → entrainement_test.py")
print("2. Remplacez la fonction configuration_entrainement_1h() par celle ci-dessus")
print("3. Changez le nom du fichier de sauvegarde en 'fruivision_test.h5'")
print("4. Lancez: python entrainement_test.py")
print("")
print("MODIFICATIONS CLÉS:")
print("   📊 500 images/fruit au lieu de 1500 (moins d'overfitting)")
print("   ✨ Augmentation x2 au lieu de x5 (moins de bruit)")
print("   🔄 30 époques au lieu de 40")
print("   📈 LR 0.0003 au lieu de 0.0008 (plus conservateur)")
print("   ⏹️ Early stopping après 5 époques sans amélioration")
print("   🔍 30% validation au lieu de 20% (test plus strict)")
print("")
print("🎯 OBJECTIF: Éviter l'overfitting qui a causé le 'faux 100%'")