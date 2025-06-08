#!/usr/bin/env python3
"""
Test d'Installation FruitVision
===============================
Script pour vérifier que toutes les dépendances sont correctement installées.
"""

import sys

def tester_import(nom_module, nom_affichage=None):
    """Tester l'importation d'un module."""
    if nom_affichage is None:
        nom_affichage = nom_module
    
    try:
        __import__(nom_module)
        print(f"✅ {nom_affichage} - Installé")
        return True
    except ImportError as e:
        print(f"❌ {nom_affichage} - ERREUR: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🧪 Test d'Installation FruitVision")
    print("=" * 50)
    
    # Liste des modules à tester
    modules_requis = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow (PIL)"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-learn"),
        ("streamlit", "Streamlit"),
        ("tqdm", "tqdm"),
        ("kaggle", "Kaggle API")
    ]
    
    resultats = []
    
    # Tester chaque module
    for nom_module, nom_affichage in modules_requis:
        succes = tester_import(nom_module, nom_affichage)
        resultats.append(succes)
    
    print("\n" + "=" * 50)
    
    # Résumé
    nb_succes = sum(resultats)
    nb_total = len(resultats)
    
    if nb_succes == nb_total:
        print(f"🎉 SUCCÈS! Tous les modules ({nb_succes}/{nb_total}) sont installés!")
        print("✅ Vous pouvez maintenant exécuter le code FruitVision.")
    else:
        print(f"⚠️  {nb_succes}/{nb_total} modules installés correctement.")
        print("❌ Veuillez installer les modules manquants avant de continuer.")
        return False
    
    # Test spécifique TensorFlow
    print("\n🔬 Tests Spéciaux:")
    try:
        import tensorflow as tf
        print(f"📊 Version TensorFlow: {tf.__version__}")
        
        # Vérifier GPU (optionnel)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 GPU détecté: {len(gpus)} GPU(s) disponible(s)")
        else:
            print("💻 Pas de GPU détecté - utilisation du CPU (normal)")
            
    except Exception as e:
        print(f"❌ Erreur lors du test TensorFlow: {e}")
    
    print("\n✅ Test d'installation terminé!")
    return True

if __name__ == "__main__":
    main()