"""
Test Rapide Edufruits - Version Corrigée
==========================================
Version simplifiée qui évite l'erreur top_2_accuracy
"""

import os
import sys
import numpy as np
from datetime import datetime

# Ajouter le dossier src au path
sys.path.append('src')

# Importer nos modules
from data_preprocessing import PreprocesseurDonnees

def test_rapide_corrige():
    print("🧪 Test d'Entraînement Rapide Edufruits - Version Corrigée")
    print("=" * 60)
    
    # Classes réelles du dataset
    classes_fruits_reelles = {
        'Pomme': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Braeburn 1',
                 'Apple Granny Smith 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Red 1'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    # Chargement des données
    preprocesseur = PreprocesseurDonnees(classes_fruits=classes_fruits_reelles)
    
    print("📥 Chargement des données...")
    X, y = preprocesseur.charger_donnees_dossier(
        "data/fruits-360/Training",
        utiliser_augmentation=False,
        max_images_par_classe=50
    )
    
    # Division des données
    ensembles = preprocesseur.creer_ensembles_donnees(X, y)
    
    # Création du modèle avec compilation manuelle
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    
    print("🧠 Création du modèle...")
    modele = models.Sequential([
        layers.Input(shape=(100, 100, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.375),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    
    # Compilation avec seulement accuracy
    modele.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']  # Seulement accuracy
    )
    
    print("✅ Modèle créé et compilé")
    
    # Entraînement
    print("🚀 Début de l'entraînement (5 époques)...")
    debut = datetime.now()
    
    historique = modele.fit(
        ensembles['X_train'], ensembles['y_train'],
        validation_data=(ensembles['X_val'], ensembles['y_val']),
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
    fin = datetime.now()
    temps_total = (fin - debut).seconds
    
    # Évaluation
    print("\n📊 Évaluation finale...")
    resultats = modele.evaluate(ensembles['X_test'], ensembles['y_test'], verbose=0)
    accuracy_finale = resultats[1]
    
    print(f"\n🎯 RÉSULTATS:")
    print(f"   📈 Précision finale: {accuracy_finale:.4f} ({accuracy_finale*100:.2f}%)")
    print(f"   ⏱️ Temps d'entraînement: {temps_total // 60}m {temps_total % 60}s")
    
    # Interprétation
    print(f"\n🤔 INTERPRÉTATION:")
    if accuracy_finale > 0.80:
        print(f"   🎉 EXCELLENT! >80% avec seulement 5 époques")
        print(f"   💡 L'entraînement complet (50 époques) sera fantastique!")
    elif accuracy_finale > 0.60:
        print(f"   ✅ BON! >60% pour un test rapide")
        print(f"   💡 C'est normal avec peu d'époques, continuez!")
    else:
        print(f"   ⚠️ Moyen, mais c'est un début avec seulement 5 époques")
    
    # Test de prédictions
    print(f"\n🔮 TEST DE PRÉDICTIONS:")
    for i in range(3):
        idx = np.random.randint(0, len(ensembles['X_test']))
        image_test = ensembles['X_test'][idx:idx+1]
        vraie_classe = np.argmax(ensembles['y_test'][idx])
        
        prediction = modele.predict(image_test, verbose=0)
        classe_predite = np.argmax(prediction[0])
        confiance = prediction[0][classe_predite]
        
        nom_vrai = preprocesseur.noms_classes[vraie_classe]
        nom_predit = preprocesseur.noms_classes[classe_predite]
        
        statut = "✅" if vraie_classe == classe_predite else "❌"
        print(f"   {statut} Vrai: {nom_vrai} | Prédit: {nom_predit} | Confiance: {confiance:.3f}")
    
    # Sauvegarde
    os.makedirs("results", exist_ok=True)
    modele.save('results/modele_test_rapide.h5')
    print(f"\n💾 Modèle sauvegardé dans results/modele_test_rapide.h5")
    
    print(f"\n🎉 Test rapide terminé avec succès!")
    return True

if __name__ == "__main__":
    test_rapide_corrige()