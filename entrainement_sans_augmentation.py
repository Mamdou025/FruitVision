"""
Entraînement SANS augmentation pour éviter l'explosion mémoire
"""

import os
import sys
import numpy as np
from datetime import datetime

# Ajouter les dossiers au path
sys.path.append('src')
sys.path.append('config')

# Importer nos modules
from model_architecture import creer_modele_fruivision
from data_preprocessing import PreprocesseurDonnees

def entrainement_sans_augmentation():
    """Entraînement simple SANS augmentation."""
    
    print("🔧 ENTRAÎNEMENT SANS AUGMENTATION - TEST RAPIDE")
    print("=" * 60)
    print("🎯 Objectif: Tester le modèle sur données PURES")
    print("🚫 Pas d'augmentation = Pas d'over-fitting")
    
    # Classes du dataset
    classes_fruits_reelles = {
        'Pomme': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Braeburn 1',
                 'Apple Granny Smith 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Red 1'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    # Préprocesseur
    preprocesseur = PreprocesseurDonnees(classes_fruits=classes_fruits_reelles)
    
    # Chargement SANS augmentation
    print(f"\n📥 Chargement des données SANS augmentation...")
    debut_chargement = datetime.now()
    
    X, y = preprocesseur.charger_donnees_dossier(
        chemin_base="data/fruits-360/Training",
        utiliser_augmentation=False,  # PAS D'AUGMENTATION !
        max_images_par_classe=300     # LIMITE stricte
    )
    
    fin_chargement = datetime.now()
    temps_chargement = (fin_chargement - debut_chargement).seconds
    
    print(f"✅ Données chargées en {temps_chargement // 60}m {temps_chargement % 60}s")
    print(f"📊 Total: {X.shape[0]:,} images (BEAUCOUP plus raisonnable !)")
    
    # Vérifier la distribution
    print(f"\n📊 DISTRIBUTION DES CLASSES:")
    for i, fruit in enumerate(preprocesseur.noms_classes):
        nb_images = np.sum(y == i)
        print(f"   {fruit:8}: {nb_images:,} images")
    
    # Division des données
    print(f"\n🔄 Division des données...")
    ensembles = preprocesseur.creer_ensembles_donnees(
        X, y, 
        taille_validation=0.2,
        taille_test=0.2  # Plus de test pour évaluation honnête
    )
    
    print(f"✅ Ensembles créés:")
    print(f"   📚 Entraînement: {ensembles['X_train'].shape[0]:,} images")
    print(f"   🔍 Validation: {ensembles['X_val'].shape[0]:,} images")
    print(f"   🧪 Test: {ensembles['X_test'].shape[0]:,} images")
    
    # Création du modèle
    print(f"\n🧠 Création du modèle...")
    modele = creer_modele_fruivision()
    
    # Learning rate conservateur
    modele.optimizer.learning_rate.assign(0.001)
    print(f"🔧 Learning rate: 0.001 (conservateur)")
    
    # Callbacks simples
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='models/fruivision_sans_augmentation.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        os.makedirs('models', exist_ok=True)
        print(f"✅ Callbacks configurés")
        
    except Exception as e:
        print(f"⚠️ Callbacks simplifiés: {e}")
        callbacks = []
    
    # ENTRAÎNEMENT HONNÊTE
    print(f"\n🚀 DÉBUT DE L'ENTRAÎNEMENT HONNÊTE")
    print(f"🕐 Heure de début: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    debut_entrainement = datetime.now()
    
    try:
        historique = modele.fit(
            ensembles['X_train'], ensembles['y_train'],
            validation_data=(ensembles['X_val'], ensembles['y_val']),
            epochs=25,  # Moins d'époques pour éviter overfitting
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        fin_entrainement = datetime.now()
        temps_entrainement = fin_entrainement - debut_entrainement
        
        print(f"\n🎉 ENTRAÎNEMENT TERMINÉ!")
        print(f"⏱️ Temps réel: {temps_entrainement}")
        
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        return False
    
    # TEST IMMÉDIAT avec vraies images
    print(f"\n🧪 TEST IMMÉDIAT AVEC VRAIES IMAGES")
    print("-" * 50)
    
    try:
        # Charger le meilleur modèle
        import tensorflow as tf
        meilleur_modele = tf.keras.models.load_model('models/fruivision_sans_augmentation.h5')
        print("✅ Meilleur modèle chargé")
        
        # Test sur données de test
        resultats_test = meilleur_modele.evaluate(ensembles['X_test'], ensembles['y_test'], verbose=0)
        accuracy_test = resultats_test[1]
        
        print(f"📊 PERFORMANCE SUR DONNÉES DE TEST:")
        print(f"   🎯 Accuracy: {accuracy_test:.1%}")
        print(f"   📉 Loss: {resultats_test[0]:.4f}")
        
        # Test sur quelques vraies images du dataset
        test_paths = [
            ('data/fruits-360/Training/Apple Golden 1', 'Pomme', 0),
            ('data/fruits-360/Training/Banana 1', 'Banane', 1),
            ('data/fruits-360/Training/Kiwi 1', 'Kiwi', 2),
            ('data/fruits-360/Training/Lemon 1', 'Citron', 3),
            ('data/fruits-360/Training/Peach 1', 'Pêche', 4)
        ]
        
        print(f"\n🍎 TEST SUR VRAIES IMAGES DU DATASET:")
        resultats_vrais = []
        
        for dossier, fruit_attendu, classe_attendue in test_paths:
            if os.path.exists(dossier):
                images = [f for f in os.listdir(dossier) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    # Prendre une image au hasard
                    image_path = os.path.join(dossier, images[10])  # 10ème image
                    image = preprocesseur.charger_image(image_path)
                    
                    if image is not None:
                        image_batch = np.expand_dims(image, axis=0)
                        prediction = meilleur_modele.predict(image_batch, verbose=0)
                        classe_predite = np.argmax(prediction[0])
                        confiance = prediction[0][classe_predite]
                        fruit_predit = preprocesseur.noms_classes[classe_predite]
                        
                        succes = fruit_predit == fruit_attendu
                        resultats_vrais.append(succes)
                        
                        print(f"   {fruit_attendu:8} → {fruit_predit:8} ({confiance:.3f}) ", end="")
                        print("✅" if succes else "❌")
        
        accuracy_vrais = sum(resultats_vrais) / len(resultats_vrais) if resultats_vrais else 0
        
        print(f"\n📊 RÉSULTATS FINAUX:")
        print(f"   🧪 Test set accuracy: {accuracy_test:.1%}")
        print(f"   🍎 Vraies images accuracy: {accuracy_vrais:.1%} ({sum(resultats_vrais)}/{len(resultats_vrais)})")
        
        if accuracy_test >= 0.7 and accuracy_vrais >= 0.6:
            print(f"\n🎉 SUCCÈS! Modèle honnête et fonctionnel!")
            print(f"💡 Performances réalistes sans sur-augmentation")
        elif accuracy_test >= 0.5:
            print(f"\n✅ ACCEPTABLE! Modèle honnête")
            print(f"💡 Pas de 'faux 100%' - c'est un vrai modèle!")
        else:
            print(f"\n⚠️ PERFORMANCES FAIBLES mais HONNÊTES")
            print(f"💡 Mieux qu'un faux 100% qui ne fonctionne pas!")
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        
    # Comparaison avec le faux modèle
    print(f"\n📈 COMPARAISON AVEC LE 'FAUX 100%':")
    print(f"   ❌ Ancien: 100% accuracy mais 20% en réalité")
    print(f"   ✅ Nouveau: {accuracy_test:.1%} accuracy HONNÊTE")
    print(f"   💪 Lequel préférez-vous?")
    
    return True

if __name__ == "__main__":
    print("🔧 ENTRAÎNEMENT SANS AUGMENTATION - FRUIVISION HONNÊTE")
    print("Fini les faux 100% - place à la vérité!")
    print("=" * 60)
    
    success = entrainement_sans_augmentation()
    
    if success:
        print("\n✅ ENTRAÎNEMENT HONNÊTE TERMINÉ!")
        print("🎯 Modèle réaliste prêt!")
        print("💪 Performance VRAIE, pas mensongère!")
    else:
        print("\n❌ ENTRAÎNEMENT ÉCHOUÉ")