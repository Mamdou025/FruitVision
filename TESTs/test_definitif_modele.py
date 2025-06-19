"""
Tests définitifs pour déterminer si le problème vient du modèle ou du preprocessing
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# Ajouter le path pour importer les modules d'entraînement
sys.path.append('src')

print("🔬 TESTS DÉFINITIFS - MODÈLE vs PREPROCESSING")
print("=" * 60)

# 1. CHARGER LE MODÈLE
model = tf.keras.models.load_model('models/fruivision_robuste_1h.h5')

# 2. TESTER AVEC LE PREPROCESSING EXACT DE L'ENTRAÎNEMENT
print("1️⃣ TEST AVEC PREPROCESSING D'ENTRAÎNEMENT")
print("-" * 50)

try:
    from data_preprocessing import PreprocesseurDonnees
    
    # Créer le même preprocesseur que l'entraînement
    classes_fruits_reelles = {
        'Pomme': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Braeburn 1',
                 'Apple Granny Smith 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Red 1'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    preprocesseur = PreprocesseurDonnees(classes_fruits=classes_fruits_reelles)
    
    print("✅ Preprocesseur d'entraînement importé")
    print(f"   Classes: {list(preprocesseur.noms_classes)}")
    print(f"   Mapping: {dict(enumerate(preprocesseur.noms_classes))}")
    
    # Test avec une vraie image du dataset d'entraînement
    if os.path.exists('data/fruits-360/Training'):
        print("\n🧪 TEST AVEC VRAIES IMAGES DU DATASET D'ENTRAÎNEMENT")
        
        # Tester une image de chaque classe
        test_paths = [
            ('data/fruits-360/Training/Apple Golden 1', 'Pomme'),
            ('data/fruits-360/Training/Banana 1', 'Banane'),  
            ('data/fruits-360/Training/Kiwi 1', 'Kiwi'),
            ('data/fruits-360/Training/Lemon 1', 'Citron'),
            ('data/fruits-360/Training/Peach 1', 'Pêche')
        ]
        
        for dossier, fruit_attendu in test_paths:
            if os.path.exists(dossier):
                # Prendre la première image
                images = [f for f in os.listdir(dossier) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    image_path = os.path.join(dossier, images[0])
                    
                    # Charger avec le preprocessing d'entraînement
                    image_orig = preprocesseur.charger_image(image_path)
                    
                    if image_orig is not None:
                        # Prédiction avec preprocessing d'entraînement
                        image_batch = np.expand_dims(image_orig, axis=0)
                        prediction = model.predict(image_batch, verbose=0)
                        classe_predite = np.argmax(prediction[0])
                        confiance = prediction[0][classe_predite]
                        fruit_predit = preprocesseur.noms_classes[classe_predite]
                        
                        print(f"   {fruit_attendu:8} → {fruit_predit:8} ({confiance:.3f}) ", end="")
                        if fruit_predit == fruit_attendu:
                            print("✅")
                        else:
                            print("❌ ÉCHEC!")
                        
                        print(f"   {'':8}   Probas: {[f'{p:.3f}' for p in prediction[0]]}")
                    else:
                        print(f"   {fruit_attendu:8} → Erreur de chargement")
                else:
                    print(f"   {fruit_attendu:8} → Pas d'images trouvées")
            else:
                print(f"   {fruit_attendu:8} → Dossier non trouvé")
    else:
        print("❌ Dataset d'entraînement non trouvé")
        
except ImportError:
    print("❌ Impossible d'importer le preprocesseur d'entraînement")

print("\n" + "="*60)

# 3. COMPARER PREPROCESSING STREAMLIT vs ENTRAÎNEMENT
print("2️⃣ COMPARAISON PREPROCESSING STREAMLIT vs ENTRAÎNEMENT")
print("-" * 50)

def preprocessing_streamlit(image_path):
    """Preprocessing tel qu'utilisé dans Streamlit."""
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((100, 100), Image.Resampling.LANCZOS)
    image_array = np.array(image_resized, dtype=np.float32)
    image_array = image_array / 255.0
    return image_array

def preprocessing_entrainement(image_path):
    """Preprocessing tel qu'utilisé pendant l'entraînement."""
    try:
        from data_preprocessing import PreprocesseurDonnees
        preprocesseur = PreprocesseurDonnees()
        return preprocesseur.charger_image(image_path)
    except:
        # Fallback si import impossible
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((100, 100), Image.Resampling.LANCZOS)
        image_array = np.array(image_resized, dtype=np.float32)
        image_array = image_array / 255.0
        return image_array

# Créer une image test simple
print("🎨 CRÉATION D'IMAGES TEST SIMPLES")

# Image de banane (jaune)
banane_test = Image.new('RGB', (200, 300), (255, 255, 0))  # Jaune
banane_test.save('test_banane.jpg')

# Image de kiwi (vert)  
kiwi_test = Image.new('RGB', (150, 150), (100, 200, 100))  # Vert
kiwi_test.save('test_kiwi.jpg')

# Image de pomme (rouge)
pomme_test = Image.new('RGB', (180, 180), (255, 50, 50))  # Rouge
pomme_test.save('test_pomme.jpg')

# Tester les deux preprocessing
images_test = [
    ('test_banane.jpg', 'Banane', 1),
    ('test_kiwi.jpg', 'Kiwi', 2), 
    ('test_pomme.jpg', 'Pomme', 0)
]

print("\n📊 RÉSULTATS COMPARATIFS:")
print("Image        | Streamlit     | Entraînement  | Attendu")
print("-" * 55)

for image_path, nom, classe_attendue in images_test:
    # Preprocessing Streamlit
    img_streamlit = preprocessing_streamlit(image_path)
    img_streamlit_batch = np.expand_dims(img_streamlit, axis=0)
    pred_streamlit = model.predict(img_streamlit_batch, verbose=0)
    classe_streamlit = np.argmax(pred_streamlit[0])
    conf_streamlit = pred_streamlit[0][classe_streamlit]
    
    # Preprocessing Entraînement
    img_entrainement = preprocessing_entrainement(image_path)
    img_entrainement_batch = np.expand_dims(img_entrainement, axis=0)
    pred_entrainement = model.predict(img_entrainement_batch, verbose=0)
    classe_entrainement = np.argmax(pred_entrainement[0])
    conf_entrainement = pred_entrainement[0][classe_entrainement]
    
    classes = ["Pomme", "Banane", "Kiwi", "Citron", "Pêche"]
    
    print(f"{nom:12} | {classes[classe_streamlit]:8} {conf_streamlit:.2f} | {classes[classe_entrainement]:8} {conf_entrainement:.2f} | {classes[classe_attendue]}")
    
    # Vérifier si différence entre les deux
    if classe_streamlit != classe_entrainement:
        print(f"{'':12} | ⚠️ DIFFÉRENCE DÉTECTÉE!")
        
        # Analyser les différences pixel par pixel
        diff = np.abs(img_streamlit - img_entrainement)
        print(f"{'':12} | Diff max: {diff.max():.6f}, Diff moyenne: {diff.mean():.6f}")

print("\n" + "="*60)

# 4. DIAGNOSTIC FINAL
print("3️⃣ DIAGNOSTIC FINAL")
print("-" * 30)

print("🔍 QUE NOUS RÉVÈLENT CES TESTS:")
print()
print("A. Si STREAMLIT = ENTRAÎNEMENT et TOUS LES DEUX ÉCHOUENT:")
print("   → ❌ MODÈLE DÉFAILLANT (sur-entraîné, données biaisées)")
print()
print("B. Si STREAMLIT ≠ ENTRAÎNEMENT:")
print("   → ❌ PROBLÈME DE PREPROCESSING dans Streamlit")
print()
print("C. Si ENTRAÎNEMENT MARCHE mais STREAMLIT NON:")
print("   → ❌ BUG dans le preprocessing Streamlit")
print()
print("D. Si TOUS LES DEUX MARCHENT avec images simples:")
print("   → ✅ MODÈLE OK, problème avec VOS images spécifiques")

# Nettoyer les fichiers test
try:
    os.remove('test_banane.jpg')
    os.remove('test_kiwi.jpg') 
    os.remove('test_pomme.jpg')
except:
    pass

print(f"\n🎯 VERDICT:")
print("Exécutez ce script et analysez les résultats.")
print("Si même les images simples échouent → MODÈLE DÉFAILLANT")
print("Si seules vos images échouent → MODÈLE OK mais trop spécialisé")