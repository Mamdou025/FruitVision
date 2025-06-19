"""
Debug pur Python - Pourquoi toutes les prédictions donnent "Pomme" ?
"""

import tensorflow as tf
import numpy as np
from PIL import Image

print("🚨 DEBUG URGENT - Prédictions toujours Pomme")
print("=" * 50)

# 1. CHARGER LE MODÈLE
try:
    model = tf.keras.models.load_model('models/fruivision_robuste_1h.h5')
    print("✅ Modèle chargé avec succès")
    print(f"   📐 Input shape: {model.input_shape}")
    print(f"   📤 Output shape: {model.output_shape}")
    print(f"   🔢 Nombre de classes: {model.output_shape[-1]}")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    exit()

print()

# 2. TEST AVEC DONNÉES ALÉATOIRES
print("1️⃣ TEST AVEC DONNÉES ALÉATOIRES")
print("-" * 30)

classes_predites = []
for i in range(10):
    random_data = np.random.random((1, 100, 100, 3)).astype(np.float32)
    prediction = model.predict(random_data, verbose=0)
    classe_predite = np.argmax(prediction[0])
    classes_predites.append(classe_predite)
    
    print(f"Test {i+1:2d}: Classe {classe_predite} | Probas: {[f'{p:.3f}' for p in prediction[0]]}")

print(f"\nClasses uniques prédites: {sorted(set(classes_predites))}")
print(f"Répartition: {[classes_predites.count(i) for i in range(5)]}")

if len(set(classes_predites)) == 1:
    print("🚨 PROBLÈME MAJEUR: Le modèle prédit TOUJOURS la même classe!")
elif len(set(classes_predites)) < 3:
    print("⚠️ PROBLÈME: Le modèle utilise très peu de classes")
else:
    print("✅ Le modèle varie ses prédictions")

print()

# 3. TEST AVEC COULEURS PURES
print("2️⃣ TEST AVEC COULEURS PURES")
print("-" * 30)

couleurs = [
    ((255, 0, 0), "Rouge"),
    ((0, 255, 0), "Vert"), 
    ((0, 0, 255), "Bleu"),
    ((255, 255, 0), "Jaune"),
    ((255, 255, 255), "Blanc"),
    ((0, 0, 0), "Noir")
]

for couleur_rgb, nom in couleurs:
    # Créer image 100x100 de couleur pure
    image = Image.new('RGB', (100, 100), couleur_rgb)
    
    # Même preprocessing que l'app
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = model.predict(image_array, verbose=0)
    classe_predite = np.argmax(prediction[0])
    confiance = prediction[0][classe_predite]
    
    print(f"{nom:10} → Classe {classe_predite} ({confiance:.3f}) | Probas: {[f'{p:.3f}' for p in prediction[0]]}")

print()

# 4. TEST AVEC PATTERNS GÉOMÉTRIQUES
print("3️⃣ TEST AVEC PATTERNS GÉOMÉTRIQUES")
print("-" * 30)

# Créer des patterns différents
patterns = []

# Pattern 1: Damier
damier = np.zeros((100, 100, 3))
for i in range(0, 100, 10):
    for j in range(0, 100, 10):
        if (i//10 + j//10) % 2 == 0:
            damier[i:i+10, j:j+10] = [1, 1, 1]
patterns.append((damier, "Damier"))

# Pattern 2: Lignes verticales
lignes_v = np.zeros((100, 100, 3))
lignes_v[:, ::10] = [1, 0, 0]
patterns.append((lignes_v, "Lignes_V"))

# Pattern 3: Cercle
cercle = np.zeros((100, 100, 3))
center = (50, 50)
for i in range(100):
    for j in range(100):
        if (i-center[0])**2 + (j-center[1])**2 < 30**2:
            cercle[i, j] = [0, 1, 0]
patterns.append((cercle, "Cercle"))

for pattern_array, nom in patterns:
    pattern_array = np.expand_dims(pattern_array.astype(np.float32), axis=0)
    prediction = model.predict(pattern_array, verbose=0)
    classe_predite = np.argmax(prediction[0])
    confiance = prediction[0][classe_predite]
    
    print(f"{nom:12} → Classe {classe_predite} ({confiance:.3f}) | Probas: {[f'{p:.3f}' for p in prediction[0]]}")

print()

# 5. ANALYSE DES POIDS DU MODÈLE
print("4️⃣ ANALYSE RAPIDE DU MODÈLE")
print("-" * 30)

# Vérifier la dernière couche
derniere_couche = model.layers[-1]
if hasattr(derniere_couche, 'get_weights'):
    poids = derniere_couche.get_weights()
    if len(poids) > 1:  # Weights + bias
        bias = poids[1]
        print(f"Bias de la couche finale: {bias}")
        print(f"Classe avec le plus gros bias: {np.argmax(bias)} (valeur: {np.max(bias):.3f})")
        
        if np.max(bias) > np.median(bias) + 2 * np.std(bias):
            print("⚠️ PROBLÈME: Un bias est anormalement élevé!")
        else:
            print("✅ Les bias semblent équilibrés")

print()

# 6. RECOMMANDATIONS
print("5️⃣ DIAGNOSTIC ET RECOMMANDATIONS")
print("-" * 30)

toutes_predictions = []
for i in range(50):
    random_data = np.random.random((1, 100, 100, 3)).astype(np.float32)
    pred = model.predict(random_data, verbose=0)
    toutes_predictions.append(np.argmax(pred[0]))

distribution = [toutes_predictions.count(i) for i in range(5)]
classes_actives = [i for i, count in enumerate(distribution) if count > 0]

print(f"Distribution sur 50 tests aléatoires: {distribution}")
print(f"Classes actives: {classes_actives}")
print(f"Classes jamais prédites: {[i for i in range(5) if i not in classes_actives]}")

if len(classes_actives) == 1:
    classe_dominante = classes_actives[0]
    print(f"\n🚨 PROBLÈME CONFIRMÉ: Modèle bloqué sur classe {classe_dominante}")
    print("   Causes possibles:")
    print("   1. Modèle sur-entraîné (early stopping trop tard)")
    print("   2. Learning rate trop élevé → convergence vers un minimum local")
    print("   3. Données d'entraînement déséquilibrées")
    print("   4. Architecture trop simple → mode mémorisation")
    
    print(f"\n💡 SOLUTIONS:")
    print("   1. Utiliser un modèle d'une époque antérieure")
    print("   2. Ré-entraîner avec learning rate plus bas")
    print("   3. Ajouter de la régularisation (dropout)")
    print("   4. Vérifier l'équilibre des données d'entraînement")
    
elif len(classes_actives) <= 2:
    print(f"\n⚠️ PROBLÈME PARTIEL: Modèle utilise seulement {len(classes_actives)} classes")
    print("   Le modèle a probablement convergé trop agressivement")
    
else:
    print(f"\n✅ MODÈLE SEMBLE FONCTIONNEL: {len(classes_actives)} classes actives")
    print("   Le problème vient probablement de l'interface Streamlit")

print(f"\n🎯 PROCHAINES ÉTAPES:")
print("   1. Si modèle bloqué → utiliser un checkpoint antérieur")
print("   2. Si modèle OK → debug du preprocessing dans Streamlit")
print("   3. Comparer avec le preprocessing exact de l'entraînement")