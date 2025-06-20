"""
Diagnostic approfondi pour identifier la source du problème
"""

import tensorflow as tf
import json
import os
import numpy as np

print("🔬 DIAGNOSTIC APPROFONDI - Edufruits")
print("=" * 50)

# 1. VÉRIFIER LES RÉSULTATS D'ENTRAÎNEMENT
print("1️⃣ VÉRIFICATION DES RÉSULTATS D'ENTRAÎNEMENT")
try:
    with open('results/entrainement_1h_resultats.json', 'r') as f:
        resultats = json.load(f)
    
    print(f"✅ Fichier résultats trouvé")
    print(f"   📊 Accuracy finale: {resultats['accuracy_finale']}")
    print(f"   🔄 Époques: {resultats['epochs_completees']}")
    print(f"   📈 Images total: {resultats['nb_images_total']}")
    print(f"   🎯 Config utilisée: {type(resultats['config'])}")
    
    if 'config' in resultats:
        config = resultats['config']
        print(f"   📝 Classes dans config: {config.get('classes', 'Non trouvé')}")
        
except FileNotFoundError:
    print("❌ Fichier results/entrainement_1h_resultats.json non trouvé")
except Exception as e:
    print(f"⚠️ Erreur lecture résultats: {e}")

print()

# 2. EXAMINER L'ARCHITECTURE DU MODÈLE
print("2️⃣ ARCHITECTURE DU MODÈLE CHARGÉ")
try:
    model = tf.keras.models.load_model('models/fruivision_robuste_1h.h5')
    
    print(f"✅ Modèle chargé successfully")
    print(f"   📐 Input shape: {model.input_shape}")
    print(f"   📤 Output shape: {model.output_shape}")
    print(f"   🔢 Nombre de classes (output): {model.output_shape[-1]}")
    print(f"   🏗️ Nombre de couches: {len(model.layers)}")
    
    # Vérifier la dernière couche (Dense)
    derniere_couche = model.layers[-1]
    print(f"   🎯 Dernière couche: {type(derniere_couche).__name__}")
    print(f"   🔢 Unités dernière couche: {derniere_couche.units}")
    
    # Test avec prédiction simple
    test_input = np.random.random((1, 100, 100, 3)).astype(np.float32)
    prediction = model.predict(test_input, verbose=0)
    print(f"   📊 Shape prédiction: {prediction.shape}")
    print(f"   📈 Nombre de sorties: {prediction.shape[1]}")
    
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")

print()

# 3. VÉRIFIER LES CLASSES ORIGINALES
print("3️⃣ RECHERCHE DES CLASSES ORIGINALES")

# Chercher dans les fichiers de code d'entraînement
fichiers_a_verifier = [
    'entrainement_1h.py',
    'src/data_preprocessing.py', 
    'entrainement_optimise.py',
    'test_entrainement_rapide.py'
]

classes_trouvees = {}

for fichier in fichiers_a_verifier:
    if os.path.exists(fichier):
        try:
            with open(fichier, 'r', encoding='utf-8') as f:
                contenu = f.read()
                
            # Chercher les définitions de classes
            if 'classes_fruits' in contenu.lower():
                print(f"   📄 Classes trouvées dans {fichier}")
                
                # Extraire les lignes avec classes_fruits
                lignes = contenu.split('\n')
                for i, ligne in enumerate(lignes):
                    if 'classes_fruits' in ligne.lower() and '=' in ligne:
                        print(f"      Ligne {i+1}: {ligne.strip()}")
                        
                        # Essayer d'extraire quelques lignes suivantes
                        for j in range(1, 8):
                            if i+j < len(lignes):
                                ligne_suivante = lignes[i+j].strip()
                                if ligne_suivante and not ligne_suivante.startswith('#'):
                                    print(f"      Ligne {i+j+1}: {ligne_suivante}")
                                if '}' in ligne_suivante:
                                    break
                        break
                        
        except Exception as e:
            print(f"   ⚠️ Erreur lecture {fichier}: {e}")

print()

# 4. TESTER PREPROCESSING IDENTIQUE
print("4️⃣ TEST PREPROCESSING vs ENTRAÎNEMENT")

# Vérifier si le preprocessing est identique
try:
    # Import du preprocessing d'entraînement si possible
    import sys
    sys.path.append('src')
    
    try:
        from data_preprocessing import PreprocesseurDonnees
        print("   ✅ Module preprocessing importé")
        
        # Créer le même encodeur que l'entraînement
        classes_fruits_reelles = {
            'Pomme': ['Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Braeburn 1',
                     'Apple Granny Smith 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3'],
            'Banane': ['Banana 1', 'Banana 3', 'Banana 4', 'Banana Red 1'],
            'Kiwi': ['Kiwi 1'],
            'Citron': ['Lemon 1', 'Lemon Meyer 1'],
            'Peche': ['Peach 1', 'Peach 2']
        }
        
        preprocesseur = PreprocesseurDonnees(classes_fruits=classes_fruits_reelles)
        
        print(f"   🏷️ Classes dans le preprocesseur:")
        for i, nom in enumerate(preprocesseur.noms_classes):
            print(f"      {i}: {nom}")
            
    except ImportError as e:
        print(f"   ❌ Import preprocessing failed: {e}")
        
        # Ordre alphabétique manuel
        fruits_ordre_alpha = sorted(['Pomme', 'Banane', 'Kiwi', 'Citron', 'Peche'])
        print(f"   📝 Ordre alphabétique théorique:")
        for i, fruit in enumerate(fruits_ordre_alpha):
            print(f"      {i}: {fruit}")
            
except Exception as e:
    print(f"   ⚠️ Erreur test preprocessing: {e}")

print()

# 5. CONCLUSION
print("5️⃣ CONCLUSION DIAGNOSTIC")
print("   🎯 Points à vérifier:")
print("   1. Le modèle a-t-il vraiment 5 classes en sortie?")
print("   2. L'ordre des classes correspond-il au LabelEncoder?") 
print("   3. Y a-t-il eu une erreur pendant l'entraînement?")
print("   4. Le preprocessing est-il identique?")

print(f"\n🔧 PROCHAINE ÉTAPE:")
print(f"   Comparez les résultats ci-dessus avec votre code d'entraînement")