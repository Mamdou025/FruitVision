"""
Test de Validation Externe - Vérification sur Vraies Images
===========================================================

OBJECTIF: Tester le modèle "100%" sur images externes
pour voir s'il fonctionne vraiment

Mamadou Fall - Validation Critique
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

def tester_modele_sur_images_externes():
    """Tester le modèle sur des images externes au dataset."""
    
    print("🔍 TEST DE VALIDATION EXTERNE")
    print("=" * 50)
    print("🎯 Objectif: Vérifier si le 100% est réel")
    
    # Charger le modèle
    try:
        model = tf.keras.models.load_model('models/fruivision_split_manuel.h5')
        print("✅ Modèle chargé")
    except:
        print("❌ Modèle non trouvé - lancez d'abord test_split_manuel_corrige.py")
        return
    
    # Charger les classes
    try:
        with open('models/classes_split_manuel.json', 'r') as f:
            classes = json.load(f)
        print(f"✅ Classes: {classes}")
    except:
        classes = ['Pomme', 'Banane', 'Kiwi', 'Citron', 'Peche']
        print(f"⚠️ Classes par défaut: {classes}")
    
    print("\n🧪 TESTS DE VALIDATION:")
    
    # Test 1: Images du dossier Test/ (même dataset mais conditions différentes)
    print("\n1️⃣ TEST SUR DOSSIER Test/ (même dataset)")
    tester_dossier_test(model, classes)
    
    # Test 2: Images synthétiques simples
    print("\n2️⃣ TEST SUR IMAGES SYNTHÉTIQUES")
    tester_images_synthetiques(model, classes)
    
    # Test 3: Instructions pour images externes
    print("\n3️⃣ INSTRUCTIONS POUR IMAGES EXTERNES")
    donner_instructions_images_externes()

def tester_dossier_test(model, classes):
    """Tester sur le dossier Test/ du dataset Fruits-360."""
    
    test_dir = "data/fruits-360/Test"
    
    if not os.path.exists(test_dir):
        print("❌ Dossier Test/ non trouvé")
        return
    
    # Mapper nos classes aux dossiers Test
    mapping_test = {
        'Pomme': ['Apple Golden 1', 'Apple Red 1'],
        'Banane': ['Banana'],
        'Kiwi': ['Kiwi'],
        'Citron': ['Lemon'],
        'Peche': ['Peach']
    }
    
    print("📂 Test sur dossier Test/ (données jamais vues):")
    
    resultats = []
    
    for i, (classe_nom, dossiers_test) in enumerate(mapping_test.items()):
        print(f"   🍎 {classe_nom}:")
        
        for dossier in dossiers_test:
            chemin_dossier = os.path.join(test_dir, dossier)
            
            if os.path.exists(chemin_dossier):
                fichiers = [f for f in os.listdir(chemin_dossier) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Tester 5 images
                for j, fichier in enumerate(fichiers[:5]):
                    chemin_image = os.path.join(chemin_dossier, fichier)
                    
                    # Charger et préprocesser
                    image = Image.open(chemin_image).convert('RGB')
                    image = image.resize((100, 100), Image.Resampling.LANCZOS)
                    image_array = np.array(image, dtype=np.float32) / 255.0
                    image_array = np.expand_dims(image_array, axis=0)
                    
                    # Prédiction
                    pred = model.predict(image_array, verbose=0)
                    pred_idx = np.argmax(pred[0])
                    confiance = pred[0][pred_idx]
                    
                    # Résultat
                    correct = pred_idx == i
                    resultats.append(correct)
                    
                    print(f"      {fichier[:15]:15} → {classes[pred_idx]:8} ({confiance:.1%}) {'✅' if correct else '❌'}")
    
    # Statistiques
    if resultats:
        accuracy = sum(resultats) / len(resultats)
        print(f"\n   📊 Accuracy sur Test/: {accuracy:.1%} ({sum(resultats)}/{len(resultats)})")
        
        if accuracy < 0.8:
            print("   🚨 ALERTE: Performance chute sur données externes!")
        elif accuracy > 0.95:
            print("   ✅ Performance maintenue")
        else:
            print("   ⚠️ Performance dégradée mais acceptable")

def tester_images_synthetiques(model, classes):
    """Créer et tester des images synthétiques simples."""
    
    print("🎨 Création d'images test synthétiques:")
    
    # Couleurs caractéristiques
    couleurs_test = [
        ((255, 0, 0), "Rouge (Pomme)"),      # Rouge
        ((255, 255, 0), "Jaune (Banane)"),   # Jaune
        ((0, 255, 0), "Vert (Kiwi)"),        # Vert
        ((255, 255, 0), "Jaune (Citron)"),   # Jaune
        ((255, 150, 100), "Orange (Pêche)")  # Orange
    ]
    
    for couleur_rgb, description in couleurs_test:
        # Créer image unie
        image = Image.new('RGB', (100, 100), couleur_rgb)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Prédiction
        pred = model.predict(image_array, verbose=0)
        pred_idx = np.argmax(pred[0])
        confiance = pred[0][pred_idx]
        
        print(f"   {description:20} → {classes[pred_idx]:8} ({confiance:.1%})")

def donner_instructions_images_externes():
    """Instructions pour tester avec images externes."""
    
    print("📷 POUR TESTER AVEC VOS PROPRES IMAGES:")
    print("")
    print("1️⃣ Prenez des photos de fruits avec votre téléphone")
    print("2️⃣ Mettez-les dans un dossier 'test_externes/'")
    print("3️⃣ Utilisez ce code pour tester:")
    print("")
    print("```python")
    print("import tensorflow as tf")
    print("from PIL import Image")
    print("import numpy as np")
    print("")
    print("model = tf.keras.models.load_model('models/fruivision_split_manuel.h5')")
    print("classes = ['Pomme', 'Banane', 'Kiwi', 'Citron', 'Peche']")
    print("")
    print("# Charger votre image")
    print("image = Image.open('test_externes/votre_image.jpg').convert('RGB')")
    print("image = image.resize((100, 100))")
    print("image_array = np.array(image) / 255.0")
    print("image_array = np.expand_dims(image_array, axis=0)")
    print("")
    print("# Prédiction")
    print("pred = model.predict(image_array)")
    print("pred_idx = np.argmax(pred[0])")
    print("print(f'Prédiction: {classes[pred_idx]} ({pred[0][pred_idx]:.1%})')")
    print("```")
    print("")
    print("🎯 SI les prédictions sont MAUVAISES sur vos photos:")
    print("   → Le 100% était bien un FAUX positif!")
    print("   → Le modèle n'a appris que les patterns triviaux du dataset")

def analyser_facilite_dataset():
    """Analyser pourquoi le dataset est si facile."""
    
    print("\n🔍 ANALYSE: POURQUOI FRUITS-360 EST TROP FACILE")
    print("=" * 60)
    
    print("❌ CARACTÉRISTIQUES PROBLÉMATIQUES:")
    print("   📷 Fond blanc uniforme (pas réaliste)")
    print("   🎯 Fruits parfaitement centrés")
    print("   💡 Éclairage standardisé")
    print("   📐 Même angle/orientation")
    print("   🔍 Même résolution/qualité")
    
    print("\n💡 CONSÉQUENCES:")
    print("   🧠 Modèle apprend les patterns triviaux:")
    print("      - Position centrale = fruit")
    print("      - Fond blanc = contexte")
    print("      - Pas les vraies caractéristiques visuelles")
    
    print("\n🎯 POUR UN VRAI MODÈLE ROBUSTE:")
    print("   📸 Images avec fonds variés")
    print("   🔄 Angles et orientations multiples")
    print("   💡 Conditions d'éclairage variées")
    print("   📱 Photos réelles (téléphone, etc.)")

if __name__ == "__main__":
    print("🔍 VALIDATION EXTERNE - VÉRIFICATION DU '100%'")
    print("Est-ce que votre modèle fonctionne vraiment?")
    print("=" * 60)
    
    tester_modele_sur_images_externes()
    analyser_facilite_dataset()
    
    print(f"\n🎯 CONCLUSION:")
    print("Si le modèle échoue sur images externes,")
    print("alors le 100% était bien un artefact du dataset facile!")