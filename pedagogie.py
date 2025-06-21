# Ce script Streamlit doit être exécuté dans un environnement local avec les bibliothèques nécessaires.
# Instructions :
# 1. Installez les dépendances : pip install streamlit pillow
# 2. Placez ce script avec les images suivantes dans le même dossier :
#    - apple_example.jpg, bloc1_output.jpg, bloc2_output.jpg, bloc3_output.jpg, bloc4_output.jpg, final_output.jpg
# 3. Exécutez avec la commande : streamlit run nom_du_fichier.py

import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="FruitVision V3 - Exploration pédagogique", layout="wide")

st.title("🍎 FruitVision V3 - Exploration pédagogique d'un CNN")

st.markdown("""
Bienvenue dans l'exploration interactive du modèle CNN utilisé dans FruitVision V3. Suivez une image (ex. une pomme) à travers chaque **bloc du réseau de neurones** pour comprendre ce qu’il apprend à chaque étape.
""")

image_path = "apple_example.jpg"
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption="Image d'entrée : une pomme", use_column_width=True)
else:
    st.warning(f"📷 Image '{image_path}' introuvable. Veuillez ajouter l'image au répertoire.")

st.markdown("---")

def afficher_bloc(titre, image_path, fonction, utilite, formule, code):
    with st.expander(titre):
        cols = st.columns([1, 2])

        with cols[0]:
            if os.path.exists(image_path):
                st.image(image_path, caption="Transformation visuelle", use_column_width=True)
            else:
                st.error(f"Image non trouvée : {image_path}")

        with cols[1]:
            st.markdown(f"""
            **🔍 Ce que cette couche fait :**  
            {fonction}

            **🎯 Utilité pédagogique :**  
            {utilite}

            **📐 Formule mathématique :**  
            `{formule}`

            **💻 Code Python :**  
            ```python
            {code}
            ```
            """)

# Définition des blocs pédagogiques
blocs = [
    ("Bloc 1 - Détection des bords", "bloc1_output.jpg", "Détecte des motifs simples comme les bords et textures.",
     "Apprentissage des formes basiques visuelles (traits, contours de la pomme).",
     "S(i,j) = ∑∑ I(i+m,j+n)·K(m,n)", "Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3))"),

    ("Bloc 2 - Formes intermédiaires", "bloc2_output.jpg", "Détecte des combinaisons de formes simples (structure de la pomme).",
     "Permet d'identifier des formes comme le rond de la pomme ou sa tige.",
     "S(i,j) = ∑∑ I(i+m,j+n)·K(m,n)", "Conv2D(64, (3, 3), activation='relu')"),

    ("Bloc 3 - Motifs complexes", "bloc3_output.jpg", "Détecte des caractéristiques plus abstraites (textures internes).",
     "Permet de différencier les pommes des tomates, par exemple.",
     "S(i,j) = ∑∑ I(i+m,j+n)·K(m,n)", "Conv2D(128, (3, 3), activation='relu')"),

    ("Bloc 4 - Résumé profond des caractéristiques", "bloc4_output.jpg", "Synthétise les motifs profonds pour la prise de décision.",
     "Prépare les données pour la couche finale de classification.",
     "S(i,j) = ∑∑ I(i+m,j+n)·K(m,n)", "Conv2D(256, (3, 3), activation='relu')"),

    ("Bloc Final - Prédiction", "final_output.jpg", "Transforme les activations en probabilités pour chaque fruit.",
     "Produit la prédiction finale : ici probablement 'Pomme'.",
     "softmax(zᵢ) = e^{zᵢ} / ∑e^{zⱼ}", "Dense(5, activation='softmax')")
]

# Affichage de chaque bloc
for bloc in blocs:
    afficher_bloc(*bloc)

st.success("🎉 Vous avez parcouru toutes les étapes du modèle CNN FruitVision V3 !")
