#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Streamlit pour EduFruis
Démonstration visuelle du modèle sur le dataset Fruits-360
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import random
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="🍎 EduFruis - Démonstration",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look moderne
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .warning-metric {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    .prediction-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Charger le modèle avec cache"""
    try:
        model = tf.keras.models.load_model('models/fruivision_split_manuel.h5')
        return model, ['Pomme', 'Banane', 'Kiwi', 'Citron', 'Peche']
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None, None

def preprocess_image(image):
    """Préprocesser l'image pour le modèle"""
    try:
        # Redimensionner à 100x100
        image = image.resize((100, 100))
        # Convertir en array et normaliser
        image_array = np.array(image) / 255.0
        # Ajouter dimension batch
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.error(f"❌ Erreur de préprocessing: {e}")
        return None

def get_sample_images():
    """Récupérer des images d'exemple du dataset"""
    sample_images = {}
    base_dir = "data/fruits-360/test"
    
    # Mapping des dossiers du dataset vers nos classes
    folder_mappings = {
        'Pomme': ['Apple Red 1', 'Apple Red 2', 'Apple Golden 1'],
        'Banane': ['Banana 1', 'Banana 3', 'Banana 4'],
        'Kiwi': ['Kiwi 1'],
        'Citron': ['Lemon 1', 'Lemon Meyer 1'],
        'Peche': ['Peach 1', 'Peach 2']
    }
    
    for class_name, folders in folder_mappings.items():
        sample_images[class_name] = []
        for folder in folders:
            folder_path = os.path.join(base_dir, folder)
            if os.path.exists(folder_path):
                images = glob.glob(os.path.join(folder_path, "*.jpg"))
                sample_images[class_name].extend(images[:10])  # Max 10 par dossier
                break  # Prendre le premier dossier disponible
    
    return sample_images

def predict_image(model, classes, image):
    """Faire une prédiction sur une image"""
    try:
        image_array = preprocess_image(image)
        if image_array is None:
            return None, None, None
        
        # Prédiction
        predictions = model.predict(image_array, verbose=0)
        
        # Résultats
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = classes[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Toutes les probabilités
        all_probs = {classes[i]: float(predictions[0][i]) for i in range(len(classes))}
        
        return predicted_class, confidence, all_probs
        
    except Exception as e:
        st.error(f"❌ Erreur de prédiction: {e}")
        return None, None, None

def create_confidence_chart(probabilities):
    """Créer un graphique des probabilités"""
    df = pd.DataFrame(list(probabilities.items()), columns=['Fruit', 'Probabilité'])
    df['Probabilité'] = df['Probabilité'] * 100  # Convertir en pourcentage
    
    fig = px.bar(
        df, 
        x='Fruit', 
        y='Probabilité',
        color='Probabilité',
        color_continuous_scale='Viridis',
        title="Distribution des Probabilités de Prédiction"
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Classes de Fruits",
        yaxis_title="Probabilité (%)"
    )
    
    return fig

def main():
    """Interface principale Streamlit"""
    
    # Header principal
    st.markdown('<h1 class="main-header">🍎 EduFruis - Démonstration Interactive</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Contrôles")
    st.sidebar.markdown("---")
    
    # Charger le modèle
    model, classes = load_model()
    
    if model is None:
        st.error("❌ Impossible de charger le modèle. Vérifiez le chemin 'models/fruivision_split_manuel.h5'")
        st.stop()
    
    st.sidebar.success("✅ Modèle chargé avec succès!")
    st.sidebar.write(f"**Classes:** {', '.join(classes)}")
    
    # Options de test
    st.sidebar.markdown("### 📋 Options de Test")
    test_mode = st.sidebar.selectbox(
        "Mode de test:",
        ["📁 Images du Dataset", "📤 Upload Personnel", "🎲 Test Aléatoire"]
    )
    
    # Colonnes principales
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🖼️ Zone de Test")
        
        if test_mode == "📁 Images du Dataset":
            st.markdown("### Testez avec les images du dataset Fruits-360")
            
            # Sélection de classe
            selected_class = st.selectbox("Choisissez une classe:", classes)
            
            # Récupérer les images d'exemple
            sample_images = get_sample_images()
            
            if selected_class in sample_images and sample_images[selected_class]:
                
                # Sélection d'image
                available_images = sample_images[selected_class]
                selected_image_path = st.selectbox(
                    "Choisissez une image:",
                    available_images,
                    format_func=lambda x: os.path.basename(x)
                )
                
                if selected_image_path:
                    # Afficher l'image
                    image = Image.open(selected_image_path).convert('RGB')
                    st.image(image, caption=f"Image sélectionnée: {os.path.basename(selected_image_path)}", width=300)
                    
                    # Bouton de prédiction
                    if st.button("🔮 Prédire", type="primary"):
                        with st.spinner("Analyse en cours..."):
                            predicted_class, confidence, all_probs = predict_image(model, classes, image)
                            
                            if predicted_class:
                                # Résultat de prédiction
                                is_correct = predicted_class == selected_class
                                status_icon = "✅" if is_correct else "❌"
                                
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <h3>{status_icon} Résultat de la Prédiction</h3>
                                    <p><strong>Prédiction:</strong> {predicted_class}</p>
                                    <p><strong>Confiance:</strong> {confidence:.1%}</p>
                                    <p><strong>Classe attendue:</strong> {selected_class}</p>
                                    <p><strong>Statut:</strong> {'✅ CORRECT' if is_correct else '❌ INCORRECT'}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Graphique des probabilités
                                fig = create_confidence_chart(all_probs)
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"❌ Aucune image disponible pour la classe {selected_class}")
        
        elif test_mode == "📤 Upload Personnel":
            st.markdown("### Uploadez votre propre image")
            
            uploaded_file = st.file_uploader(
                "Choisissez une image...",
                type=['jpg', 'jpeg', 'png'],
                help="Uploadez une image de fruit pour tester le modèle"
            )
            
            if uploaded_file is not None:
                # Afficher l'image uploadée
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Image uploadée", width=300)
                
                # Prédiction automatique
                with st.spinner("Analyse en cours..."):
                    predicted_class, confidence, all_probs = predict_image(model, classes, image)
                    
                    if predicted_class:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🔮 Résultat de la Prédiction</h3>
                            <p><strong>Prédiction:</strong> {predicted_class}</p>
                            <p><strong>Confiance:</strong> {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Graphique des probabilités
                        fig = create_confidence_chart(all_probs)
                        st.plotly_chart(fig, use_container_width=True)
        
        elif test_mode == "🎲 Test Aléatoire":
            st.markdown("### Test automatique sur images aléatoires")
            
            if st.button("🎲 Tester 5 images aléatoires", type="primary"):
                sample_images = get_sample_images()
                
                # Récupérer 5 images aléatoires
                all_images = []
                for class_name, images in sample_images.items():
                    for img_path in images[:3]:  # Max 3 par classe
                        all_images.append((img_path, class_name))
                
                random_selection = random.sample(all_images, min(5, len(all_images)))
                
                results = []
                
                for i, (img_path, true_class) in enumerate(random_selection):
                    st.markdown(f"#### Test {i+1}/5")
                    
                    col_img, col_result = st.columns([1, 1])
                    
                    with col_img:
                        image = Image.open(img_path).convert('RGB')
                        st.image(image, caption=os.path.basename(img_path), width=200)
                    
                    with col_result:
                        predicted_class, confidence, all_probs = predict_image(model, classes, image)
                        
                        if predicted_class:
                            is_correct = predicted_class == true_class
                            status_icon = "✅" if is_correct else "❌"
                            
                            st.markdown(f"""
                            **{status_icon} Prédiction:** {predicted_class}  
                            **Confiance:** {confidence:.1%}  
                            **Attendu:** {true_class}  
                            **Statut:** {'CORRECT' if is_correct else 'INCORRECT'}
                            """)
                            
                            results.append({
                                'Image': os.path.basename(img_path),
                                'Attendu': true_class,
                                'Prédit': predicted_class,
                                'Confiance': f"{confidence:.1%}",
                                'Correct': is_correct
                            })
                
                # Résumé des résultats
                if results:
                    st.markdown("### 📊 Résumé des Tests")
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                    
                    accuracy = df_results['Correct'].mean()
                    st.metric("🎯 Accuracy", f"{accuracy:.1%}")
    
    with col2:
        st.markdown("## 📊 Informations")
        
        # Métriques du modèle
        st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
        st.markdown("### 🎯 Performance Modèle")
        st.markdown("**Dataset:** Fruits-360")
        st.markdown("**Accuracy officielle:** 100%")
        st.markdown("**Classes:** 5 fruits")
        st.markdown("**Architecture:** CNN")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Avertissement
        st.markdown('<div class="metric-card warning-metric">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Important")
        st.markdown("Ce modèle fonctionne parfaitement sur le dataset Fruits-360 mais peut avoir des performances limitées sur des images du monde réel.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Informations techniques
        st.markdown("### 🔧 Détails Techniques")
        st.markdown("""
        - **Taille d'entrée:** 100x100 pixels
        - **Normalisation:** 0-1
        - **Format:** RGB
        - **Batch Size:** 1 (prédiction)
        """)
        
        # Statistiques de session
        if 'prediction_count' not in st.session_state:
            st.session_state.prediction_count = 0
        
        st.markdown("### 📈 Session Stats")
        st.metric("Prédictions effectuées", st.session_state.prediction_count)
        
        # Bouton de reset
        if st.button("🔄 Reset Session"):
            st.session_state.prediction_count = 0
            st.experimental_rerun()

if __name__ == "__main__":
    main()