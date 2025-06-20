"""
Edufruits - Application Streamlit
===================================

Interface web pour la reconnaissance de fruits par IA
Modèle CNN avec 100% de précision sur 5 classes de fruits

Auteur: Mamadou Fall (22307101)
Cours: INF1402 - Apprentissage automatique
Date: 2025
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import time
from datetime import datetime
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="🍎 Edufruits - Reconnaissance de Fruits par IA",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .result-success {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    
    .result-warning {
        padding: 1rem;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    
    .model-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Classes de fruits (mapping CORRECT basé sur l'ordre d'entraînement)
CLASSES_FRUITS = {
    0: {"nom": "Pomme", "emoji": "🍎", "couleur": "#ff6b6b"},
    1: {"nom": "Banane", "emoji": "🍌", "couleur": "#feca57"},
    2: {"nom": "Kiwi", "emoji": "🥝", "couleur": "#6c5ce7"},
    3: {"nom": "Citron", "emoji": "🍋", "couleur": "#fdcb6e"},
    4: {"nom": "Pêche", "emoji": "🍑", "couleur": "#fd79a8"}
}

# NOTE: Ordre confirmé par diagnostic_approfondi.py
# Le LabelEncoder a utilisé l'ordre de création du dictionnaire, pas l'ordre alphabétique

@st.cache_resource
def charger_modele():
    """Chargement du modèle avec cache pour éviter de le recharger à chaque interaction."""
    try:
        # Essayer de charger le modèle équilibré en premier
        model_path = 'models/fruivision_equilibre.h5'
        model = tf.keras.models.load_model(model_path)
        model_info = {
            'nom': 'Edufruits Équilibré',
            'precision': '70-85% (honnête)',
            'images_entrainement': '1,000 (200 par fruit)',
            'epoques': 'Early stopping intelligent',
            'temps_entrainement': '3-5 minutes',
            'equilibre': 'Parfaitement équilibré'
        }
        return model, model_info, True
    except:
        try:
            # Fallback vers modèle sans augmentation
            model_path = 'models/fruivision_sans_augmentation.h5'
            model = tf.keras.models.load_model(model_path)
            model_info = {
                'nom': 'Edufruits Sans Augmentation',
                'precision': '~99% (mais déséquilibré)',
                'images_entrainement': '4,987 (déséquilibré)',
                'epoques': 'Early stopping à 19',
                'temps_entrainement': '8 minutes',
                'equilibre': 'Déséquilibré'
            }
            return model, model_info, True
        except:
            try:
                # Fallback vers ancien modèle robuste
                model_path = 'models/fruivision_robuste_1h.h5'
                model = tf.keras.models.load_model(model_path)
                model_info = {
                    'nom': 'Edufruits Robuste (DÉFAILLANT)',
                    'precision': '100% (FAUX - 20% réel)',
                    'images_entrainement': '28,225 (sur-augmenté)',
                    'epoques': '16 (early stopping prématuré)',
                    'temps_entrainement': '16 minutes',
                    'equilibre': 'Très déséquilibré'
                }
                return model, model_info, True
            except:
                return None, None, False

def preprocesser_image(image):
    """Preprocessing identique à l'entraînement avec gestion des formats d'image."""
    
    # CORRECTION: Forcer la conversion en RGB (3 canaux) si nécessaire
    if image.mode == 'RGBA':
        # Créer un fond blanc et coller l'image avec transparence
        fond_blanc = Image.new('RGB', image.size, (255, 255, 255))
        fond_blanc.paste(image, mask=image.split()[-1])  # Utiliser le canal alpha comme masque
        image = fond_blanc
    elif image.mode == 'P':
        # Images avec palette - convertir en RGB
        image = image.convert('RGB')
    elif image.mode == 'L':
        # Images en niveaux de gris - convertir en RGB
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        # Tout autre format - forcer RGB
        image = image.convert('RGB')
    
    # Redimensionner à 100x100
    image_resized = image.resize((100, 100), Image.Resampling.LANCZOS)
    
    # Convertir en array numpy
    image_array = np.array(image_resized, dtype=np.float32)
    
    # Vérification de sécurité - s'assurer qu'on a bien 3 canaux
    if len(image_array.shape) == 2:
        # Image en niveaux de gris - dupliquer les canaux
        image_array = np.stack([image_array, image_array, image_array], axis=-1)
    elif image_array.shape[-1] == 4:
        # Image RGBA - prendre seulement RGB
        image_array = image_array[:, :, :3]
    elif image_array.shape[-1] != 3:
        raise ValueError(f"Format d'image non supporté: {image_array.shape}")
    
    # Normaliser [0, 255] -> [0, 1]
    image_array = image_array / 255.0
    
    # Ajouter dimension batch
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predire_fruit(image, model, debug_mode=False):
    """Effectuer la prédiction sur une image avec gestion d'erreurs robuste."""
    try:
        # Debugging info (une seule fois, pas de checkbox répétées)
        if debug_mode:
            st.write(f"🔧 **Debug**: Format image: {image.mode}, Taille: {image.size}")
        
        # Preprocessing avec gestion des formats
        image_preprocessed = preprocesser_image(image)
        
        # DEBUG CRITIQUE - Vérifications avant prédiction
        if debug_mode:
            st.write(f"🔧 **Debug**: Shape: {image_preprocessed.shape}")
            st.write(f"🔧 **Debug**: Min/Max: {image_preprocessed.min():.3f}/{image_preprocessed.max():.3f}")
        
        # Prédiction
        debut = time.time()
        predictions = model.predict(image_preprocessed, verbose=0)
        temps_prediction = time.time() - debut
        
        # DEBUG CRITIQUE - Examiner les prédictions brutes
        if debug_mode:
            st.write(f"🔧 **Debug**: Probas brutes: {[f'{p:.3f}' for p in predictions[0]]}")
        
        # Extraire les probabilités
        probabilites = predictions[0]
        classe_predite = np.argmax(probabilites)
        confiance = probabilites[classe_predite]
        
        # DEBUG CRITIQUE - Vérifier le mapping
        if debug_mode:
            st.write(f"🔧 **Debug**: Classe {classe_predite} → {CLASSES_FRUITS[int(classe_predite)]['nom']}")
        
        # Créer le résultat détaillé
        resultat = {
            'classe': int(classe_predite),
            'fruit': CLASSES_FRUITS[int(classe_predite)]['nom'],
            'emoji': CLASSES_FRUITS[int(classe_predite)]['emoji'],
            'confiance': float(confiance),
            'probabilites': [float(p) for p in probabilites],
            'temps_prediction': round(temps_prediction * 1000, 2)
        }
        
        return resultat
        
    except Exception as e:
        st.error(f"❌ **Erreur**: {str(e)}")
        
        # Retourner un résultat d'erreur
        return {
            'classe': 0,
            'fruit': "Erreur",
            'emoji': "❌",
            'confiance': 0.0,
            'probabilites': [0.0, 0.0, 0.0, 0.0, 0.0],
            'temps_prediction': 0.0
        }

def afficher_resultats_prediction(resultat, seuil_confiance):
    """Afficher les résultats de prédiction de manière attrayante."""
    
    fruit = resultat['fruit']
    emoji = resultat['emoji']
    confiance = resultat['confiance']
    temps = resultat['temps_prediction']
    
    # Résultat principal
    if confiance >= seuil_confiance:
        st.markdown(f"""
        <div class="result-success">
            <h2 style="margin: 0; text-align: center;">
                {emoji} <strong>{fruit}</strong>
            </h2>
            <p style="margin: 0.5rem 0 0 0; text-align: center; font-size: 1.1rem;">
                Confiance: <strong>{confiance:.1%}</strong> | Temps: <strong>{temps} ms</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-warning">
            <h2 style="margin: 0; text-align: center;">
                🤔 <strong>Prédiction incertaine</strong>
            </h2>
            <p style="margin: 0.5rem 0 0 0; text-align: center; font-size: 1.1rem;">
                Meilleure hypothèse: {emoji} <strong>{fruit}</strong> ({confiance:.1%})
            </p>
            <p style="margin: 0.5rem 0 0 0; text-align: center;">
                Seuil requis: {seuil_confiance:.1%} | Temps: {temps} ms
            </p>
        </div>
        """, unsafe_allow_html=True)

def afficher_graphique_probabilites(resultat, image_index=0):
    """Afficher un graphique des probabilités pour tous les fruits."""
    
    probabilites = resultat['probabilites']
    
    # Préparer les données
    fruits_data = []
    for i, prob in enumerate(probabilites):
        fruits_data.append({
            'Fruit': f"{CLASSES_FRUITS[i]['emoji']} {CLASSES_FRUITS[i]['nom']}",
            'Probabilité': prob,
            'Couleur': CLASSES_FRUITS[i]['couleur']
        })
    
    df = pd.DataFrame(fruits_data)
    
    # Graphique en barres avec Plotly
    fig = px.bar(
        df, 
        x='Fruit', 
        y='Probabilité',
        color='Couleur',
        color_discrete_map={row['Couleur']: row['Couleur'] for _, row in df.iterrows()},
        title="📊 Probabilités de classification pour chaque fruit"
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Fruits",
        yaxis_title="Probabilité",
        yaxis=dict(range=[0, 1]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Ajouter des annotations pour les valeurs
    for i, prob in enumerate(probabilites):
        fig.add_annotation(
            x=i,
            y=prob + 0.02,
            text=f"{prob:.1%}",
            showarrow=False,
            font=dict(size=12, color="black")
        )
    
    # CORRECTION: Ajouter une clé unique pour chaque graphique
    st.plotly_chart(fig, use_container_width=True, key=f"probabilities_chart_{image_index}")
    
    # Afficher aussi un tableau détaillé des probabilités
    st.subheader("📋 Détail des Probabilités")
    
    # Créer un tableau plus lisible
    prob_data = []
    for i, prob in enumerate(probabilites):
        prob_data.append({
            'Rang': i + 1,
            'Fruit': f"{CLASSES_FRUITS[i]['emoji']} {CLASSES_FRUITS[i]['nom']}",
            'Probabilité': f"{prob:.1%}",
            'Score': f"{prob:.4f}"
        })
    
    # Trier par probabilité décroissante
    prob_data_sorted = sorted(prob_data, key=lambda x: float(x['Score']), reverse=True)
    
    # Afficher le tableau
    prob_df = pd.DataFrame(prob_data_sorted)
    prob_df['Rang'] = range(1, len(prob_df) + 1)  # Reclasser les rangs
    
    st.dataframe(prob_df[['Rang', 'Fruit', 'Probabilité']], use_container_width=True, hide_index=True)

def afficher_metriques_detaillees(resultat):
    """Afficher des métriques détaillées dans des colonnes."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4 style="margin: 0; color: #007bff;">🎯 Précision</h4>
            <h2 style="margin: 0.5rem 0 0 0; color: #333;">""" + f"{resultat['confiance']:.1%}" + """</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4 style="margin: 0; color: #28a745;">⚡ Vitesse</h4>
            <h2 style="margin: 0.5rem 0 0 0; color: #333;">""" + f"{resultat['temps_prediction']}" + """ ms</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculer l'incertitude (entropie)
        probs = np.array(resultat['probabilites'])
        probs = probs[probs > 0]  # Éviter log(0)
        entropie = -np.sum(probs * np.log2(probs))
        incertitude = entropie / np.log2(len(CLASSES_FRUITS))  # Normaliser
        
        st.markdown("""
        <div class="metric-container">
            <h4 style="margin: 0; color: #ffc107;">🤔 Incertitude</h4>
            <h2 style="margin: 0.5rem 0 0 0; color: #333;">""" + f"{incertitude:.1%}" + """</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Deuxième meilleure prédiction
        probs_sorted = sorted(enumerate(resultat['probabilites']), key=lambda x: x[1], reverse=True)
        deuxieme_classe = probs_sorted[1][0]
        deuxieme_prob = probs_sorted[1][1]
        
        st.markdown("""
        <div class="metric-container">
            <h4 style="margin: 0; color: #6c757d;">🥈 Alternative</h4>
            <h2 style="margin: 0.5rem 0 0 0; color: #333;">""" + f"{CLASSES_FRUITS[deuxieme_classe]['emoji']} {deuxieme_prob:.1%}" + """</h2>
        </div>
        """, unsafe_allow_html=True)

def generer_rapport_json(resultats_session):
    """Générer un rapport JSON des prédictions de la session."""
    
    # Nettoyer les données pour éviter les erreurs de sérialisation
    resultats_nettoyes = []
    for r in resultats_session:
        resultat_propre = {
            'classe': int(r['classe']) if 'classe' in r else 0,
            'fruit': str(r['fruit']) if 'fruit' in r else '',
            'emoji': str(r['emoji']) if 'emoji' in r else '',
            'confiance': float(r['confiance']) if 'confiance' in r else 0.0,
            'probabilites': [float(p) for p in r['probabilites']] if 'probabilites' in r else [],
            'temps_prediction': float(r['temps_prediction']) if 'temps_prediction' in r else 0.0,
            'nom_fichier': str(r['nom_fichier']) if 'nom_fichier' in r else '',
            'timestamp': str(r['timestamp']) if 'timestamp' in r else ''
        }
        resultats_nettoyes.append(resultat_propre)
    
    rapport = {
        'session_info': {
            'timestamp': datetime.now().isoformat(),
            'nb_predictions': len(resultats_nettoyes),
            'application': 'Edufruits',
            'version': '1.0',
            'auteur': 'Mamadou Fall (22307101)'
        },
        'predictions': resultats_nettoyes,
        'statistiques': {
            'confiance_moyenne': float(np.mean([r['confiance'] for r in resultats_nettoyes])) if resultats_nettoyes else 0.0,
            'temps_moyen_ms': float(np.mean([r['temps_prediction'] for r in resultats_nettoyes])) if resultats_nettoyes else 0.0,
            'fruits_detectes': list(set([r['fruit'] for r in resultats_nettoyes]))
        }
    }
    
    return json.dumps(rapport, indent=2, ensure_ascii=False)

def main():
    """Application principale Streamlit."""
    
    # En-tête principal
    st.markdown('<h1 class="main-header">🍎 Edufruits</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Reconnaissance intelligente de fruits par intelligence artificielle</p>', unsafe_allow_html=True)
    
    # Chargement du modèle
    with st.spinner("🤖 Chargement du modèle d'IA..."):
        model, model_info, success = charger_modele()
    
    if not success:
        st.error("❌ **Erreur**: Impossible de charger le modèle. Vérifiez que le fichier existe dans le dossier `models/`")
        st.info("📂 Modèles recherchés: `models/fruivision_robuste_1h.h5` ou `models/fruivision_final_optimise.h5`")
        st.stop()
    
    # Informations sur le modèle dans la sidebar
    with st.sidebar:
        st.header("🤖 Informations du Modèle")
        
        # Afficher le type de modèle avec un indicateur de qualité
        if "Équilibré" in model_info['nom']:
            st.success("✅ Modèle Recommandé")
        elif "Sans Augmentation" in model_info['nom']:
            st.warning("⚠️ Modèle Déséquilibré")
        elif "DÉFAILLANT" in model_info['nom']:
            st.error("❌ Modèle Défaillant")
        
        st.markdown(f"""
        <div class="model-info">
            <h4>{model_info['nom']}</h4>
            <p><strong>Précision:</strong> {model_info['precision']}</p>
            <p><strong>Images d'entraînement:</strong> {model_info['images_entrainement']}</p>
            <p><strong>Époques:</strong> {model_info['epoques']}</p>
            <p><strong>Temps d'entraînement:</strong> {model_info['temps_entrainement']}</p>
            <p><strong>Équilibre des données:</strong> {model_info['equilibre']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Avertissement pour modèles problématiques
        if "DÉFAILLANT" in model_info['nom']:
            st.error("⚠️ **ATTENTION**: Ce modèle a des performances trompeuses. Les 100% affichés sont faux due à une sur-augmentation défaillante.")
        elif "Déséquilibré" in model_info['equilibre']:
            st.warning("⚠️ **Note**: Ce modèle peut être biaisé vers certains fruits due à un déséquilibre dans les données d'entraînement.")
        
        st.header("⚙️ Paramètres")
        
        # Paramètres utilisateur adaptés
        if "Équilibré" in model_info['nom']:
            seuil_defaut = 0.6  # Plus permissif pour modèle équilibré
            aide_seuil = "Modèle équilibré: seuil plus bas recommandé"
        elif "Sans Augmentation" in model_info['nom']:
            seuil_defaut = 0.7  # Moyen pour modèle déséquilibré
            aide_seuil = "Modèle déséquilibré: seuil modéré recommandé"
        else:
            seuil_defaut = 0.8  # Plus strict pour modèle défaillant
            aide_seuil = "Modèle défaillant: seuil élevé pour filtrer les erreurs"
        
        seuil_confiance = st.slider(
            "🎯 Seuil de confiance minimal", 
            min_value=0.0, 
            max_value=1.0, 
            value=seuil_defaut, 
            step=0.05,
            help=aide_seuil
        )
        
        afficher_probabilites = st.checkbox("📊 Afficher le graphique des probabilités", value=True)
        afficher_metriques = st.checkbox("📈 Afficher les métriques détaillées", value=True)
        mode_debug = st.checkbox("🔧 Mode debug", value=False)
        
        # Ajout d'informations sur les performances attendues
        st.header("📊 Performances Attendues")
        if "Équilibré" in model_info['nom']:
            st.success("✅ Distinction fiable banane/kiwi")
            st.success("✅ Performances honnêtes")
            st.success("✅ Pas de biais majeur")
        elif "Sans Augmentation" in model_info['nom']:
            st.warning("⚠️ Possible biais vers pommes/pêches")
            st.warning("⚠️ Performances variables")
        else:
            st.error("❌ Prédictions peu fiables")
            st.error("❌ Confusion fréquente entre fruits")
        
        # Fruits supportés
        st.header("🍓 Fruits Reconnus")
        for info in CLASSES_FRUITS.values():
            st.write(f"{info['emoji']} {info['nom']}")
    
    # Zone principale - Upload et prédiction
    st.header("📤 Analysez vos fruits")
    
    # Instructions adaptées selon le modèle
    if "Équilibré" in model_info['nom']:
        st.info("💡 **Instructions**: Téléchargez une ou plusieurs images de fruits. Ce modèle équilibré offre des prédictions honnêtes et fiables.")
    elif "Sans Augmentation" in model_info['nom']:
        st.warning("⚠️ **Instructions**: Ce modèle peut être biaisé vers les pommes et pêches. Les résultats pour kiwis peuvent être moins fiables.")
    else:
        st.error("❌ **Attention**: Ce modèle a des performances trompeuses. Utilisez les résultats avec précaution.")
    
    # Upload de fichiers
    uploaded_files = st.file_uploader(
        "Choisissez vos images de fruits",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Formats supportés: JPG, JPEG, PNG. Vous pouvez télécharger plusieurs images à la fois."
    )
    
    # Initialiser l'historique de session
    if 'historique_predictions' not in st.session_state:
        st.session_state.historique_predictions = []
    
    if uploaded_files:
        for file_index, uploaded_file in enumerate(uploaded_files):
            # Séparateur visuel entre les images
            st.markdown("---")
            
            # Affichage en colonnes
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Afficher l'image originale
                image = Image.open(uploaded_file)
                st.image(image, caption=f"📷 {uploaded_file.name}", width=300)
                
                # Informations sur l'image
                if mode_debug:
                    st.write(f"**Dimensions originales**: {image.size}")
                    st.write(f"**Mode couleur**: {image.mode}")
                    st.write(f"**Taille fichier**: {len(uploaded_file.getvalue()) / 1024:.1f} KB")
                    
                    # Afficher l'image après conversion RGB si nécessaire
                    if image.mode != 'RGB':
                        image_rgb = image.convert('RGB')
                        st.image(image_rgb, caption="🔧 Après conversion RGB", width=200)
            
            with col2:
                # Prédiction avec mode debug global
                with st.spinner("🔍 Analyse en cours..."):
                    resultat = predire_fruit(image, model, mode_debug)
                
                # Vérifier si la prédiction a échoué
                if resultat['fruit'] == "Erreur":
                    st.error("❌ Erreur lors de la prédiction")
                    continue
                
                # Afficher les résultats
                afficher_resultats_prediction(resultat, seuil_confiance)
                
                # Métriques détaillées
                if afficher_metriques:
                    afficher_metriques_detaillees(resultat)
                
                # Diagnostic pour les prédictions incertaines
                if resultat['confiance'] < seuil_confiance:
                    st.warning("🔍 **Analyse de l'incertitude**:")
                    
                    # Analyser pourquoi la prédiction est incertaine
                    max_prob = max(resultat['probabilites'])
                    second_max = sorted(resultat['probabilites'], reverse=True)[1]
                    
                    if max_prob - second_max < 0.2:
                        st.write("• Les probabilités sont très proches entre plusieurs fruits")
                    if max_prob < 0.6:
                        st.write("• Le modèle n'est pas très confiant sur cette image")
                    if len([p for p in resultat['probabilites'] if p > 0.1]) > 2:
                        st.write("• Plusieurs fruits ont des probabilités significatives")
                    
                    # Suggestions
                    st.info("💡 **Suggestions**: Essayez une image plus claire, mieux éclairée, ou avec un fond plus neutre.")
                
                # Ajouter à l'historique avec index unique
                resultat_avec_image = resultat.copy()
                resultat_avec_image['nom_fichier'] = uploaded_file.name
                resultat_avec_image['timestamp'] = datetime.now().isoformat()
                resultat_avec_image['file_index'] = file_index  # Index unique
                st.session_state.historique_predictions.append(resultat_avec_image)
            
            # Graphique des probabilités (pleine largeur) avec clé unique
            if afficher_probabilites:
                afficher_graphique_probabilites(resultat, file_index)
    
    # Section historique et export
    if st.session_state.historique_predictions:
        st.header("📊 Historique de la Session")
        
        # Statistiques de session
        nb_predictions = len(st.session_state.historique_predictions)
        confiance_moyenne = np.mean([r['confiance'] for r in st.session_state.historique_predictions])
        fruits_uniques = len(set([r['fruit'] for r in st.session_state.historique_predictions]))
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔢 Prédictions", nb_predictions)
        col2.metric("🎯 Confiance Moyenne", f"{confiance_moyenne:.1%}")
        col3.metric("🍓 Fruits Différents", fruits_uniques)
        col4.metric("⚡ Temps Total", f"{sum([r['temps_prediction'] for r in st.session_state.historique_predictions]):.0f} ms")
        
        # Tableau récapitulatif
        historique_df = pd.DataFrame([
            {
                'Fichier': r['nom_fichier'],
                'Fruit': f"{r['emoji']} {r['fruit']}",
                'Confiance': f"{r['confiance']:.1%}",
                'Temps (ms)': r['temps_prediction']
            }
            for r in st.session_state.historique_predictions
        ])
        
        st.dataframe(historique_df, use_container_width=True)
        
        # Export des résultats
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            rapport_json = generer_rapport_json(st.session_state.historique_predictions)
            
            st.download_button(
                label="💾 Télécharger le rapport JSON",
                data=rapport_json,
                file_name=f"fruivision_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Télécharger un rapport détaillé de toutes les prédictions de cette session"
            )
        
        with col3:
            if st.button("🗑️ Vider l'historique"):
                st.session_state.historique_predictions = []
                st.experimental_rerun()
    
    # Footer avec informations sur les modèles
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p><strong>Edufruits v2.0</strong> - Développé par Mamadou Fall (22307101)</p>
        <p>Cours INF1402 - Apprentissage automatique | 2025</p>
        <p>🔬 Projet avec analyse critique des performances ML</p>
        <details>
            <summary>📊 Historique des Modèles</summary>
            <p><strong>V1 (Défaillant):</strong> 100% accuracy (faux) - Sur-augmentation massive</p>
            <p><strong>V2 (Honnête):</strong> 70-85% accuracy (réel) - Données équilibrées</p>
            <p><strong>Leçon:</strong> Validation externe > Métriques d'entraînement</p>
        </details>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()