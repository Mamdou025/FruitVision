# Enhanced FruitVision V3 - Streamlit Educational App
# Ce script améliore votre version existante avec des illustrations SVG intégrées
# Instructions :
# 1. Installez les dépendances : pip install streamlit pillow
# 2. Exécutez avec : streamlit run enhanced_fruitvision.py

import streamlit as st
from PIL import Image
import os
import base64

# Configuration de la page
st.set_page_config(
    page_title="FruitVision V3 - Exploration pédagogique", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .bloc-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    
    .transformation-arrow {
        text-align: center;
        font-size: 2rem;
        color: #007bff;
        margin: 1rem 0;
    }
    
    .stage-indicator {
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .tech-specs {
        background: #f0f7ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour créer des illustrations SVG
def create_apple_svg(stage="original"):
    """Génère des illustrations SVG de pomme selon le stage de traitement"""
    
    if stage == "original":
        return """
        <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <radialGradient id="appleGrad" cx="0.3" cy="0.3">
                    <stop offset="0%" stop-color="#ff6b6b"/>
                    <stop offset="70%" stop-color="#e74c3c"/>
                    <stop offset="100%" stop-color="#c0392b"/>
                </radialGradient>
                <radialGradient id="highlight" cx="0.25" cy="0.25">
                    <stop offset="0%" stop-color="rgba(255,255,255,0.4)"/>
                    <stop offset="100%" stop-color="rgba(255,255,255,0)"/>
                </radialGradient>
            </defs>
            <circle cx="100" cy="110" r="70" fill="url(#appleGrad)"/>
            <ellipse cx="100" cy="110" rx="60" ry="70" fill="url(#highlight)"/>
            <path d="M85 50 Q100 40 115 50 Q110 60 100 60 Q90 60 85 50" fill="#654321"/>
            <path d="M105 54 Q120 46 130 60 Q125 66 115 62 Q110 58 105 54" fill="#55aa55"/>
            <text x="100" y="190" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">Image originale (100×100×3)</text>
        </svg>
        """
    
    elif stage == "preprocessed":
        return """
        <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <radialGradient id="appleGrad2" cx="0.3" cy="0.3">
                    <stop offset="0%" stop-color="#ff7b7b"/>
                    <stop offset="70%" stop-color="#e85c5c"/>
                    <stop offset="100%" stop-color="#d04949"/>
                </radialGradient>
                <filter id="enhance">
                    <feColorMatrix values="1.2 0 0 0 0  0 1.1 0 0 0  0 0 0.9 0 0  0 0 0 1 0"/>
                </filter>
            </defs>
            <circle cx="100" cy="110" r="70" fill="url(#appleGrad2)" filter="url(#enhance)"/>
            <path d="M85 50 Q100 40 115 50 Q110 60 100 60 Q90 60 85 50" fill="#654321"/>
            <path d="M105 54 Q120 46 130 60 Q125 66 115 62 Q110 58 105 54" fill="#55aa55"/>
            <text x="100" y="190" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">Normalisé & augmenté</text>
        </svg>
        """
    
    elif stage == "edges":
        return """
        <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="200" height="180" fill="#222"/>
            <circle cx="100" cy="110" r="70" fill="none" stroke="#fff" stroke-width="4"/>
            <ellipse cx="100" cy="110" rx="60" ry="70" fill="none" stroke="#ccc" stroke-width="2"/>
            <path d="M85 50 Q100 40 115 50" fill="none" stroke="#aaa" stroke-width="3"/>
            <text x="100" y="190" text-anchor="middle" font-family="Arial" font-size="14" fill="#fff">Détection des bords (32 filtres)</text>
        </svg>
        """
    
    elif stage == "features":
        return """
        <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <radialGradient id="featureGrad" cx="0.3" cy="0.3">
                    <stop offset="0%" stop-color="#ff8b8b"/>
                    <stop offset="70%" stop-color="#ee4e4e"/>
                    <stop offset="100%" stop-color="#cc2e2e"/>
                </radialGradient>
                <filter id="saturate">
                    <feColorMatrix values="1.8 0 0 0 0  0 1.4 0 0 0  0 0 1.4 0 0  0 0 0 1 0"/>
                </filter>
            </defs>
            <circle cx="100" cy="110" r="70" fill="url(#featureGrad)" filter="url(#saturate)"/>
            <path d="M85 50 Q100 40 115 50 Q110 60 100 60 Q90 60 85 50" fill="#654321"/>
            <path d="M105 54 Q120 46 130 60 Q125 66 115 62 Q110 58 105 54" fill="#55aa55"/>
            <!-- Ajout d'overlays pour représenter les features -->
            <circle cx="100" cy="110" r="70" fill="none" stroke="#4CAF50" stroke-width="2" opacity="0.7"/>
            <text x="100" y="190" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">Caractéristiques complexes (256 filtres)</text>
        </svg>
        """
    
    elif stage == "classified":
        return """
        <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <radialGradient id="finalGrad" cx="0.3" cy="0.3">
                    <stop offset="0%" stop-color="#ff6b6b"/>
                    <stop offset="70%" stop-color="#e74c3c"/>
                    <stop offset="100%" stop-color="#c0392b"/>
                </radialGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <circle cx="100" cy="110" r="70" fill="url(#finalGrad)" filter="url(#glow)"/>
            <path d="M85 50 Q100 40 115 50 Q110 60 100 60 Q90 60 85 50" fill="#654321"/>
            <path d="M105 54 Q120 46 130 60 Q125 66 115 62 Q110 58 105 54" fill="#55aa55"/>
            <!-- Indicateur de confiance -->
            <circle cx="100" cy="110" r="75" fill="none" stroke="#4CAF50" stroke-width="6" opacity="0.8"/>
            <text x="100" y="190" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">Classification finale (94.7% Pomme)</text>
        </svg>
        """

def create_filter_visualization():
    """Crée une visualisation des filtres de convolution"""
    return """
    <svg width="300" height="150" viewBox="0 0 300 150" xmlns="http://www.w3.org/2000/svg">
        <!-- Filtre 3x3 -->
        <g transform="translate(20, 20)">
            <text x="0" y="-5" font-size="12" fill="#333">Filtre 3×3 (détection de bords)</text>
            <rect x="0" y="0" width="30" height="30" fill="#e3f2fd" stroke="#1976d2"/>
            <rect x="30" y="0" width="30" height="30" fill="#bbdefb" stroke="#1976d2"/>
            <rect x="60" y="0" width="30" height="30" fill="#e3f2fd" stroke="#1976d2"/>
            <rect x="0" y="30" width="30" height="30" fill="#bbdefb" stroke="#1976d2"/>
            <rect x="30" y="30" width="30" height="30" fill="#2196f3" stroke="#1976d2"/>
            <rect x="60" y="30" width="30" height="30" fill="#bbdefb" stroke="#1976d2"/>
            <rect x="0" y="60" width="30" height="30" fill="#e3f2fd" stroke="#1976d2"/>
            <rect x="30" y="60" width="30" height="30" fill="#bbdefb" stroke="#1976d2"/>
            <rect x="60" y="60" width="30" height="30" fill="#e3f2fd" stroke="#1976d2"/>
            
            <!-- Valeurs du filtre -->
            <text x="15" y="20" text-anchor="middle" font-size="10">-1</text>
            <text x="45" y="20" text-anchor="middle" font-size="10">0</text>
            <text x="75" y="20" text-anchor="middle" font-size="10">1</text>
            <text x="15" y="50" text-anchor="middle" font-size="10">-2</text>
            <text x="45" y="50" text-anchor="middle" font-size="10">0</text>
            <text x="75" y="50" text-anchor="middle" font-size="10">2</text>
            <text x="15" y="80" text-anchor="middle" font-size="10">-1</text>
            <text x="45" y="80" text-anchor="middle" font-size="10">0</text>
            <text x="75" y="80" text-anchor="middle" font-size="10">1</text>
        </g>
        
        <!-- Flèche -->
        <path d="M 130 60 L 160 60" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
            </marker>
        </defs>
        
        <!-- Résultat -->
        <g transform="translate(180, 40)">
            <text x="0" y="-5" font-size="12" fill="#333">Carte de caractéristiques</text>
            <rect width="80" height="80" fill="url(#featureMapGrad)" stroke="#333"/>
            <defs>
                <linearGradient id="featureMapGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#4CAF50"/>
                    <stop offset="100%" stop-color="#81C784"/>
                </linearGradient>
            </defs>
        </g>
    </svg>
    """

def create_neural_network_diagram():
    """Crée un diagramme du réseau de neurones"""
    return """
    <svg width="400" height="200" viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
        <!-- Input layer -->
        <g transform="translate(50, 50)">
            <circle cx="0" cy="20" r="8" fill="#ff6b6b"/>
            <circle cx="0" cy="50" r="8" fill="#ff6b6b"/>
            <circle cx="0" cy="80" r="8" fill="#ff6b6b"/>
            <text x="0" y="110" text-anchor="middle" font-size="10">256 features</text>
        </g>
        
        <!-- Hidden layer 1 -->
        <g transform="translate(150, 30)">
            <circle cx="0" cy="20" r="8" fill="#4ecdc4"/>
            <circle cx="0" cy="40" r="8" fill="#4ecdc4"/>
            <circle cx="0" cy="60" r="8" fill="#4ecdc4"/>
            <circle cx="0" cy="80" r="8" fill="#4ecdc4"/>
            <circle cx="0" cy="100" r="8" fill="#4ecdc4"/>
            <text x="0" y="130" text-anchor="middle" font-size="10">512 units</text>
        </g>
        
        <!-- Hidden layer 2 -->
        <g transform="translate(250, 40)">
            <circle cx="0" cy="20" r="8" fill="#45b7d1"/>
            <circle cx="0" cy="40" r="8" fill="#45b7d1"/>
            <circle cx="0" cy="60" r="8" fill="#45b7d1"/>
            <circle cx="0" cy="80" r="8" fill="#45b7d1"/>
            <text x="0" y="110" text-anchor="middle" font-size="10">256 units</text>
        </g>
        
        <!-- Output layer -->
        <g transform="translate(330, 50)">
            <circle cx="0" cy="10" r="8" fill="#4CAF50"/>
            <circle cx="0" cy="30" r="8" fill="#ffeb3b"/>
            <circle cx="0" cy="50" r="8" fill="#f44336"/>
            <circle cx="0" cy="70" r="8" fill="#4CAF50"/>
            <circle cx="0" cy="90" r="8" fill="#ffeb3b"/>
            <text x="0" y="110" text-anchor="middle" font-size="10">5 classes</text>
        </g>
        
        <!-- Connections -->
        <g stroke="#ccc" stroke-width="1" opacity="0.6">
            <!-- Input to hidden1 -->
            <line x1="58" y1="70" x2="142" y2="50"/>
            <line x1="58" y1="100" x2="142" y2="70"/>
            <line x1="58" y1="130" x2="142" y2="90"/>
            
            <!-- Hidden1 to hidden2 -->
            <line x1="158" y1="50" x2="242" y2="60"/>
            <line x1="158" y1="70" x2="242" y2="80"/>
            <line x1="158" y1="90" x2="242" y2="100"/>
            
            <!-- Hidden2 to output -->
            <line x1="258" y1="60" x2="322" y2="60"/>
            <line x1="258" y1="80" x2="322" y2="80"/>
            <line x1="258" y1="100" x2="322" y2="100"/>
        </g>
    </svg>
    """

def create_probability_bars():
    """Crée des barres de probabilité animées avec Streamlit"""
    fruits = ["🍎 Pomme", "🍌 Banane", "🍅 Tomate", "🥒 Concombre", "🍋 Citron"]
    probabilities = [94.7, 3.2, 1.8, 0.2, 0.1]
    colors = ["#4CAF50", "#ffeb3b", "#f44336", "#4CAF50", "#ffeb3b"]
    
    for i, (fruit, prob, color) in enumerate(zip(fruits, probabilities, colors)):
        col1, col2, col3 = st.columns([2, 6, 1])
        with col1:
            st.write(fruit)
        with col2:
            st.progress(prob/100)
        with col3:
            st.write(f"{prob}%")

# Interface principale
def main():
    # En-tête principal avec style
    st.markdown("""
    <div class="main-header">
        <h1>🍎 FruitVision V3 - Exploration pédagogique d'un CNN</h1>
        <p>Suivez une pomme à travers chaque bloc du réseau de neurones pour comprendre l'apprentissage automatique</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar avec informations sur le modèle
    with st.sidebar:
        st.header("📊 Spécifications du modèle")
        st.info("""
        **Architecture FruitVision V3:**
        - Input: 100×100×3
        - Conv Blocks: 32→64→128→256
        - Dense: 512→256→5
        - Classes: Pomme, Banane, Tomate, Concombre, Citron
        - Précision: 96.8%
        """)
        
        st.header("🎯 Objectifs pédagogiques")
        st.write("""
        - Comprendre la convolution
        - Visualiser l'extraction de features
        - Observer la classification
        - Apprendre l'architecture CNN
        """)

    # Section d'introduction avec image
    st.header("🔬 Image d'entrée")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(create_apple_svg("original"), unsafe_allow_html=True)
        st.markdown('<div class="stage-indicator">Point de départ : Une pomme à classifier</div>', 
                   unsafe_allow_html=True)

    st.markdown("---")

    # Définition des blocs avec illustrations SVG
    blocs_data = [
        {
            "titre": "🔍 Bloc 1 - Détection des bords (32 filtres)",
            "svg_stage": "edges",
            "fonction": "Détecte des motifs simples comme les bords, lignes et textures de base.",
            "utilite": "Apprentissage des formes basiques visuelles - contours de la pomme, transitions de couleur.",
            "formule": "Convolution: S(i,j) = ∑∑ I(i+m,j+n)·K(m,n)",
            "code": """Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3))
BatchNormalization()
Conv2D(32, (3, 3), activation='relu')
MaxPooling2D(2, 2)
Dropout(0.25)""",
            "color": "#3498db"
        },
        {
            "titre": "🧩 Bloc 2 - Formes intermédiaires (64 filtres)",
            "svg_stage": "features",
            "fonction": "Combine les bords pour détecter des formes plus complexes et des textures.",
            "utilite": "Permet d'identifier des formes comme le rond de la pomme, sa tige, les variations de surface.",
            "formule": "Feature Map: F = ReLU(Conv(Input) + Bias)",
            "code": """Conv2D(64, (3, 3), activation='relu')
BatchNormalization()
Conv2D(64, (3, 3), activation='relu')
MaxPooling2D(2, 2)
Dropout(0.25)""",
            "color": "#9b59b6"
        },
        {
            "titre": "🎨 Bloc 3 - Motifs complexes (128 filtres)",
            "svg_stage": "features",
            "fonction": "Détecte des caractéristiques abstraites et des patterns spécifiques aux fruits.",
            "utilite": "Permet de différencier les pommes des tomates, reconnaître les textures spécifiques.",
            "formule": "Deep Features: F_deep = φ(W·F_prev + b)",
            "code": """Conv2D(128, (3, 3), activation='relu')
BatchNormalization()
Conv2D(128, (3, 3), activation='relu')
MaxPooling2D(2, 2)
Dropout(0.25)""",
            "color": "#e67e22"
        },
        {
            "titre": "🧠 Bloc 4 - Synthèse profonde (256 filtres)",
            "svg_stage": "features",
            "fonction": "Synthétise tous les patterns pour créer une représentation riche de l'objet.",
            "utilite": "Prépare une représentation vectorielle optimale pour la classification finale.",
            "formule": "Global Features: G = GlobalAvgPool(F_256)",
            "code": """Conv2D(256, (3, 3), activation='relu')
BatchNormalization()
Dropout(0.25)
GlobalAveragePooling2D()""",
            "color": "#e74c3c"
        },
        {
            "titre": "🎯 Classification finale - Prédiction (5 classes)",
            "svg_stage": "classified",
            "fonction": "Transforme les features en probabilités pour chaque classe de fruit.",
            "utilite": "Produit la prédiction finale avec un niveau de confiance pour chaque fruit.",
            "formule": "Softmax: P(y=i) = e^{z_i} / ∑e^{z_j}",
            "code": """Dense(512, activation='relu')
BatchNormalization()
Dropout(0.5)
Dense(256, activation='relu')
Dense(5, activation='softmax')""",
            "color": "#27ae60"
        }
    ]

    # Affichage des blocs avec illustrations
    for i, bloc in enumerate(blocs_data):
        with st.expander(bloc["titre"], expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(create_apple_svg(bloc["svg_stage"]), unsafe_allow_html=True)
                
                # Ajout de visualisations spécifiques selon le bloc
                if i == 0:  # Premier bloc - montrer les filtres
                    st.markdown("**Exemple de filtre de convolution:**")
                    st.markdown(create_filter_visualization(), unsafe_allow_html=True)
                elif i == 4:  # Dernier bloc - montrer les probabilités
                    st.markdown("**Probabilités de classification:**")
                    create_probability_bars()
            
            with col2:
                st.markdown(f"""
                <div class="tech-specs">
                    <h4 style="color: {bloc['color']}">🔍 Ce que cette couche fait :</h4>
                    <p>{bloc['fonction']}</p>
                    
                    <h4 style="color: {bloc['color']}">🎯 Utilité pédagogique :</h4>
                    <p>{bloc['utilite']}</p>
                    
                    <h4 style="color: {bloc['color']}">📐 Formule mathématique :</h4>
                    <code>{bloc['formule']}</code>
                    
                    <h4 style="color: {bloc['color']}">💻 Code Python :</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.code(bloc['code'], language='python')
            
            # Flèche de progression (sauf pour le dernier)
            if i < len(blocs_data) - 1:
                st.markdown('<div class="transformation-arrow">⬇️</div>', unsafe_allow_html=True)

    # Section finale avec résumé
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success("🎉 Vous avez parcouru toutes les étapes du modèle CNN FruitVision V3 !")
        st.markdown(create_neural_network_diagram(), unsafe_allow_html=True)
        
        st.info("""
        **🔑 Points clés à retenir :**
        - Chaque couche apprend des patterns de plus en plus complexes
        - Les filtres de convolution détectent des caractéristiques spécifiques
        - La classification finale combine toutes les features extraites
        - L'architecture progressive permet une reconnaissance robuste
        """)

    # Section interactive pour tester la compréhension
    st.header("🧪 Testez votre compréhension")
    
    with st.expander("Quiz interactif"):
        q1 = st.radio(
            "Que font principalement les premières couches convolutionnelles ?",
            ["Classifient directement les fruits", "Détectent des bords et textures simples", "Calculent les probabilités finales"]
        )
        if q1 == "Détectent des bords et textures simples":
            st.success("✅ Correct ! Les premières couches extraient des features basiques.")
        
        q2 = st.slider("Combien de classes FruitVision V3 peut-il distinguer ?", 1, 10, 5)
        if q2 == 5:
            st.success("✅ Exact ! Pomme, Banane, Tomate, Concombre, Citron.")

if __name__ == "__main__":
    main()