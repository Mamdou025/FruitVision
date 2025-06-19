import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="EduFruits - CNN Visualization",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .step-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define CNN steps
CNN_STEPS = [
    {
        "id": 0,
        "title": "🖼️ Image d'entrée",
        "description": "Image originale du fruit (32×32×3 RGB)",
        "details": """
        **Étape de préparation :**
        - L'image originale est redimensionnée à 32×32 pixels
        - Format RGB avec 3 canaux de couleur
        - Normalisation des valeurs de pixels (0-255 → 0-1)
        - Cette étape prépare l'image pour le traitement par le réseau
        """,
        "dimensions": "32×32×3",
        "operation": "Prétraitement",
        "params": {"pixels": 3072, "channels": 3}
    },
    {
        "id": 1,
        "title": "🔍 Couche Conv2D #1",
        "description": "Extraction des caractéristiques de base (32 filtres 3×3)",
        "details": """
        **Détection des contours et textures :**
        - 32 filtres convolutionnels de taille 3×3
        - Détection des bords, contours, et textures simples
        - Activation ReLU pour la non-linéarité
        - Chaque filtre apprend un motif différent
        """,
        "dimensions": "30×30×32",
        "operation": "Conv2D + ReLU",
        "params": {"filters": 32, "kernel_size": "3×3", "output_size": 28800}
    },
    {
        "id": 2,
        "title": "📉 Max Pooling #1",
        "description": "Réduction de dimensionalité (2×2)",
        "details": """
        **Sous-échantillonnage :**
        - Fenêtre de pooling 2×2
        - Conserve seulement la valeur maximale de chaque région
        - Réduit la taille spatiale de moitié
        - Invariance aux petites translations
        """,
        "dimensions": "15×15×32",
        "operation": "MaxPool2D",
        "params": {"pool_size": "2×2", "output_size": 7200}
    },
    {
        "id": 3,
        "title": "🔍 Couche Conv2D #2",
        "description": "Détection de motifs complexes (64 filtres)",
        "details": """
        **Motifs de niveau intermédiaire :**
        - 64 filtres convolutionnels
        - Détection de formes et textures combinées
        - Capture des motifs plus abstraits
        - Accumulation des caractéristiques locales
        """,
        "dimensions": "13×13×64",
        "operation": "Conv2D + ReLU",
        "params": {"filters": 64, "kernel_size": "3×3", "output_size": 10816}
    },
    {
        "id": 4,
        "title": "📉 Max Pooling #2",
        "description": "Deuxième réduction dimensionnelle",
        "details": """
        **Concentration des caractéristiques :**
        - Nouvelle réduction spatiale
        - Focus sur les caractéristiques les plus saillantes
        - Préparation pour les couches suivantes
        - Réduction du surapprentissage
        """,
        "dimensions": "6×6×64",
        "operation": "MaxPool2D",
        "params": {"pool_size": "2×2", "output_size": 2304}
    },
    {
        "id": 5,
        "title": "🔍 Couche Conv2D #3",
        "description": "Caractéristiques de haut niveau (128 filtres)",
        "details": """
        **Caractéristiques spécifiques aux fruits :**
        - 128 filtres pour capturer des détails fins
        - Reconnaissance de motifs spécifiques aux classes
        - Dernière étape d'extraction de caractéristiques
        - Préparation pour la classification
        """,
        "dimensions": "4×4×128",
        "operation": "Conv2D + ReLU",
        "params": {"filters": 128, "kernel_size": "3×3", "output_size": 2048}
    },
    {
        "id": 6,
        "title": "📊 Aplatissement (Flatten)",
        "description": "Conversion 3D vers vecteur 1D",
        "details": """
        **Préparation pour la classification :**
        - Transformation des cartes 3D en vecteur 1D
        - 2048 valeurs numériques
        - Chaque valeur représente une caractéristique extraite
        - Interface entre extraction et classification
        """,
        "dimensions": "2048×1",
        "operation": "Flatten",
        "params": {"input_shape": "4×4×128", "output_size": 2048}
    },
    {
        "id": 7,
        "title": "🧠 Couche Dense",
        "description": "Combinaison des caractéristiques (128 neurones)",
        "details": """
        **Apprentissage des associations :**
        - 128 neurones entièrement connectés
        - Combinaison intelligente des 2048 caractéristiques
        - Dropout (50%) pour éviter le surapprentissage
        - Apprentissage des relations complexes
        """,
        "dimensions": "128×1",
        "operation": "Dense + ReLU + Dropout",
        "params": {"neurons": 128, "dropout": 0.5, "connections": 262144}
    },
    {
        "id": 8,
        "title": "🎯 Couche de Sortie",
        "description": "Classification finale (5 classes)",
        "details": """
        **Décision finale :**
        - 5 neurones (un par fruit)
        - Activation Softmax pour les probabilités
        - Somme des probabilités = 100%
        - Prédiction = classe avec probabilité maximale
        """,
        "dimensions": "5×1",
        "operation": "Dense + Softmax",
        "params": {"classes": 5, "activation": "softmax", "connections": 640}
    }
]

def create_feature_map_visualization(step_id, cols=8):
    """Create a visualization of feature maps for convolutional layers"""
    step = CNN_STEPS[step_id]
    
    if "Conv2D" in step["operation"]:
        filters = step["params"]["filters"]
        
        # Create a grid of feature maps
        rows = (filters + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Filter {i+1}" for i in range(filters)],
            horizontal_spacing=0.02,
            vertical_spacing=0.05
        )
        
        for i in range(filters):
            row = i // cols + 1
            col = i % cols + 1
            
            # Generate synthetic feature map data
            np.random.seed(i)  # For reproducible visualization
            if step_id == 1:  # First conv layer - edges and simple features
                size = 30
                feature_map = np.random.rand(size, size) * 0.5
                # Add some edge-like patterns
                if i % 4 == 0:  # Vertical edges
                    feature_map[:, size//3:size//3+2] = 1.0
                elif i % 4 == 1:  # Horizontal edges
                    feature_map[size//3:size//3+2, :] = 1.0
                elif i % 4 == 2:  # Diagonal edges
                    np.fill_diagonal(feature_map, 1.0)
                else:  # Texture patterns
                    feature_map = np.sin(np.linspace(0, 4*np.pi, size*size)).reshape(size, size)
                    feature_map = (feature_map + 1) / 2
            elif step_id == 3:  # Second conv layer - complex patterns
                size = 13
                feature_map = np.random.rand(size, size) * 0.3
                # Add more complex patterns
                center = size // 2
                y, x = np.ogrid[:size, :size]
                if i % 3 == 0:  # Circular patterns
                    mask = (x - center)**2 + (y - center)**2 <= (size//3)**2
                    feature_map[mask] = 1.0
                elif i % 3 == 1:  # Corner patterns
                    feature_map[:size//2, :size//2] = 1.0
                else:  # Gradient patterns
                    feature_map = np.linspace(0, 1, size*size).reshape(size, size)
            else:  # Third conv layer - high-level features
                size = 4
                feature_map = np.random.rand(size, size)
                # High-level, more abstract patterns
                if i % 2 == 0:
                    feature_map = np.ones((size, size)) * (0.3 + 0.7 * (i % 10) / 10)
                else:
                    feature_map = np.random.rand(size, size)
            
            fig.add_trace(
                go.Heatmap(
                    z=feature_map,
                    colorscale='Viridis',
                    showscale=False,
                    hovertemplate=f'Filter {i+1}<br>Value: %{{z:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=max(400, rows * 100),
            title_text=f"Cartes de caractéristiques - {step['title']}",
            showlegend=False
        )
        
        # Remove axis labels for cleaner look
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig
    
    return None

def create_architecture_flow():
    """Create a flowchart of the CNN architecture"""
    fig = go.Figure()
    
    # Define positions for each layer
    positions = [
        (0, 4, "Input\n32×32×3"),
        (1, 4, "Conv2D\n30×30×32"),
        (2, 4, "MaxPool\n15×15×32"),
        (3, 4, "Conv2D\n13×13×64"),
        (4, 4, "MaxPool\n6×6×64"),
        (5, 4, "Conv2D\n4×4×128"),
        (6, 4, "Flatten\n2048"),
        (7, 4, "Dense\n128"),
        (8, 4, "Output\n5")
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#82E0AA']
    
    # Add boxes for each layer
    for i, (x, y, text) in enumerate(positions):
        fig.add_shape(
            type="rect",
            x0=x-0.4, y0=y-0.3, x1=x+0.4, y1=y+0.3,
            fillcolor=colors[i],
            line=dict(color="white", width=2)
        )
        
        fig.add_annotation(
            x=x, y=y,
            text=text,
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor=colors[i],
            bordercolor="white",
            borderwidth=1
        )
        
        # Add arrows between layers
        if i < len(positions) - 1:
            fig.add_annotation(
                x=x+0.5, y=y,
                ax=x+0.4, ay=y,
                xref="x", yref="y",
                axref="x", ayref="y",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="gray"
            )
    
    fig.update_layout(
        xaxis=dict(range=[-0.5, 8.5], showgrid=False, showticklabels=False),
        yaxis=dict(range=[3, 5], showgrid=False, showticklabels=False),
        height=200,
        title="Architecture du CNN EduFruits",
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig

def create_probability_chart(fruits, probabilities):
    """Create a bar chart for prediction probabilities"""
    df = pd.DataFrame({
        'Fruit': fruits,
        'Probabilité': probabilities
    })
    
    fig = px.bar(
        df, x='Probabilité', y='Fruit',
        orientation='h',
        color='Probabilité',
        color_continuous_scale='Viridis',
        title="Probabilités de classification"
    )
    
    fig.update_layout(
        height=300,
        xaxis_title="Probabilité (%)",
        yaxis_title="Type de fruit"
    )
    
    return fig






# Main Streamlit App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍎 EduFruits - Visualisation CNN Interactive</h1>
        <p>Explorez chaque étape du processus de reconnaissance de fruits</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    
    # Step selection
    step_titles = [f"Étape {step['id']+1}: {step['title']}" for step in CNN_STEPS]
    selected_step_idx = st.sidebar.selectbox(
        "Choisissez une étape:",
        range(len(CNN_STEPS)),
        format_func=lambda x: step_titles[x],
        index=0
    )
    
    current_step = CNN_STEPS[selected_step_idx]
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Progression")
    progress = (selected_step_idx + 1) / len(CNN_STEPS)
    st.sidebar.progress(progress)
    st.sidebar.write(f"Étape {selected_step_idx + 1} sur {len(CNN_STEPS)}")
    
    # Quick navigation buttons
    st.sidebar.markdown("### ⚡ Navigation rapide")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("⬅️ Précédent", disabled=selected_step_idx == 0):
            selected_step_idx = max(0, selected_step_idx - 1)
            st.experimental_rerun()
    
    with col2:
        if st.button("Suivant ➡️", disabled=selected_step_idx == len(CNN_STEPS) - 1):
            selected_step_idx = min(len(CNN_STEPS) - 1, selected_step_idx + 1)
            st.experimental_rerun()
    
    # Architecture overview
    if st.sidebar.checkbox("🏗️ Vue d'ensemble de l'architecture"):
        st.sidebar.plotly_chart(create_architecture_flow(), use_container_width=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Step header
        st.markdown(f"""
        <div class="step-card">
            <h2>{current_step['title']}</h2>
            <p><strong>{current_step['description']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        display_step_metrics(current_step)
        
        # Visualization
        st.markdown("### 🎨 Visualisation")
        
        if "Conv2D" in current_step['operation']:
            # Show feature maps for convolutional layers
            fig = create_feature_map_visualization(current_step['id'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        elif current_step['id'] == 0:  # Input image
            # Show sample input image
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.image("https://via.placeholder.com/200x200/FF6B6B/FFFFFF?text=🍎", 
                        caption="Image d'exemple - Pomme (32×32×3)")
        
        elif current_step['id'] == 6:  # Flatten
            # Show flattening visualization
            fig = go.Figure()
            
            # Create a visualization of flattening process
            x_vals = list(range(2048))
            y_vals = np.random.rand(2048) * 0.8 + 0.1
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name='Valeurs aplaties',
                line=dict(color='#667eea', width=2)
            ))
            
            fig.update_layout(
                title="Vecteur de caractéristiques aplati (2048 valeurs)",
                xaxis_title="Index de la caractéristique",
                yaxis_title="Valeur normalisée",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif current_step['id'] == 8:  # Output
            # Show prediction probabilities
            fruits = ['🍎 Pomme', '🍌 Banane', '🥝 Kiwi', '🍋 Citron', '🍑 Pêche']
            probabilities = [85.2, 8.1, 4.2, 1.8, 0.7]
            
            fig = create_probability_chart(fruits, probabilities)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show prediction result
            st.success(f"🎯 **Prédiction: Pomme** (Confiance: {probabilities[0]}%)")
    
    with col2:
        # Information panel
        st.markdown("### 📚 Informations détaillées")
        
        st.markdown(f"""
        <div class="info-box">
            {current_step['details']}
        </div>
        """, unsafe_allow_html=True)
        
        # Technical parameters
        st.markdown("### ⚙️ Paramètres techniques")
        for param, value in current_step['params'].items():
            st.write(f"**{param.replace('_', ' ').title()}:** {value}")
        
        # Educational notes
        st.markdown("### 💡 Notes pédagogiques")
        
        if current_step['id'] == 0:
            st.info("💭 **Réflexion:** Pourquoi normaliser les pixels entre 0 et 1?")
        elif "Conv2D" in current_step['operation']:
            st.info("💭 **Réflexion:** Comment les filtres apprennent-ils automatiquement?")
        elif "Pool" in current_step['operation']:
            st.info("💭 **Réflexion:** Que perdons-nous lors du pooling?")
        elif current_step['id'] == 6:
            st.info("💭 **Réflexion:** Pourquoi aplatir avant la classification?")
        elif current_step['id'] == 7:
            st.info("💭 **Réflexion:** Quel est le rôle du dropout?")
        else:
            st.info("💭 **Réflexion:** Comment Softmax garantit-il des probabilités?")
    
    # Footer with educational resources
    st.markdown("---")
    st.markdown("### 📖 Ressources pédagogiques")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎥 Vidéos recommandées:**
        - [3Blue1Brown - CNN](https://www.youtube.com/watch?v=KuXjwB4LzSA)
        - [Convolution visualized](https://www.youtube.com/watch?v=KuXjwB4LzSA)
        """)
    
    with col2:
        st.markdown("""
        **📚 Lectures:**
        - [CS231n Stanford](https://cs231n.github.io/)
        - [Deep Learning Book](https://www.deeplearningbook.org/)
        """)
    
    with col3:
        st.markdown("""
        **🛠️ Outils interactifs:**
        - [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
        - [TensorFlow Playground](https://playground.tensorflow.org/)
        """)

if __name__ == "__main__":
    main()