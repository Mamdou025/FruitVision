"""
FruitVision - Preprocessing pour Images du Monde Réel
====================================================

Module spécialisé pour traiter les images uploadées par les utilisateurs:
- Fonds complexes
- Éclairages variables  
- Orientations diverses
- Qualités différentes

Auteur: Mamadou Fall
Date: 2025
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class PreprocesseurMondeReel:
    """
    Classe pour preprocesser les images réelles uploadées par les utilisateurs.
    
    Défis spécifiques:
    - Fonds non-uniformes
    - Éclairages variables
    - Fruits non-centrés
    - Différentes résolutions
    """
    
    def __init__(self, taille_cible=(100, 100)):
        """
        Initialiser le preprocesseur pour images réelles.
        
        Args:
            taille_cible (tuple): Taille finale pour le modèle
        """
        self.taille_cible = taille_cible
        
    def preprocesser_image_utilisateur(self, image_path: str, debug=False) -> np.ndarray:
        """
        Pipeline complet pour preprocesser une image utilisateur.
        
        Étapes:
        1. Charger et normaliser l'image
        2. Améliorer le contraste si nécessaire  
        3. Redimensionner intelligemment
        4. Centrer le fruit si possible
        5. Normaliser pour le modèle
        
        Args:
            image_path (str): Chemin vers l'image
            debug (bool): Afficher les étapes de preprocessing
            
        Returns:
            np.ndarray: Image prête pour prédiction (1, 100, 100, 3)
        """
        
        if debug:
            print(f"🔄 Preprocessing de l'image: {image_path}")
        
        # ÉTAPE 1: Charger l'image
        try:
            image_originale = Image.open(image_path)
            if debug:
                print(f"   📐 Taille originale: {image_originale.size}")
                print(f"   🎨 Mode: {image_originale.mode}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return None
        
        # ÉTAPE 2: Convertir en RGB si nécessaire
        if image_originale.mode != 'RGB':
            image_originale = image_originale.convert('RGB')
            if debug:
                print("   🔄 Conversion en RGB")
        
        # ÉTAPE 3: Amélioration automatique de l'image
        image_amelioree = self._ameliorer_image(image_originale, debug)
        
        # ÉTAPE 4: Redimensionnement intelligent
        image_redimensionnee = self._redimensionner_intelligent(image_amelioree, debug)
        
        # ÉTAPE 5: Normalisation finale
        image_finale = self._normaliser_pour_modele(image_redimensionnee, debug)
        
        if debug:
            print(f"   ✅ Forme finale: {image_finale.shape}")
        
        return image_finale
    
    def _ameliorer_image(self, image: Image.Image, debug=False) -> Image.Image:
        """
        Améliorer automatiquement la qualité de l'image.
        
        Améliorations:
        - Contraste adaptatif
        - Luminosité si trop sombre
        - Netteté si flou
        """
        
        # Analyser l'image pour déterminer les améliorations nécessaires
        image_array = np.array(image)
        luminosite_moyenne = np.mean(image_array)
        
        if debug:
            print(f"   💡 Luminosité moyenne: {luminosite_moyenne:.1f}")
        
        # Amélioration du contraste
        enhancer_contraste = ImageEnhance.Contrast(image)
        if luminosite_moyenne < 100:  # Image sombre
            facteur_contraste = 1.3
        elif luminosite_moyenne > 200:  # Image très claire
            facteur_contraste = 1.1
        else:  # Image normale
            facteur_contraste = 1.2
            
        image = enhancer_contraste.enhance(facteur_contraste)
        
        # Amélioration de la luminosité si nécessaire
        if luminosite_moyenne < 80:
            enhancer_luminosite = ImageEnhance.Brightness(image)
            image = enhancer_luminosite.enhance(1.3)
            if debug:
                print("   🔆 Luminosité améliorée")
        
        # Amélioration de la netteté
        enhancer_nettete = ImageEnhance.Sharpness(image)
        image = enhancer_nettete.enhance(1.1)
        
        if debug:
            print(f"   ✨ Améliorations appliquées (contraste: {facteur_contraste})")
        
        return image
    
    def _redimensionner_intelligent(self, image: Image.Image, debug=False) -> Image.Image:
        """
        Redimensionner en préservant l'aspect ratio et en centrant.
        
        Stratégie:
        1. Calculer le ratio pour que la plus grande dimension = 100px
        2. Redimensionner en gardant les proportions
        3. Centrer sur un canvas 100x100 avec fond flou
        """
        
        largeur, hauteur = image.size
        
        # Calculer le facteur de redimensionnement
        facteur = min(self.taille_cible[0] / largeur, self.taille_cible[1] / hauteur)
        
        nouvelle_largeur = int(largeur * facteur)
        nouvelle_hauteur = int(hauteur * facteur)
        
        # Redimensionner l'image
        image_redim = image.resize((nouvelle_largeur, nouvelle_hauteur), Image.Resampling.LANCZOS)
        
        if debug:
            print(f"   📏 Redimensionnement: {largeur}x{hauteur} → {nouvelle_largeur}x{nouvelle_hauteur}")
        
        # Créer un canvas avec fond intelligent
        canvas = self._creer_fond_intelligent(image, self.taille_cible)
        
        # Centrer l'image redimensionnée
        pos_x = (self.taille_cible[0] - nouvelle_largeur) // 2
        pos_y = (self.taille_cible[1] - nouvelle_hauteur) // 2
        
        canvas.paste(image_redim, (pos_x, pos_y))
        
        if debug:
            print(f"   🎯 Image centrée à ({pos_x}, {pos_y})")
        
        return canvas
    
    def _creer_fond_intelligent(self, image: Image.Image, taille: Tuple[int, int]) -> Image.Image:
        """
        Créer un fond intelligent pour l'image.
        
        Stratégies:
        1. Fond flou de l'image originale (effet bokeh)
        2. Couleur dominante des bords
        3. Gris neutre en dernier recours
        """
        
        # Stratégie 1: Fond flou de l'image
        try:
            # Redimensionner l'image pour remplir le canvas
            image_fond = image.resize(taille, Image.Resampling.LANCZOS)
            # Appliquer un flou gaussien fort
            from PIL import ImageFilter
            image_fond = image_fond.filter(ImageFilter.GaussianBlur(radius=8))
            # Réduire l'opacité
            enhancer = ImageEnhance.Brightness(image_fond)
            image_fond = enhancer.enhance(0.7)  # Plus sombre
            return image_fond
        except:
            # Stratégie de secours: fond gris clair
            return Image.new('RGB', taille, color=(240, 240, 240))
    
    def _normaliser_pour_modele(self, image: Image.Image, debug=False) -> np.ndarray:
        """
        Normaliser l'image pour qu'elle soit compatible avec le modèle.
        
        Returns:
            np.ndarray: Array (1, 100, 100, 3) normalisé entre 0 et 1
        """
        
        # Convertir en array numpy
        image_array = np.array(image, dtype=np.float32)
        
        # Normaliser entre 0 et 1
        image_array = image_array / 255.0
        
        # Ajouter dimension batch
        image_array = np.expand_dims(image_array, axis=0)
        
        if debug:
            print(f"   📊 Valeurs min/max: {image_array.min():.3f}/{image_array.max():.3f}")
        
        return image_array
    
    def visualiser_preprocessing(self, image_path: str):
        """
        Visualiser les étapes du preprocessing pour debug.
        
        Args:
            image_path (str): Chemin vers l'image à analyser
        """
        
        print("🔍 Visualisation du Preprocessing")
        print("=" * 50)
        
        # Image originale
        image_originale = Image.open(image_path).convert('RGB')
        
        # Étapes du preprocessing
        image_amelioree = self._ameliorer_image(image_originale, debug=True)
        image_redimensionnee = self._redimensionner_intelligent(image_amelioree, debug=True)
        image_finale = self._normaliser_pour_modele(image_redimensionnee, debug=True)
        
        # Affichage visuel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_originale)
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
        
        axes[1].imshow(image_amelioree)
        axes[1].set_title('Après Amélioration')
        axes[1].axis('off')
        
        axes[2].imshow(image_redimensionnee)
        axes[2].set_title('Prête pour Modèle')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return image_finale
    
    def evaluer_qualite_image(self, image_path: str) -> dict:
        """
        Évaluer la qualité d'une image pour prédiction.
        
        Returns:
            dict: Scores de qualité et recommandations
        """
        
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Métriques de qualité
        luminosite = np.mean(image_array)
        contraste = np.std(image_array)
        netteté = self._calculer_nettete(image_array)
        
        # Scores (0-100)
        score_luminosite = min(100, max(0, 100 - abs(luminosite - 128) * 2))
        score_contraste = min(100, contraste * 2)
        score_nettete = min(100, netteté * 10)
        
        score_global = (score_luminosite + score_contraste + score_nettete) / 3
        
        # Recommandations
        recommandations = []
        if score_luminosite < 70:
            recommandations.append("Améliorer l'éclairage")
        if score_contraste < 50:
            recommandations.append("Augmenter le contraste")
        if score_nettete < 60:
            recommandations.append("Image plus nette recommandée")
        
        return {
            'score_global': score_global,
            'luminosite': score_luminosite,
            'contraste': score_contraste,
            'nettete': score_nettete,
            'recommandations': recommandations,
            'qualite': 'Excellente' if score_global > 80 else 
                      'Bonne' if score_global > 60 else
                      'Moyenne' if score_global > 40 else 'Faible'
        }
    
    def _calculer_nettete(self, image_array: np.ndarray) -> float:
        """Calculer un score de netteté basé sur le gradient."""
        gris = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gris, cv2.CV_64F).var()


# Fonction utilitaire pour tests rapides
def tester_preprocessing_rapide(image_path: str):
    """
    Test rapide du preprocessing sur une image.
    
    Args:
        image_path (str): Chemin vers l'image de test
    """
    
    preprocesseur = PreprocesseurMondeReel()
    
    print(f"🧪 Test de preprocessing: {image_path}")
    
    # Évaluer la qualité
    qualite = preprocesseur.evaluer_qualite_image(image_path)
    print(f"📊 Qualité: {qualite['qualite']} (Score: {qualite['score_global']:.1f}/100)")
    
    if qualite['recommandations']:
        print("💡 Recommandations:")
        for rec in qualite['recommandations']:
            print(f"   - {rec}")
    
    # Preprocesser
    image_preprocessee = preprocesseur.preprocesser_image_utilisateur(image_path, debug=True)
    
    if image_preprocessee is not None:
        print("✅ Preprocessing réussi!")
        return image_preprocessee
    else:
        print("❌ Échec du preprocessing")
        return None


if __name__ == "__main__":
    """
    Test du module de preprocessing monde réel.
    """
    
    print("🌍 Test du Preprocessing Monde Réel")
    print("=" * 50)
    
    # Test avec une image d'exemple (remplacez par votre image)
    image_test = "test_image.jpg"  # Remplacez par le chemin de votre image
    
    if os.path.exists(image_test):
        preprocesseur = PreprocesseurMondeReel()
        
        # Visualisation complète
        image_finale = preprocesseur.visualiser_preprocessing(image_test)
        
        # Évaluation de qualité
        qualite = preprocesseur.evaluer_qualite_image(image_test)
        print(f"\n📊 Évaluation de qualité:")
        print(f"   Score global: {qualite['score_global']:.1f}/100")
        print(f"   Qualité: {qualite['qualite']}")
        
        print(f"\n✅ Module de preprocessing prêt pour images réelles!")
    else:
        print(f"⚠️ Image de test non trouvée: {image_test}")
        print("💡 Placez une image test pour essayer le preprocessing")