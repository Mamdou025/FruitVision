import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.regularizers import l2

def creer_modele_cnn(forme_entree=(100, 100, 3), nb_classes=5, taux_dropout=0.5):
    """
    Crée un modèle de réseau de neurones convolutif (CNN) pour la classification de fruits.

    Pourquoi cette architecture ?
    - Simple : s'entraîne rapidement sur un GPU gratuit (Colab)
    - Profonde : détecte efficacement les caractéristiques visuelles des fruits
    - Moderne : intègre Dropout, régularisation L2, et bonnes pratiques

    Arguments :
        forme_entree (tuple) : dimensions des images d'entrée (hauteur, largeur, canaux)
        nb_classes (int) : nombre de classes à prédire
        taux_dropout (float) : pourcentage de neurones désactivés pendant l'entraînement

    Retour :
        Un modèle Keras compilé, prêt à être entraîné
    """

    modele = models.Sequential([
        
        # Couche d'entrée
        layers.Input(shape=forme_entree, name='images_entree'),

        # Bloc convolutionnel 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_1'),
        layers.Dropout(taux_dropout * 0.5, name='dropout_1'),  # Dropout léger en début de réseau

        # Bloc convolutionnel 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_2'),
        layers.Dropout(taux_dropout * 0.5, name='dropout_2'),

        # Bloc convolutionnel 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_3'),
        layers.Dropout(taux_dropout * 0.75, name='dropout_3'),  # Dropout plus fort en fin de réseau

        # Aplatissement de la sortie convolutive
        layers.Flatten(name='flatten'),

        # Couche dense entièrement connectée
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense_1'),

        # Dernier Dropout avant la sortie
        layers.Dropout(taux_dropout, name='dropout_final'),

        # Couche de sortie (classification)
        layers.Dense(nb_classes, activation='softmax', name='predictions')
    ])

    return modele


def compiler_modele(modele, taux_apprentissage=0.001):
    """
    Compile le modèle avec un optimiseur, une fonction de perte, et des métriques.

    Choix :
    - Adam : très utilisé pour sa rapidité et sa stabilité
    - categorical_crossentropy : adaptée à la classification multi-classes avec labels one-hot
    - accuracy et top_2_accuracy : évaluation utile dans un problème à 5 classes

    Retour :
        Modèle compilé prêt à l'entraînement
    """

    optimiseur = optimizers.Adam(
        learning_rate=taux_apprentissage,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    modele.compile(
    optimizer=optimiseur,
    loss='categorical_crossentropy',
    metrics=['accuracy']      # ← GARDEZ SEULEMENT ACCURACY
)

    return modele


def afficher_infos_modele(modele):
    """
    Affiche les informations détaillées du modèle : nombre de paramètres, mémoire estimée, etc.
    """

    total = modele.count_params()
    entrainables = sum([tf.keras.backend.count_params(w) for w in modele.trainable_weights])
    non_entrainables = total - entrainables
    taille_memoire = total * 4 / (1024 * 1024)

    print(f"📊 Paramètres totaux       : {total:,}")
    print(f"🔧 Paramètres entraînables : {entrainables:,}")
    print(f"❌ Non entraînables        : {non_entrainables:,}")
    print(f"💾 Taille mémoire estimée  : {taille_memoire:.2f} MB")
    print(f"📐 Forme d'entrée          : {modele.input_shape}")
    print(f"📤 Forme de sortie         : {modele.output_shape}")
    print(f"🔢 Nombre de couches       : {len(modele.layers)}")


def creer_modele_fruivision():
    """
    Fonction centrale qui crée, compile et affiche les infos du modèle Fruivision.
    """

    print("🔧 Création du modèle Fruivision...")

    modele = creer_modele_cnn(
        forme_entree=(100, 100, 3),
        nb_classes=5,
        taux_dropout=0.5
    )

    modele = compiler_modele(modele, taux_apprentissage=0.001)

    afficher_infos_modele(modele)

    return modele


# Test simple du modèle
if __name__ == "__main__":
    print("🧪 Test du modèle Fruivision...")

    modele = creer_modele_fruivision()
    modele.summary()

    # Test avec une image aléatoire
    import numpy as np
    print("\n🎯 Prédiction test avec une image fictive...")
    faux_input = np.random.random((1, 100, 100, 3))

    try:
        prediction = modele.predict(faux_input, verbose=0)
        print(f"✅ Prédiction générée avec succès : {prediction[0]}")
        print(f"🏆 Classe prédite : {np.argmax(prediction[0])}")
    except Exception as e:
        print(f"❌ Erreur pendant la prédiction : {e}")
