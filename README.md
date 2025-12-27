# Système de Recommandation Multimodal - Produits Chanel
### ---- Projet M2 Big Data Indexation de données: Système multimodal de recommandation de produits Chanel   ----

## 📋 Description
Système intelligent de recommandation de produits Chanel combinant analyse visuelle et textuelle. L'application utilise des modèles d'IA modernes (CLIP et DistilBERT) pour comprendre images et descriptions textuelles.

##  Fonctionnalités
- **Recherche par image** : Produits similaires visuellement
- **Recherche par texte** : Description en langage naturel
- **Recherche multimodale** : Combinaison image + texte
- **Interface intuitive** : Application web Streamlit

##  Installation

### Prérequis
- Python 3.8+
- pip installé

### Étapes
1. Cloner le dépôt :
```bash
git clone https://github.com/desbaa32/Projet-M2BD-Index-Syst-me_multimodal_recommandation_produits.git
cd Projet-M2BD-Index-Syst-me_multimodal_recommandation_produits bash 
``` 
Installer les dépendances :

```bash
pip install -r requirements.txt
 ``` 
Lancer l'application :

```bash
streamlit run app.py
``` 
##  -> Utilisation
Mode Texte : Entrez une description produit

Mode Image : Sélectionnez produit ou URL d'image

Mode Multimodal : Combinez image et texte

Paramètres : Ajustez nombre de recommandations et poids modalités

## -> Structure
```bash
Projet-M2BD-Index-Syst-me_multimodal_recommandation_produits/
├── app.py                                    # Application principale Streamlit
├── products_database_recom.csv               # Base de données des produits
├── visual_descriptors_recom.npy              # Descripteurs visuels pré-calculés
├── text_descriptors_recom.npy                # Descripteurs textuels pré-calculés
├── chanel_clean_dataset.csv                  # Dataset nettoyé
├── DataExploration_and_DescriptorExtraction__f.ipynb  # Analyse et extraction
├── Recommandation_system.ipynb               # Développement du système de recommandation
└── README.md                                 # Documentation
```
## ->Technologies
Backend : Python, PyTorch

Modèles : CLIP (vision), DistilBERT (texte)

Interface : Streamlit

Similarité : Cosine similarity

📝 Auteur
desbaa32 
