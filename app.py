from typing import List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
import re

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------------
# Classe du système de recommandation
# -----------------------------------------------------------------------------------
class MultimodalRecommendationSystem:
    def __init__(self, df, visual_descriptors, text_descriptors):
        self.df = df.reset_index(drop=True)
        self.visual_descriptors = visual_descriptors
        self.text_descriptors = text_descriptors
        self.visual_descriptors_norm = normalize(visual_descriptors.astype(np.float32))
        self.text_descriptors_norm = normalize(text_descriptors.astype(np.float32))
        self.batch_size = 32
        self.max_text_length = 128

        self.clip_model = None
        self.clip_processor = None
        self.text_tokenizer = None
        self.text_model = None

    def _load_clip_model(self):
        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(device)
            self.clip_model.eval()

    def _load_distilbert_model(self):
        if self.text_model is None:
            model_name = "distilbert-base-uncased"
            self.text_tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.text_model = DistilBertModel.from_pretrained(model_name)
            self.text_model.eval()
            self.text_model.to(device)

    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s\-&]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_external_visual_features(self, image_url):
        self._load_clip_model()
        try:
            response = requests.get(image_url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert('RGB')

            inputs = self.clip_processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                features = features.squeeze().cpu().numpy()
            return features
        except Exception as e:
            st.error(f"Erreur lors de l'extraction de l'image : {e}")
            return None

    def _extract_distilbert_features_batch(self, texts_batch: list) -> Optional[np.ndarray]:
        self._load_distilbert_model()
        try:
            inputs = self.text_tokenizer(
                texts_batch,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.text_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings.astype(np.float32)

        except Exception as e:
            st.error(f"Erreur dans l'extraction batch : {e}")
            return None

    def _extract_external_text_features(self, text: str) -> Optional[np.ndarray]:
        clean_text = self._clean_text(text)
        text_embedding = self._extract_distilbert_features_batch([clean_text])
        if text_embedding is not None and len(text_embedding) > 0:
            return text_embedding[0]
        else:
            return None

    def recommend_by_image(self, query_image_url, top_k=10, exclude_self=True):
        is_in_db = query_image_url in self.df['imageurl'].values
        query_idx = None

        if is_in_db:
            query_idx = self.df[self.df['imageurl'] == query_image_url].index[0]
            query_descriptor = self.visual_descriptors_norm[query_idx:query_idx+1]
        else:
            query_descriptor = self._extract_external_visual_features(query_image_url)
            if query_descriptor is None:
                return None
            query_descriptor = normalize(query_descriptor.reshape(1, -1).astype(np.float32))

        similarities = cosine_similarity(query_descriptor, self.visual_descriptors_norm)[0]

        if exclude_self and query_idx is not None:
            similarities[query_idx] = -1

        top_indices = similarities.argsort()[-top_k:][::-1]
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        results['rank'] = range(1, len(results) + 1)
        return results

    def recommend_by_text(self, query_text, top_k=10):
        query_text_clean = self._clean_text(query_text)
        query_descriptor = self._extract_external_text_features(query_text_clean)
        if query_descriptor is None:
            return None

        query_descriptor = normalize(query_descriptor.reshape(1, -1).astype(np.float32))
        similarities = cosine_similarity(query_descriptor, self.text_descriptors_norm)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        results['rank'] = range(1, len(results) + 1)
        return results

    def recommend_multimodal(self, query_image_url=None, query_text=None,
                            alpha=0.5, top_k=10, exclude_self=True):
        if query_image_url is None and query_text is None:
            st.error("Erreur : au moins une modalité (image ou texte) doit être fournie")
            return None

        similarities = np.zeros(len(self.df), dtype=np.float32)
        query_idx = None

        if query_image_url is not None:
            is_in_db = query_image_url in self.df['imageurl'].values

            if is_in_db:
                query_idx = self.df[self.df['imageurl'] == query_image_url].index[0]
                query_visual = self.visual_descriptors_norm[query_idx:query_idx+1]
            else:
                query_visual = self._extract_external_visual_features(query_image_url)
                if query_visual is not None:
                    query_visual = normalize(query_visual.reshape(1, -1).astype(np.float32))

            if query_visual is not None:
                visual_sim = cosine_similarity(query_visual, self.visual_descriptors_norm)[0]
                similarities += alpha * visual_sim

        if query_text is not None:
            query_text_clean = self._clean_text(query_text)
            query_textual = self._extract_external_text_features(query_text_clean)

            if query_textual is not None:
                query_textual = normalize(query_textual.reshape(1, -1).astype(np.float32))
                text_sim = cosine_similarity(query_textual, self.text_descriptors_norm)[0]
                similarities += (1 - alpha) * text_sim

        if exclude_self and query_idx is not None:
            similarities[query_idx] = -1

        top_indices = similarities.argsort()[-top_k:][::-1]
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        results['rank'] = range(1, len(results) + 1)
        return results


# -----------------------------------------------------------------------------------
# Chargement des données et initialisation du système de recommandation
# -----------------------------------------------------------------------------------
@st.cache_resource
def load_data_and_recommender():
    try:
        df_products_recom = pd.read_csv('products_database_recom.csv')
        visual_descriptors_recom = np.load('visual_descriptors_recom.npy')
        text_descriptors_recom = np.load('text_descriptors_recom.npy')

        recommender_system = MultimodalRecommendationSystem(
            df_products_recom,
            visual_descriptors_recom,
            text_descriptors_recom
        )
        return df_products_recom, recommender_system
    except Exception as e:
        st.error(f"Erreur lors du chargement des données ou du système de recommandation : {e}")
        st.stop()

df_products, recommender = load_data_and_recommender()


# -----------------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Chanel Recommender")

st.title(" Système de Recommandation Multimodal pour Produits Chanel")
st.write("Recommandations basées sur des requêtes textuelles, des images ou une combinaison des deux.")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres de Recommandation")
selected_mode = st.sidebar.radio("Choisissez un mode de recherche",
                                 ("Texte", "Image", "Multimodal (Texte + Image)"))
top_k = st.sidebar.slider("Nombre de recommandations à afficher", 1, 20, 5)


results = None
query_info = ""

if selected_mode == "Texte":
    st.header("Recommandation par Texte")
    query_text = st.text_input("Entrez une description de produit (ex: 'sac en cuir noir élégant')", "sac en cuir noir élégant")
    if st.button("Recommander par Texte"):
        if query_text:
            with st.spinner("Recherche de recommandations textuelles..."):
                results = recommender.recommend_by_text(query_text, top_k=top_k)
                query_info = f"Texte: '{query_text}'"
        else:
            st.warning("Veuillez entrer un texte pour la recherche.")

elif selected_mode == "Image":
    st.header("Recommandation par Image")
    image_url_option = st.radio("Source de l'image", ("Sélectionner une image existante", "URL d'une image externe"))

    query_image_url = None
    if image_url_option == "Sélectionner une image existante":
        st.subheader("Choisissez un produit de la base de données")
        product_titles = df_products['title'].tolist()
        selected_title = st.selectbox("Produit", product_titles)
        if selected_title:
            query_image_url = df_products[df_products['title'] == selected_title]['imageurl'].iloc[0]
            st.image(query_image_url, caption=selected_title, width=200)
    else:
        query_image_url = st.text_input("Entrez l'URL d'une image (ex: https://example.com/image.jpg)")
        if query_image_url:
            try:
                response = requests.get(query_image_url, timeout=5)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption='Image externe', width=200)
            except Exception:
                st.error("URL d'image invalide ou impossible de charger l'image.")
                query_image_url = None

    if st.button("Recommander par Image"):
        if query_image_url:
            with st.spinner("Recherche de recommandations visuelles..."):
                results = recommender.recommend_by_image(query_image_url, top_k=top_k)
                query_info = f"Image: '{query_image_url[:50]}...'"
        else:
            st.warning("Veuillez fournir une image pour la recherche.")

elif selected_mode == "Multimodal (Texte + Image)":
    st.header("Recommandation Multimodale")

    # Entrée texte
    query_text_multi = st.text_input("Entrez une description (ex: 'bijou or')", "bijou en or")

    # Entrée image
    st.subheader("Sélectionnez une image (facultatif)")
    image_url_option_multi = st.radio("Source de l'image pour multimodal", ("Aucune image", "Sélectionner une image existante", "URL d'une image externe"))
    query_image_url_multi = None

    if image_url_option_multi == "Sélectionner une image existante":
        product_titles_multi = df_products['title'].tolist()
        selected_title_multi = st.selectbox("Produit (Image)", product_titles_multi)
        if selected_title_multi:
            query_image_url_multi = df_products[df_products['title'] == selected_title_multi]['imageurl'].iloc[0]
            st.image(query_image_url_multi, caption=selected_title_multi, width=150)
    elif image_url_option_multi == "URL d'une image externe":
        query_image_url_multi = st.text_input("URL d'image (multimodal)")
        if query_image_url_multi:
            try:
                response = requests.get(query_image_url_multi, timeout=5)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption='Image externe (multimodal)', width=150)
            except Exception:
                st.error("URL d'image invalide ou impossible de charger l'image.")
                query_image_url_multi = None

    alpha = st.sidebar.slider("Poids de l'image (alpha)", 0.0, 1.0, 0.5)

    if st.button("Recommander en Multimodal"):
        if not query_text_multi and not query_image_url_multi:
            st.warning("Veuillez fournir au moins un texte ou une image pour la recherche multimodale.")
        else:
            with st.spinner("Recherche de recommandations multimodales..."):
                results = recommender.recommend_multimodal(
                    query_image_url=query_image_url_multi,
                    query_text=query_text_multi,
                    alpha=alpha,
                    top_k=top_k
                )
                query_info = f"Multimodal - Texte: '{query_text_multi}', Image: '{'Oui' if query_image_url_multi else 'Non'}'"

# Affichage des résultats
st.markdown("--- ")
st.header(" Résultats de la Recommandation")

if results is not None and not results.empty:
    st.subheader(f"Top {len(results)} Recommandations pour {query_info}")

    # Créer une disposition en grille pour les produits
    cols = st.columns(min(len(results), 5)) # Afficher max 5 colonnes

    for i, (idx, product) in enumerate(results.iterrows()):
        with cols[i % 5]: # Utilise le modulo pour enrouler sur les colonnes
            st.markdown(f"**Rank {int(product['rank'])}**")
            st.markdown(f"**Score: {product['similarity_score']:.3f}**")
            st.markdown(f"**{product['category2_code']}**")
            st.markdown(f"{product['price_eur']:.0f}€")
            st.caption(product['title'])

            try:
                response = requests.get(product['imageurl'], timeout=5)
                img = Image.open(BytesIO(response.content))
                st.image(img, use_column_width=True)
            except Exception:
                st.warning("Image non disponible")
else:
    st.info("Aucune recommandation pour le moment. Essayez une autre recherche !")

