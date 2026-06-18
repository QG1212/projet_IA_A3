import streamlit as st
import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import streamlit.components.v1 as components

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse de Clusters - Bornes FR", layout="wide")

st.title("Application de Clustering - Bornes en France Métropolitaine")


# --- 1. CHARGEMENT ET NETTOYAGE DES DONNÉES (PARTAGÉ OU ACCESSIBLE PAR TOUS LES BESOINS) ---
@st.cache_data
def load_data():
    # Ajustement du chemin vers le fichier CSV
    try:
        df = pd.read_csv('../donnees_nettoyees_excel_fin.csv', sep=';', encoding='latin-1', low_memory=False)
    except FileNotFoundError:
        df = pd.read_csv('donnees_nettoyees_excel_fin.csv', sep=';', encoding='latin-1', low_memory=False)

    # Conversion et nettoyage
    df['lon'] = pd.to_numeric(df['consolidated_longitude'].astype(str).str.replace(',', '.'), errors='coerce')
    df['lat'] = pd.to_numeric(df['consolidated_latitude'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['lon', 'lat'])
    df = df[(df['lat'] >= 41.3) & (df['lat'] <= 51.1) & (df['lon'] >= -5.2) & (df['lon'] <= 9.6)]
    return df


with st.spinner("Chargement et traitement des données géographiques..."):
    df = load_data()

X = df[['lon', 'lat']].values


# --- FONCTION DE CALCUL DES SCORES (MISE EN CACHE) ---
@st.cache_data
def calculer_scores_performance(X_values):
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_values)
        inertias.append(kmeans.inertia_)

        if k > 1:
            if len(X_values) > 5000:
                np.random.seed(42)
                indices = np.random.choice(X_values.shape[0], 5000, replace=False)
                score = silhouette_score(X_values[indices], labels[indices])
            else:
                score = silhouette_score(X_values, labels)
            silhouette_scores.append(score)
            calinski_scores.append(calinski_harabasz_score(X_values, labels))
            davies_scores.append(davies_bouldin_score(X_values, labels))

    return inertias, silhouette_scores, calinski_scores, davies_scores


# --- NAVIGATION : SÉLECTION DU BESOIN ---
st.sidebar.header("Navigation")
choix_besoin = st.sidebar.selectbox(
    "Choisissez le besoin à exécuter :",
    ["Besoin 1", "Besoin 2", "Besoin 3", "Besoin 4"],
    index=1  # Sélectionne le Besoin 2 par défaut
)

st.sidebar.markdown("---")

#besoin 1
if choix_besoin == "Besoin 1":
    st.header("Analyse - Besoin 1")
    st.write("Insérez ici le code ou les composants spécifiques au Besoin 1.")
    st.info("Emplacement libre pour le code du Besoin 1.")


#besoin 2
elif choix_besoin == "Besoin 2":
    st.write(
        "Cette application analyse la répartition géographique de vos données et détermine les centres de clusters optimaux.")

    # --- 2. BARRE LATÉRALE SPECIFIQUE AU BESOIN 2 ---
    st.sidebar.header("Vos Coordonnées (Besoin 2)")
    st.sidebar.write("Entrez un point en France métropolitaine.")

    user_lat = st.sidebar.number_input("Latitude (41.3 à 51.1)", min_value=41.3, max_value=51.1, value=46.5, step=0.1)
    user_lon = st.sidebar.number_input("Longitude (-5.2 à 9.6)", min_value=-5.2, max_value=9.6, value=2.5, step=0.1)

    activer_point = st.sidebar.checkbox("Afficher mon point personnalisé", value=True)

    if not activer_point:
        user_lat, user_lon = None, None

    tab1, tab2 = st.tabs(["Évaluation Mathématique", "Carte Interactive"])

    # --- ONGLET 1 : ANALYSE DES SCORES ---
    with tab1:
        st.header("Analyse de la performance du K-Means")
        st.write("Cochez l'option ci-dessous pour générer et afficher les graphiques d'optimisation.")

        visualiser_graphes = st.checkbox("Visualiser les graphiques de performance (K = 1 à 10)")

        if visualiser_graphes:
            with st.spinner("Calcul des scores en cours (méthode du coude, silhouette...)..."):
                inertias, silhouette_scores, calinski_scores, davies_scores = calculer_scores_performance(X)
                K_range = range(1, 11)

                # Génération des graphiques
                fig, axs = plt.subplots(2, 2, figsize=(14, 9))
                k_opt = 5

                # 1. Coude
                axs[0, 0].plot(K_range, inertias, marker='o', linestyle='--', color='b')
                axs[0, 0].plot(k_opt, inertias[k_opt - 1], 'ro', markersize=8)
                axs[0, 0].set_title('1. Méthode du Coude (Inertie)')
                axs[0, 0].grid(True)

                # 2. Silhouette
                axs[0, 1].plot(range(2, 11), silhouette_scores, marker='s', linestyle='--', color='g')
                axs[0, 1].plot(k_opt, silhouette_scores[k_opt - 2], 'ro', markersize=8)
                axs[0, 1].set_title('2. Méthode de la Silhouette')
                axs[0, 1].grid(True)

                # 3. Calinski-Harabasz
                axs[1, 0].plot(range(2, 11), calinski_scores, marker='^', linestyle='--', color='orange')
                axs[1, 0].plot(k_opt, calinski_scores[k_opt - 2], 'ro', markersize=8)
                axs[1, 0].set_title('3. Indice Calinski-Harabasz')
                axs[1, 0].grid(True)

                # 4. Davies-Bouldin
                axs[1, 1].plot(range(2, 11), davies_scores, marker='v', linestyle='--', color='r')
                axs[1, 1].plot(k_opt, davies_scores[k_opt - 2], 'ro', markersize=8)
                axs[1, 1].set_title('4. Indice Davies-Bouldin')
                axs[1, 1].grid(True)

                plt.tight_layout()
                st.pyplot(fig)
                st.success("Analyses graphiques générées avec succès.")
        else:
            st.info("Cochez la case ci-dessus pour lancer les calculs et afficher les graphiques.")

    #carte interactive
    with tab2:
        st.header("Visualisation des Centres de Clusters")

        nombre_clusters = st.slider("Sélectionnez le nombre de clusters (K) pour la carte :", min_value=2, max_value=10,
                                    value=5)

        with st.spinner("Modélisation géospatiale en cours..."):
            kmeans = KMeans(n_clusters=nombre_clusters, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(X)
            centres = kmeans.cluster_centers_

            idx_centroide_proche = None
            if user_lat is not None and user_lon is not None:
                distances = [np.sqrt((c[0] - user_lon) ** 2 + (c[1] - user_lat) ** 2) for c in centres]
                idx_centroide_proche = np.argmin(distances)
                st.success(
                    f"Le centre de cluster le plus proche de votre position est le Cluster n°{idx_centroide_proche}.")

            centre_lat, centre_lon = df['lat'].mean(), df['lon'].mean()
            carte = folium.Map(location=[centre_lat, centre_lon], zoom_start=6)

            for i, centre in enumerate(centres):
                nb_bornes = int((df['cluster'] == i).sum())

                if i == idx_centroide_proche:
                    couleur_marqueur = 'orange'
                    texte_popup = f"<b>CENTRE DU CLUSTER {i}</b><br>Nombre de bornes : {nb_bornes:,}<br><b>(LE PLUS PROCHE DE VOUS)</b>"
                else:
                    couleur_marqueur = 'black'
                    texte_popup = f"<b>CENTRE DU CLUSTER {i}</b><br>Nombre de bornes : {nb_bornes:,}"

                folium.Marker(
                    location=[centre[1], centre[0]],
                    popup=texte_popup,
                    icon=folium.Icon(color=couleur_marqueur, icon='info-sign')
                ).add_to(carte)

            if user_lat is not None and user_lon is not None:
                folium.Marker(
                    location=[user_lat, user_lon],
                    popup="<b>Votre point personnalisé</b>",
                    icon=folium.Icon(color='darkgreen', icon='home')
                ).add_to(carte)

            components.html(carte._repr_html_(), height=600)


#code  besoin 3
elif choix_besoin == "Besoin 3":
    st.header("Analyse - Besoin 3")
    st.write("Insérez ici le code ou les composants spécifiques au Besoin 3.")
    st.info("Emplacement libre pour le code du Besoin 3.")

#code besoin 4
elif choix_besoin == "Besoin 4":
    st.header("Analyse - Besoin 4")
    st.write("Insérez ici le code ou les composants spécifiques au Besoin 4.")
    st.info("Emplacement libre pour le code du Besoin 4.")