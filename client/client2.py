import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Chargement des données
df = pd.read_csv('../donnees_nettoyees_excel_fin.csv', sep=';', encoding='latin-1', low_memory=False)

# Correction des virgules résiduelles dans le fichier et conversion en nombres
df['lon'] = pd.to_numeric(df['consolidated_longitude'].astype(str).str.replace(',', '.'), errors='coerce')
df['lat'] = pd.to_numeric(df['consolidated_latitude'].astype(str).str.replace(',', '.'), errors='coerce')

# FILTRAGE CRUCIAL : On élimine les points vides et hors de la France métropolitaine
df = df.dropna(subset=['lon', 'lat'])
df = df[(df['lat'] >= 41.3) & (df['lat'] <= 51.1) & (df['lon'] >= -5.2) & (df['lon'] <= 9.6)]

# Extraction des coordonnées propres pour le clustering
X = df[['lon', 'lat']].values

# Demande à l'utilisateur de rentrer des coordonnées
print("Saisie de vos coordonnées (France métropolitaine uniquement)")
user_lat = None
user_lon = None

# Un seul nom de fichier fixe pour éviter l'accumulation de fichiers HTML
nom_fichier = "carte_clusters_FR.html"

try:
    input_lat = input("Entrez la latitude (entre 41.3 et 51.1) : ").replace(',', '.')
    input_lon = input("Entrez la longitude (entre -5.2 et 9.6) : ").replace(',', '.')

    val_lat = float(input_lat)
    val_lon = float(input_lon)

    # Validation stricte selon les limites géographiques de la métropole
    if (41.3 <= val_lat <= 51.1) and (-5.2 <= val_lon <= 9.6):
        user_lat = val_lat
        user_lon = val_lon
        print("Coordonnées valides en France métropolitaine.")
    else:
        print("Le point demandé se trouve en dehors de la France métropolitaine.")
        print("Affichage de la carte par défaut (sans votre marqueur).")
except ValueError:
    print("Saisie invalide ou vide. Affichage de la carte par défaut (sans votre marqueur).")

print("Calcul des clusters et génération de la carte...")

# --- CONFIGURATION DES TESTS DE CLUSTERING ---
inertias = []
silhouette_scores = []
calinski_scores = []
davies_scores = []
K_range = range(1, 11)

# Teste de 1 à 10 groupes pour trouver la configuration idéale
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)

    # Calcule les autres indicateurs de performance (Silhouette, etc.)
    if k > 1:
        if len(X) > 5000:
            np.random.seed(42)
            indices = np.random.choice(X.shape[0], 5000, replace=False)
            score = silhouette_score(X[indices], labels[indices])
        else:
            score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        calinski_scores.append(calinski_harabasz_score(X, labels))
        davies_scores.append(davies_bouldin_score(X, labels))

# Affichage des 4 graphiques de performance
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
k_opt = 5

# Graphique 1 : Le Coude
axs[0, 0].plot(K_range, inertias, marker='o', linestyle='--', color='b')
axs[0, 0].plot(k_opt, inertias[k_opt - 1], 'ro', markersize=8)
axs[0, 0].set_xlabel('Nombre de clusters (K)')
axs[0, 0].set_ylabel('Inertie')
axs[0, 0].set_title('1. Méthode du Coude')
axs[0, 0].grid(True)

# Graphique 2 : La Silhouette
axs[0, 1].plot(range(2, 11), silhouette_scores, marker='s', linestyle='--', color='g')
axs[0, 1].plot(k_opt, silhouette_scores[k_opt - 2], 'ro', markersize=8)
axs[0, 1].set_xlabel('Nombre de clusters (K)')
axs[0, 1].set_ylabel('Score de Silhouette')
axs[0, 1].set_title('2. Méthode de la Silhouette')
axs[0, 1].grid(True)

# Graphique 3 : Calinski-Harabasz
axs[1, 0].plot(range(2, 11), calinski_scores, marker='^', linestyle='--', color='orange')
axs[1, 0].plot(k_opt, calinski_scores[k_opt - 2], 'ro', markersize=8)
axs[1, 0].set_xlabel('Nombre de clusters (K)')
axs[1, 0].set_ylabel('Indice Calinski-Harabasz')
axs[1, 0].set_title('3. Indice Calinski-Harabasz')
axs[1, 0].grid(True)

# Graphique 4 : Davies-Bouldin
axs[1, 1].plot(range(2, 11), davies_scores, marker='v', linestyle='--', color='r')
axs[1, 1].plot(k_opt, davies_scores[k_opt - 2], 'ro', markersize=8)
axs[1, 1].set_xlabel('Nombre de clusters (K)')
axs[1, 1].set_ylabel('Indice Davies-Bouldin')
axs[1, 1].set_title('4. Indice Davies-Bouldin')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

#Application du K-Means final (5 clusters)
nombre_clusters = 5
kmeans = KMeans(n_clusters=nombre_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

#Création de la carte Folium
print("Génération de la carte Folium...")
centre_lat, centre_lon = df['lat'].mean(), df['lon'].mean()
carte = folium.Map(location=[centre_lat, centre_lon], zoom_start=6)

# Récupération des centres de clusters
centres = kmeans.cluster_centers_

# Calcul du centre le plus proche si l'utilisateur est localisé
idx_centroide_proche = None
if user_lat is not None and user_lon is not None:
    distances = [np.sqrt((c[0] - user_lon) ** 2 + (c[1] - user_lat) ** 2) for c in centres]
    idx_centroide_proche = np.argmin(distances)

for i, centre in enumerate(centres):
    # Compte du nombre de bornes associées à ce cluster
    nb_bornes = int((df['cluster'] == i).sum())

    if i == idx_centroide_proche:
        couleur_marqueur = 'orange'
        texte_popup = f"<b>CENTRE DU CLUSTER {i}</b><br>Nombre de bornes : {nb_bornes:,}<br><b>(LE PLUS PROCHE DE VOUS)</b>".replace(',', ' ')
    else:
        couleur_marqueur = 'black'
        texte_popup = f"<b>CENTRE DU CLUSTER {i}</b><br>Nombre de bornes : {nb_bornes:,}".replace(',', ' ')

    folium.Marker(
        location=[centre[1], centre[0]],
        popup=texte_popup,
        icon=folium.Icon(color=couleur_marqueur, icon='info-sign')
    ).add_to(carte)

# Ajout du marqueur utilisateur (si valide)
if user_lat is not None and user_lon is not None:
    folium.Marker(
        location=[user_lat, user_lon],
        popup="<b>Votre point personnalisé</b>",
        icon=folium.Icon(color='darkgreen', icon='home')
    ).add_to(carte)

# Enregistrement du fichier HTML final
carte.save(nom_fichier)
print(f"La carte a été mise à jour avec succès dans : '{nom_fichier}'")