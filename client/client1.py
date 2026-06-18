
import pandas as pd
import numpy as np

df = pd.read_csv(
    "/content/drive/MyDrive/TP_ia/projet_ia/donnees_nettoyees.csv",
    sep=",",
    encoding="utf-8",
)
#df= pd.read_csv(
 #   "/content/drive/MyDrive/TP_ia/projet_ia/donnees_nettoyees_excel_fin2.csv",
  #  sep=";",
   # encoding="latin1",
    #on_bad_lines="skip",
    #low_memory=False
#)

#IMPLANT_VALIDES = [
 #   "Voirie",
  #  "Parking privé à usage public",
   # "Parking public",
    #"Station dédiée à la recharge rapide",
    #"Parking privé réservé à la clientèle",
#]

#df = df[df["implantation_station"].isin(IMPLANT_VALIDES)]

#/content/drive/MyDrive/TP_ia/projet_ia/donnees_nettoyees.csv   /content/drive/MyDrive/TP_ia/projet_ia/donnees_nettoyees_excel_fin2.csv
print("hello")
print(df)

COLS = [
    "implantation_station",
    "puissance_nominale",
    "consolidated_latitude",
    "consolidated_longitude",
    "consolidated_commune",
    "consolidated_code_postal",
    "nbre_pdc",
    "condition_acces",
]

data = df[COLS].copy()

cols = ["consolidated_latitude", "consolidated_longitude", "puissance_nominale"]

#remplace les virgules par des points
df[cols] = df[cols].replace(",", ".", regex=True)
# en float
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

df = df[df["consolidated_latitude"].notna() & df["consolidated_longitude"].notna()]

from sklearn.preprocessing import LabelEncoder

#encodage des colonnes
le_impl = LabelEncoder()
df["implantation_station_enc"] = le_impl.fit_transform(df["implantation_station"])

le_acces = LabelEncoder()
df["condition_acces_enc"] = le_acces.fit_transform(df["condition_acces"].fillna("non renseigné"))

# affichage les correspondances
print("encodage implantation_station :")
print("\n".join(f"  {code} → {cls}" for code, cls in enumerate(le_impl.classes_)))

print("\n encodage condition_acces :")
print("\n".join(f"  {code} → {cls}" for code, cls in enumerate(le_acces.classes_)))

from folium.plugins import HeatMap, MarkerCluster
import folium
#cas ou est la station
#base de la carte centrée sur la France
center = [46.2276, 2.2137]
#carte = folium.Map(location=center, zoom_start=6, tiles="OpenStreetMap") me met des errreurs
carte = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")
#eviter que ca crach
df_ech= df.sample(n=min(45000, len(df)), random_state=42)

COLOR_MAP = {
    "Voirie": "pink",
    "Parking privé à usage public": "violet",
    "Parking public": "blue",
    "Station dédiée à la recharge rapide": "red",
    "Parking privé réservé à la clientèle": "yellow",
}

# Heatmap
data_points = df_ech[["consolidated_latitude", "consolidated_longitude"]].values.tolist()
HeatMap(data_points, radius=12, blur=18, min_opacity=0.4).add_to(carte)
carte.save("/content/drive/MyDrive/TP_ia/projet_ia/carte_heatmap.html")

# clusters
cluster = MarkerCluster().add_to(carte)
for col, row in df_ech.iterrows():
    implant = row["implantation_station"]
    couleur = COLOR_MAP.get(implant, "gray")  # attribue la couleur selon le type


    p = row["puissance_nominale"]
    if pd.notna(p):
    # si p est un vrai chiffre
      p_str = f"{p:.0f} kw"  
    else:
    # si p est un NaN ou manquant
      p_str = "N/A"  # ="N/A"

    folium.CircleMarker(location=[row["consolidated_latitude"], row["consolidated_longitude"]],radius=4,color=couleur,fill=True,fill_color=couleur,fill_opacity=0.7,weight=0.5,
        #  affiche le type et la puissance quand tu click sur un point en html
        popup=folium.Popup( f"Type : {implant}Puissance : {p_str}", max_width=200 )).add_to(cluster)


carte.save("/content/drive/MyDrive/TP_ia/projet_ia/cartecluster2.html")

#cas de la puissance

center = [46.2276, 2.2137]
carte = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

# (limite vu que se lance pas sinon
df_ech = df.sample(n=min(5000, len(df)), random_state=42).copy()

# couleur=puissance
def attribuer_couleur_puissance(puissance):
    if pd.isna(puissance):
        return "pink"
    elif puissance <= 7:
        return "blue"
    elif puissance <= 22:
        return "violet"
    elif puissance <= 50:
        return "orange"
    else:
        return "red"



# cluster
cluster = MarkerCluster().add_to(carte)
for col, row in df_ech.iterrows():
    p = row["puissance_nominale"]
    couleur = attribuer_couleur_puissance(p)
    implant = row["implantation_station"]

    folium.CircleMarker(
        location=[row["consolidated_latitude"], row["consolidated_longitude"]],
        radius=4,
        color=couleur,
        fill=True,
        fill_color=couleur,
        fill_opacity=0.7,
        weight=0.5,).add_to(cluster)

carte.save("/content/drive/MyDrive/TP_ia/projet_ia/carte_puis.html")

import matplotlib.pyplot as plt

# enlève Na
df_clean = df.dropna(subset=["nbre_pdc", "implantation_station"])

# 2. Préparation des groupes
groupes = df_clean.groupby("implantation_station")["nbre_pdc"]
donnees = [groupe.values for nom, groupe in groupes]
labels = [nom for nom, groupe in groupes]

# 3. Dessin du boxplot (avec 'tick_labels' pour éviter le Warning)
plt.figure(figsize=(10, 6))
plt.boxplot(donnees,
  tick_labels=labels,  #erreur sinon
    patch_artist=True,showfliers=False, boxprops=dict(facecolor="lightblue"))

# 4. Titres et mise en page
plt.title("implantation vs nb points de charge")
plt.xlabel("implantation")
plt.ylabel("nb de pdc")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.7)

plt.tight_layout()
plt.show()

df_clean = df.dropna(subset=["puissance_nominale", "implantation_station"])


groupes = df_clean.groupby("implantation_station")["puissance_nominale"]
donnees = [groupe.values for nom, groupe in groupes]
labels = [nom for nom, groupe in groupes]

#  Dessin du boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(donnees,tick_labels=labels,patch_artist=True,
    showfliers=False,          #cache les points extrêmes
)

# 4. Titres et mise en page
plt.title("implantation vs puissance nominale")
plt.xlabel("implantation")
plt.ylabel("puissance")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

df_clean = df.dropna(subset=["consolidated_longitude", "consolidated_latitude", "nbre_pdc"])
plt.figure(figsize=(9, 8))

# geom_point(alpha = 0.5) + scale_color_gradient(low = "blue", high = "red")
scatter = plt.scatter(
    x=df_clean["consolidated_longitude"],
    y=df_clean["consolidated_latitude"],
    c=df_clean["nbre_pdc"],
    cmap="viridis",              
    alpha=0.5,
    s=5                           # Taille des points pour garder la carte lisible
)

#barre de légende pour la couleur
cbar = plt.colorbar(scatter)
cbar.set_label("nb de PDC")

plt.title("localisation vs Nombre de points de charge")
plt.xlabel("longitude")
plt.ylabel("latitude")