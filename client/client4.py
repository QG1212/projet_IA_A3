import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score


#ouvre la data
df = pd.read_csv(
    "/content/drive/MyDrive/TP_ia/projet_ia/donnees_nettoyees.csv",
    sep=",",
    encoding="utf-8",
)
print(df)

#gest puissance
df = df.dropna(subset=["puissance_nominale"])
df = df[df["puissance_nominale"] > 0]

#definition des bornes
bornes = [0, 8, 22, 50, np.inf]  # np.inf permet de couvrir tout ce qui est strictement supérieur à 50
labels = ["lente", "standard", "rapide", "ultra-rapide"]

# couper
df["classe_puissance"] = pd.cut( df["puissance_nominale"], bins=bornes, labels=labels, right=True)

# distribution
dist = df["classe_puissance"].value_counts()
pourcentages = df["classe_puissance"].value_counts(normalize=True) * 100


#colones interesantes, divider par categorie pour apres
col_CAT  = ["implantation_station", "condition_acces"]
col_BOOL = [ "prise_type_ef", "prise_type_2", "prise_type_combo_ccs", "prise_type_chademo", "prise_type_autre", "paiement_acte", "paiement_cb", "paiement_autre", "reservation"
]
col_NUM  = ["nbre_pdc", "consolidated_longitude", "consolidated_latitude"]
targ_continu = "puissance_nominale"
targ_classif = "classe_puissance"
tout = col_CAT + col_BOOL + col_NUM


df_model = df[tout + [targ_continu, targ_classif]].copy()
#remplacer les na par la mediane que moyenne car elle est robuste aux valeurs extrêmes
df_model["nbre_pdc"] = df_model["nbre_pdc"].fillna(df_model["nbre_pdc"].median())

# evite le crash des modèles
df_model["paiement_acte"] = df_model["paiement_acte"].fillna(False).astype(bool)
df_model.loc[df_model["paiement_acte"] == False, ["paiement_cb", "paiement_autre"]] = False
df_model["paiement_cb"] = df_model["paiement_cb"].fillna(False).astype(bool)
df_model["paiement_autre"] = (
    df_model["paiement_autre"].fillna(False).astype(bool))



# Test Kruskal-wallis (quali/bool vs puissance)

print("analyse statistique kruskal wallis" )

resultats = []
for col in col_BOOL + col_CAT:
    df_sub = df[[col, targ_continu]].dropna()
    #diviser les données numériques en autant de groupes qu'il y a de catégories.
    groupes = []
    for v in df_sub[col].unique():
        #on filtre les lignes, on garde la colonne cible, et on extrait les valeurs
        valeurs_du_groupe = df_sub[df_sub[col] == v][targ_continu].values
        # on vérifie que le groupe n'est pas vide avant de l'ajouter
        if len(valeurs_du_groupe) > 0:
            groupes.append(valeurs_du_groupe)

    if len(groupes) > 1:
        stat, p = kruskal(*groupes)
    else:
        stat = 0
        p = 1.0
    resultats.append(
        {
            "Variable": col,
            "Test": "kruskal-wallis",
            "Stat": round(stat, 2),   #plus elle est grande, plus les groupes comparés ont des distributions de puissance différentes.
            "p-value": f"{p:.2e}",
            "Significatif": "v" if p < 0.05 else "x",
        }
    )
    print(f"  {col:<35} KW={stat:>10.1f}  p={p:.2e}")

#  tests spearman (num et géo vs puissance)
print("\n" )
print("analyse stat  SPEARMAN )")
print("\n")

for col in col_NUM + col_NUM:
    df_sub = df[[col, targ_continu]].dropna()
    stat, p = spearmanr(df_sub[col], df_sub[targ_continu])

    resultats.append(
        {
            "Variable": col,
            "Test": "Spearman",
            "Stat": round(stat, 4),   #proche de 1 quand la variable augmente, la puissance augmente aussi , 0pas de relation
            "p-value": f"{p:.2e}",
            "Significatif": "v" if p < 0.05 else "x",
        }
    )
    print(f"  {col:<35} Rho={stat:>10.4f}  p={p:.2e}")

# Tableau récap global
recap = pd.DataFrame(resultats)
print("\n" )
print("tableau recap")
print("\n" )
print(
    recap[["Test", "Variable", "Stat", "p-value", "Significatif"]].to_string(
        index=False
    )
)

#  Graph1:boxplots
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(
    "impact des connecteurs sur la puissance (kW)", fontweight="bold"
)

prises_interet = [ "prise_type_combo_ccs", "prise_type_chademo","prise_type_2","prise_type_ef",]
for i in range(len(prises_interet)):
    ax = axes[i]
    col = prises_interet[i]
    sns.boxplot(data=df, x=col, y=targ_continu, ax=ax, palette="Set2", showfliers=False)
    ax.set_title(col.replace("_", " "), fontsize=10, fontweight="bold")
    ax.set_xlabel("0=Non  1=Oui")
    ax.set_ylabel("puissance kW" if ax == axes[0] else "")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("b4_boxplots_prises.png", dpi=150)
plt.close()

#Graph2
plt.figure(figsize=(12, 6))
sns.boxplot(data=df,x="implantation_station",y=targ_continu,palette="viridis",showfliers=False,)
plt.title("puissance par type d'implantation", fontweight="bold")
plt.xticks(rotation=25, ha="right")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("b4_boxplot_implantation.png", dpi=150)
plt.close()

# Graph 3 carte
plt.figure(figsize=(10, 8))

df_geo = df[col_NUM + [targ_continu]].dropna()
sc = plt.scatter(
    df_geo["consolidated_longitude"],
    df_geo["consolidated_latitude"],
    c=df_geo[targ_continu],
    cmap="turbo",
    alpha=0.6,
    s=1,
    vmin=0,
    vmax=150,
)
plt.colorbar(sc, label="Puissance (kW) — tronqué à 150")
plt.title("Carte de la puissance des bornes en France", fontweight="bold")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim(-5.5, 10)  # on apparait sur la france
plt.ylim(41, 51.5)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("b4_carte_puissance.png", dpi=150)
plt.close()



X = df_model[tout]
y = df_model[targ_classif]


enco=LabelEncoder()
y_encoded = enco.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


#  OneHotEncoder pour les catégories au lieu de LabelEncoder
preprocessor = ColumnTransformer(
    transformers=[
        (
            "encode",# nom du transformateur
            OneHotEncoder(drop="first", # supprime la 1ere colonne pour éviter la redondance
                handle_unknown="ignore"# ignore les  inconnues pour prédiction
            ), col_CAT,   # appliqué uniquement sur les colonnes avec nom
        ),("num", #nom
            StandardScaler(), # centre et réduit (moyenne=0, écart-type=1)
            col_NUM, # que nb
        ),
    ],
    remainder="passthrough", # les booléens passent
)

#change nom en nb
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# recup le nom des features générées
feature_names = preprocessor.get_feature_names_out()


print(" GridSearch RANDOM FOREST")


# grille d'hyperparamètres bien dimensionnée (12 combinaisons au total)
param_rf = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"],
}

# VAL croisée stratifiée pour gerer le déséquilibre des classes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),param_rf,  cv=cv,
    scoring="f1_weighted",# critere de reussite
    n_jobs=-1,  #  plus rapide
    verbose=1,
)


grid_rf.fit(X_train_processed, y_train)

#garde le meilleur modèle
best_rf = grid_rf.best_estimator_


print(f"\n  meilleurs hyperparamètres: {grid_rf.best_params_}")
print(f"  meilleur F1-Score: {grid_rf.best_score_:.4f}")


print("GridSearchCV decision tree")

# grille d'hyperparamètres
param_dt = {
    "max_depth": [None, 5, 10, 20],       #teste s'il faut stopper la croissance de l'arbre
    "min_samples_split": [2, 5, 10],       # nb minimum d'individus pour créer une nouvelle branche
    "class_weight": ["balanced"],         # le déséquilibre des bornes rares
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#configuration de la recherche pour l'Arbre de Décision
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid=param_dt,cv=cv,
    scoring="f1_weighted",  # Même critère de réussite (F1-score pondéré)
    n_jobs=-1,              # Utilise tous les cœurs du processeur
    verbose=1
)

grid_dt.fit(X_train_processed, y_train)

# auvegarde le meilleur arbre
best_dt = grid_dt.best_estimator_

print(f"\n  meilleurs hyperparamètres  : {grid_dt.best_params_}")
print(f"  meilleur F1-Score  : {grid_dt.best_score_:.4f}")




os.makedirs("models", exist_ok=True)
class_names = enco.classes_
modeles_a_tester = [("Random Forest", best_rf), ("Decision Tree", best_dt)]

print("score")
for nom, model in [modeles_a_tester]:
    y_pred = model.predict(X_test_processed)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n {nom}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 weighted : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))


print("\n graphique \n")

#Figure1: matrices de confusion
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("matrices de confusion", fontsize=14, fontweight="bold")



for i, (nom, model) in enumerate(modeles_a_tester):
    ax = axes[i]
    cm = confusion_matrix(y_test, model.predict(X_test_processed))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    ax.set_title(nom)
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("models/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

#Figure2 :feature (Random Forest uniquement)
importances = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
importances.plot(kind="barh", ax=ax, color="#1976D2")
ax.set_title("features random Forest", fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

#figure 3 : Comparaison des F1-scores par classe
fig, ax = plt.subplots(figsize=(10, 5))
results = {}

for name, model in [modeles_a_tester]:
    report = classification_report(y_test, model.predict(X_test_processed), target_names=class_names, output_dict=True)
    results[name] = [report[c]["f1-score"] for c in class_names]

x = np.arange(len(class_names))
w = 0.35

ax.bar(x - w/2, results["Random Forest"], w, label="random forest", color="#1976D2")
ax.bar(x + w/2, results["Decision Tree"], w, label="decision tree", color="#FF9800")

ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_ylabel("F1-score")
ax.set_ylim(0, 1.05)
ax.set_title("F1 score par classe random forest vs decision tree", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("models/f1_comparaison.png", dpi=150, bbox_inches="tight")
plt.close()


import joblib

print("\n sauvegarde ")

artefacts = {
    "best_rf": best_rf,          
    "best_dt": best_dt,          
    "preprocessor": preprocessor, 
    "enco": enco                 
}

joblib.dump(artefacts, "models/pipeline_bornes.joblib")

print("fini")