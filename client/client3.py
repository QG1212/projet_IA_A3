import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt # Importation pour les graphiques
import seaborn as sns           # Importation pour les graphiques

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Ajout de GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split # Ajout de cet import vital !
from sklearn.model_selection import GridSearchCV # Ajout de l'import pour la recherche par grille

chemin_fichier = r'C:\Users\mskit\OneDrive\Documents\donnees_nettoyees_excel.csv'

# Fonction de préparation de données
########################################

def preparer_donnees_implantation_v2(df):
    
    colonnes_numeriques = ['nbre_pdc', 'puissance_nominale', 'consolidated_longitude', 'consolidated_latitude']
    colonnes_categorielles = ['condition_acces', 'reservation', 'paiement_cb', 'prise_type_2', 'accessibilite_pmr']
    colonne_cible = 'implantation_station'
    
    toutes_colonnes_utiles = colonnes_numeriques + colonnes_categorielles + [colonne_cible]
    
    # Sélection des colonnes
    colonnes_presentes = []
    for col in toutes_colonnes_utiles:
        if col in df.columns:
            colonnes_presentes.append(col)
    
    df_filtre = df[colonnes_presentes].copy()    

    # Tri des variables
    cols_num_presentes = []
    for col in colonnes_numeriques:
        if col in df_filtre.columns:
            cols_num_presentes.append(col)
            
    # Tri des variables de catégorie
    cols_cat_presentes = []
    for col in colonnes_categorielles:
        if col in df_filtre.columns:
            cols_cat_presentes.append(col)

    # Nettoyage des nombres
    for col in cols_num_presentes:
        colonne_en_texte = df_filtre[col].astype(str)
        colonne_avec_points = colonne_en_texte.str.replace(',', '.', regex=False)
        df_filtre[col] = pd.to_numeric(colonne_avec_points, errors='coerce')   

    # Sécurisation
    for col in cols_cat_presentes:
        df_filtre[col] = df_filtre[col].astype(str)

    # Suppression ligne vide
    df_propre = df_filtre.dropna(subset=['implantation_station']).copy() # Ajout du .copy() ici pour être sûr
    
    # écrase les données par celle corrigée, en supprimant les espaces et ajoute majuscule
    df_propre[colonne_cible] = df_propre[colonne_cible].astype(str).str.strip().str.capitalize()
    
    # Élimination stricte des adresses et anomalies textuelles restantes
    # On rejette ce qui commence par un chiffre, contient des codes postaux, ou des mots clés d'adresse (route, rue, chemin...)
    mots_cles_adresses = r'^\d+|;|â|Route|Rue|Avenue|Boulevard|Chemin|Allée|Place|Zac|Frvire'
    masque_valide = ~df_propre[colonne_cible].str.contains(mots_cles_adresses, case=False, regex=True) & (df_propre[colonne_cible].str.len() > 3)
    df_propre = df_propre[masque_valide].copy()
    
    # En France métropolitaine : Longitude entre -5 et 10, Latitude entre 41 et 52
    if 'consolidated_longitude' in df_propre.columns:
        df_propre.loc[(df_propre['consolidated_longitude'] < -10) | (df_propre['consolidated_longitude'] > 20), 'consolidated_longitude'] = np.nan
    if 'consolidated_latitude' in df_propre.columns:
        df_propre.loc[(df_propre['consolidated_latitude'] < 35) | (df_propre['consolidated_latitude'] > 55), 'consolidated_latitude'] = np.nan

    # On identifie les 15 types d'implantation les plus fréquents (ou moins s'il y en a moins de valides)
    categories_valides = df_propre[colonne_cible].value_counts()
    top_15_implantations = categories_valides.nlargest(15).index
    
    # On remplace tout le reste par la catégorie "Autre"
    df_propre[colonne_cible] = df_propre[colonne_cible].apply(
        lambda x: x if x in top_15_implantations else 'Autre'
    )
    print("Nettoyage : Réduction à", df_propre[colonne_cible].nunique(), "classes pour la cible.")

    # [GRAPHIQUE 1 REAJUSTE : Distribution de la variable cible]
    plt.figure(figsize=(12, 5))
    # On filtre pour ne pas afficher de barres vides ou insignifiantes
    ordre_categories = df_propre[colonne_cible].value_counts().index
    sns.countplot(
        y=df_propre[colonne_cible], 
        order=ordre_categories, 
        palette="viridis"
    )
    plt.title("Distribution de la variable cible nettoyée")
    plt.xlabel("Nombre d'occurrences")
    plt.ylabel("Catégories d'implantation")
    plt.tight_layout()
    plt.savefig('distribution_variable_cible.png')
    plt.close()

    # Séparation du tab en x et y
    X = df_propre[cols_num_presentes + cols_cat_presentes].copy()
    y = df_propre[colonne_cible]
    
    # Encodage de la cible
    encodeur_cible = LabelEncoder()
    y_encode = encodeur_cible.fit_transform(y)
    
    # Normalisation 
    scaler = StandardScaler()
    
    for col in cols_num_presentes[:]: 
        if X[col].nunique() <= 1:
            print("Nettoyage : Suppression de la colonne constante", col)
            cols_num_presentes.remove(col)

    # On s'assure que toutes les colonnes sont bien numériques
    for col in cols_num_presentes:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # On remplace toutes les valeurs manquantes (NaN) par la médiane (très important pour les coordonnées nettoyées)
    X[cols_num_presentes] = X[cols_num_presentes].fillna(X[cols_num_presentes].median())

    # [GRAPHIQUE 2 REAJUSTE : Boxplots verticaux plus lisibles pour chaque variable]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Avant Normalisation (On utilise showfliers=False pour que l'échelle soit propre et lisible sans être écrasée par les rares outliers restants)
    sns.boxplot(data=X[cols_num_presentes], ax=ax1, palette="Set2", showfliers=False)
    ax1.set_title("Distributions AVANT Normalisation (Zoom sur la boîte)")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15)

    # Normalisation
    X_num_transforme = scaler.fit_transform(X[cols_num_presentes])
    df_num = pd.DataFrame(X_num_transforme, columns=cols_num_presentes, index=X.index)
    
    # Après Normalisation
    sns.boxplot(data=df_num, ax=ax2, palette="Set2", showfliers=False)
    ax2.set_title("Distributions APRÈS Normalisation (StandardScaler)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15)
    plt.tight_layout()
    plt.savefig('comparatif_normalisation_avant_apres.png')
    plt.close()

    # Encodage des variable de catégorie
    encodeur_categories = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_transforme = encodeur_categories.fit_transform(X[cols_cat_presentes])
    nouveaux_noms = encodeur_categories.get_feature_names_out(cols_cat_presentes)
    df_cat = pd.DataFrame(X_cat_transforme, columns=nouveaux_noms, index=X.index)
    
    # Recombinaison
    X_final = pd.concat([df_num, df_cat], axis=1)
    
    # Sauvegarde
    joblib.dump(scaler, 'modele_scaler.pkl')
    joblib.dump(encodeur_categories, 'modele_encodeur_cat.pkl')
    joblib.dump(encodeur_cible, 'modele_encodeur_cible.pkl')
    
    print("Préparation terminée, nombre de lignes conservées :", len(X_final))
    
    return X_final, y_encode


# Entrainement du modèle Random Forest
######################################

def entrainer_et_evaluer_modele(X_train, X_test, y_train, y_test):
    print("\n Entraînement du modèle de Classification > Random Forest")
    
    modele_base = RandomForestClassifier(
        class_weight='balanced', # Prise en compte du déséquilibre des classes
        n_jobs=1,                # 1 seul cœur utilisé pour stabiliser la RAM
        random_state=42
    )

    # Définition de la grille des paramètres à tester
    param_grid = {
        # Nombre d'arbre
        'n_estimators': [50, 100],
        # profondeur d'arbre 
        'max_depth': [15, 20],
        # Nombre de donnée
        'min_samples_split': [10, 15]
    }

    # Configuration du GridSearchCV ciblé sur la métrique F1 Macro
    grille_recherche = GridSearchCV(
        estimator=modele_base,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        n_jobs=1
    )

    # Lancement de l'entraînement
    grille_recherche.fit(X_train, y_train)
    print("Entraînement Random Forest terminé")
    
    # Récupération du meilleur modèle trouvé par le GridSearch
    modele = grille_recherche.best_estimator_
    print("Meilleurs paramètres trouvés (RF) :", grille_recherche.best_params_)

    print("\n Évaluation du modèle Random Forest sur les données de test ")
    
    score_train = modele.score(X_train, y_train)
    score_test = modele.score(X_test, y_test)
    print("Précision sur les données d'entraînement (RF) :", round(score_train * 100,2), "%")
    print("Précision sur les données de test (RF)        :", round(score_test * 100,2), "%")

    # Prédictions et rapport détaillé
    predictions = modele.predict(X_test)
    precision = accuracy_score(y_test, predictions)
    print(" Précision globale finale (RF)  :", round(precision * 100,2), "%")

    print("\n Rapport détaillé par catégorie (RF) :")
    print(classification_report(y_test, predictions))

    # Sauvegarde du modèle
    print("Sauvegarde du modèle Random Forest...")
    joblib.dump(modele, 'modele_random_forest.pkl')
    
    return modele


# Entrainement du modèle Gradient Boosting (Nouveau)
#####################################################

def entrainer_et_evaluer_gradient_boosting(X_train, X_test, y_train, y_test):
    print("\n Entraînement du modèle de Classification > Gradient Boosting")
    
    # Initialisation du modèle de base
    modele_base = GradientBoostingClassifier(random_state=42)

    # Définition de la grille des paramètres à tester pour le Gradient Boosting
    param_grid = {
        # Nombre d'étapes de boosting (arbres)
        'n_estimators': [50, 100],
        # Taux d'apprentissage (Learning rate)
        'learning_rate': [0.05, 0.1],
        # Profondeur maximale de chaque arbre
        'max_depth': [3, 5]
    }

    # Configuration du GridSearchCV ciblé sur la métrique F1 Macro (identique à votre choix RF)
    grille_recherche = GridSearchCV(
        estimator=modele_base,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        n_jobs=1 # Conservé à 1 pour stabiliser la RAM
    )

    # Lancement de l'entraînement
    grille_recherche.fit(X_train, y_train)
    print("Entraînement Gradient Boosting terminé")
    
    # Récupération du meilleur modèle
    modele = grille_recherche.best_estimator_
    print("Meilleurs paramètres trouvés (GB) :", grille_recherche.best_params_)

    print("\n Évaluation du modèle Gradient Boosting sur les données de test ")
    
    score_train = modele.score(X_train, y_train)
    score_test = modele.score(X_test, y_test)
    print("Précision sur les données d'entraînement (GB) :", round(score_train * 100,2), "%")
    print("Précision sur les données de test (GB)        :", round(score_test * 100,2), "%")

    # Prédictions et rapport détaillé
    predictions = modele.predict(X_test)
    precision = accuracy_score(y_test, predictions)
    print(" Précision globale finale (GB)  :", round(precision * 100,2), "%")

    print("\n Rapport détaillé par catégorie (GB) :")
    print(classification_report(y_test, predictions))

    # Sauvegarde du modèle
    print("Sauvegarde du modèle Gradient Boosting...")
    joblib.dump(modele, 'modele_gradient_boosting.pkl')
    
    return modele


if __name__ == "__main__":
    print("\n Chargement des données")
    df = pd.read_csv(chemin_fichier, sep=';', quotechar='"', low_memory=False, encoding='latin1')    
    print("Info : Nombre de lignes chargées depuis le CSV : ", len(df))
    
    print("\n Lancement du nettoyage et de la préparation explicite")
    # On récupère l'intégralité des données préparées (X_pret)
    X_pret, y_pret = preparer_donnees_implantation_v2(df)
    
    print("\n Optimisation + nettoyage")
    selector = VarianceThreshold(threshold=0.0).set_output(transform="pandas")
    X_pret_clean = selector.fit_transform(X_pret)
    print("Colonnes conservées :", X_pret_clean.shape[1] , "sur", X_pret.shape[1], "initialement")

    print("Sauvegarde du filtre VarianceThreshold...")
    joblib.dump(selector, 'modele_selector.pkl')
    
    print("Conversion des données en float32 pour économiser la RAM")
    X_pret_clean = X_pret_clean.astype('float32')
    
    print("\n Séparation des données en Train/Test (80% / 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pret_clean, 
        y_pret, 
        test_size=0.2, 
        random_state=42
    )
    
    # Entraînement du premier modèle (Random Forest)
    modele_rf_final = entrainer_et_evaluer_modele(X_train, X_test, y_train, y_test)
    
    # Entraînement du second modèle (Gradient Boosting) - APPEL AJOUTÉ
    modele_gb_final = entrainer_et_evaluer_gradient_boosting(X_train, X_test, y_train, y_test)
    
    print("\n Tous les modèles ont été entraînés et évalués avec succès !")