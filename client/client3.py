import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split # Ajout de cet import vital !
from sklearn.model_selection import GridSearchCV # Ajout de l'import pour la recherche par grille

chemin_fichier = r'C:\Users\mskit\OneDrive\Documents\donnees_nettoyees.csv'

# Foncion de prépation de donnée
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
        # Utilisation de .loc pour éviter les avertissements SettingWithCopyWarning de Pandas
        df_filtre[col] = pd.to_numeric(colonne_avec_points, errors='coerce')   

    # Sécurisastion
    for col in cols_cat_presentes:
        df_filtre[col] = df_filtre[col].astype(str)

    # Suppression ligne vide
    df_propre = df_filtre.dropna(subset=['implantation_station']).copy() # Ajout du .copy() ici pour être sûr
    
    # On identifie les 15 types d'implantation les plus fréquents
    top_15_implantations = df_propre[colonne_cible].value_counts().nlargest(15).index
    
    # On remplace tout le reste par la catégorie "Autre"
    df_propre[colonne_cible] = df_propre[colonne_cible].apply(
        lambda x: x if x in top_15_implantations else 'Autre'
    )
    print("Nettoyage : Réduction à", len(top_15_implantations) + 1, "classes pour la cible.")

    
    # Séparation du tab en x et y
    # .copy() permet de travailler sur un tableau indépendant pour éviter les avertissements
    X = df_propre[cols_num_presentes + cols_cat_presentes].copy()
    y = df_propre[colonne_cible]
    
    # Encodage de la cible
    encodeur_cible = LabelEncoder()
    y_encode = encodeur_cible.fit_transform(y)
    
    # Normalisation 
    scaler = StandardScaler()
    
    for col in cols_num_presentes[:]: 
        if X[col].nunique() == 1:
            printf("Nettoyage : Suppression de la colonne constante", 'col')
            cols_num_presentes.remove(col)

    # On s'assure que toutes les colonnes sont bien numériques
    for col in cols_num_presentes:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # On remplace toutes les valeurs manquantes (NaN) par la médiane de leur colonne
    X[cols_num_presentes] = X[cols_num_presentes].fillna(X[cols_num_presentes].median())

    # Normalisation
    X_num_transforme = scaler.fit_transform(X[cols_num_presentes])
    df_num = pd.DataFrame(X_num_transforme, columns=cols_num_presentes, index=X.index)
    
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


# Entrainement du modèle
########################

def entrainer_et_evaluer_modele(X_train, X_test, y_train, y_test):
    print("\n Entraînement du modèle de Classification > Random Forest")
    
    modele_base = RandomForestClassifier(
        class_weight='balanced', # Prise en compte du déséquilibre des classes
        n_jobs=1,                # 1 seul cœur utilisé pour stabiliser la RAM
        random_state=42
    )

    # Définition de la grille des paramètres à tester
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [15, 20],
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
    print("Entraînement terminé")
    
    # Récupération du meilleur modèle trouvé par le GridSearch
    modele = grille_recherche.best_estimator_
    print("Meilleurs paramètres trouvés :", grille_recherche.best_params_)

    print("\n Évaluation du modèle sur les données de test ")
    
    score_train = modele.score(X_train, y_train)
    score_test = modele.score(X_test, y_test)
    print("Précision sur les données d'entraînement :", round(score_train * 100,2), "%")
    print("Précision sur les données de test        :", round(score_test * 100,2), "%")

    # Prédictions et rapport détaillé
    predictions = modele.predict(X_test)
    precision = accuracy_score(y_test, predictions)
    print(" Précision globale finale  :", round(precision * 100,2), "%")

    print("\n Rapport détaillé par catégorie :")
    print(classification_report(y_test, predictions))

    # Sauvegarde du modèle
    print("Sauvegarde du modèle Random Forest...")
    joblib.dump(modele, 'modele_random_forest.pkl')
    
    
    # Renvoie le modèle
    return modele




if __name__ == "__main__":
    print("\n Chargement des données")
    df = pd.read_csv(chemin_fichier, low_memory=False)
    print("Info : Nombre de lignes chargées depuis le CSV : ", len(df))
    
    print("\n Lancement du nettoyage et de la préparation explicite")
    # On récupère l'intégralité des données préparées (X_pret)
    X_pret, y_pret = preparer_donnees_implantation_v2(df)
    
    print("\n Optimissation + nettoyage")
    selector = VarianceThreshold(threshold=0.0)
    X_pret_clean = selector.fit_transform(X_pret)
    print("Colonnes conservées :", X_pret_clean.shape[1] , "sur", X_pret.shape[1], "initialement")
    
    print("Conversion des données en float32 pour économiser la RAM")
    X_pret_clean = X_pret_clean.astype('float32')
    
    print("\n Séparation des données en Train/Test (80% / 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pret_clean, 
        y_pret, 
        test_size=0.2, 
        random_state=42
    )
    
    # Nouveau modèle
    modele_final = entrainer_et_evaluer_modele(X_train, X_test, y_train, y_test)
    
    print("\n modèle entrainé")