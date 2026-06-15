import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

chemin_fichier = r'C:\Users\mskit\OneDrive\Documents\donnees_nettoyees.csv'
df = pd.read_csv(chemin_fichier, low_memory=False)
print(f"-> Info : Nombre de lignes chargées depuis le CSV : {len(df)}")

# Préparation des données
#########################

def preparer_donnees_implantation_v2(df):
    
    colonnes_numeriques = ['nbre_pdc', 'puissance_nominale', 'consolidated_longitude', 'consolidated_latitude']
    colonnes_categorielles = ['condition_acces', 'reservation', 'paiement_cb', 'prise_type_2', 'accessibilite_pmr']
    colonne_cible = 'implantation_station'
    
    toutes_colonnes_utiles = colonnes_numeriques + colonnes_categorielles + [colonne_cible]
    
    
    # --- ÉTAPE EXPLICITE 1 : SÉLECTION DES COLONNES PRÉSENTES ---
    # On crée une liste vide pour stocker uniquement les colonnes qui existent vraiment dans ton fichier
    colonnes_presentes = []
    # On lance une boucle pour examiner chaque colonne de notre "liste idéale"
    for col in toutes_colonnes_utiles:
        # Si la colonne examinée fait bien partie des colonnes de ton fichier Excel/CSV
        if col in df.columns:
            # Alors on l'ajoute à notre liste de colonnes valides grâce à .append()
            colonnes_presentes.append(col)
    
    # On crée un nouveau tableau (df_filtre) en ne gardant que les colonnes valides qu'on vient de trouver
    df_filtre = df[colonnes_presentes]
    
    
    # --- ÉTAPE EXPLICITE 2 : TRI DES VARIABLES NUMÉRIQUES PRÉSENTES ---
    # On crée une liste vide pour stocker les colonnes numériques qui ont survécu au filtre
    cols_num_presentes = []
    # On examine chaque colonne numérique de notre liste de départ
    for col in colonnes_numeriques:
        # Si elle est bien présente dans notre tableau filtré
        if col in df_filtre.columns:
            # On l'ajoute à notre liste finale des colonnes numériques valides
            cols_num_presentes.append(col)
            
            
    # --- ÉTAPE EXPLICITE 3 : TRI DES VARIABLES CATÉGORIELLES PRÉSENTES ---
    # On crée une liste vide pour stocker les colonnes catégorielles qui ont survécu au filtre
    cols_cat_presentes = []
    # On examine chaque colonne catégorielle de notre liste de départ
    for col in colonnes_categorielles:
        # Si elle est bien présente dans notre tableau filtré
        if col in df_filtre.columns:
            # On l'ajoute à notre liste finale des colonnes catégorielles valides
            cols_cat_presentes.append(col)


    # --- ÉTAPE EXPLICITE 4 : NETTOYAGE INDIVIDUEL DES NOMBRES (VIRGULES ET TYPES) ---
    # On lance une boucle pour traiter chaque colonne numérique valide une par une
    for col in cols_num_presentes:
        # On convertit temporairement toute la colonne en texte pour pouvoir manipuler les caractères
        colonne_en_texte = df_filtre[col].astype(str)
        # On remplace les virgules des fichiers français par des points décimaux informatiques
        colonne_avec_points = colonne_en_texte.str.replace(',', '.', regex=False)
        # On force Python à transformer ce texte en vrai nombre. S'il y a un bug, la case devient "Vide" (NaN)
        df_filtre[col] = pd.to_numeric(colonne_avec_points, errors='coerce')
        
        
    # --- ÉTAPE EXPLICITE 5 : SÉCURISATION DES CATÉGORIES ---
    # On lance une boucle pour traiter chaque colonne de texte une par une
    for col in cols_cat_presentes:
        # On force le contenu à être du texte pur pour éviter que l'encodeur ne confonde des chiffres cachés
        df_filtre[col] = df_filtre[col].astype(str)


    # --- ÉTAPE EXPLICITE 6 : SUPPRESSION DES LIGNES COMPORTANT DES VIDES ---
    # On jette toutes les lignes du tableau qui contiennent au moins une case "Vide" (NaN)
    df_propre = df_filtre.dropna(subset=['implantation_station'])
    
    
    # --- ÉTAPE EXPLICITE 7 : SÉPARATION DU TABLEAU EN X ET Y ---
    # X reçoit uniquement les caractéristiques (numériques + catégorielles) qui servent à la prédiction
    X = df_propre[cols_num_presentes + cols_cat_presentes]
    # y reçoit uniquement la colonne cible (la réponse que l'IA doit trouver)
    y = df_propre[colonne_cible]
    
    
    # --- ÉTAPE EXPLICITE 8 : ENCODAGE MATHÉMATIQUE DE LA CIBLE ---
    # On initialise l'outil qui attribue un numéro à chaque catégorie de ta cible
    encodeur_cible = LabelEncoder()
    # On transforme les textes de ta cible (ex: "Parking public") en nombres (ex: 0, 1, 2)
    y_encode = encodeur_cible.fit_transform(y)
    
    
    # --- ÉTAPE EXPLICITE 9 : NORMALISATION DES VARIABLES NUMÉRIQUES ---
    # On initialise l'outil qui va centrer et réduire nos nombres (moyenne = 0, écart-type = 1)
    scaler = StandardScaler()
    # --- AUTOMATISATION : ON SUPPRIME LES COLONNES CONSTANTES ---
    
    # On crée une copie de la liste pour pouvoir la modifier en bouclant dessus
    for col in cols_num_presentes[:]: 
        # Si la colonne n'a qu'une seule valeur unique (ex: que des 0)
        if X[col].nunique() == 1:
            print(f"-> Nettoyage : Suppression de la colonne constante '{col}'")
            # On la retire de notre liste de colonnes numériques pour le scaler
            cols_num_presentes.remove(col)

    # --- TA LIGNE DU SCALER ---
    X_num_transforme = scaler.fit_transform(X[cols_num_presentes])
    # On applique la normalisation mathématique sur nos colonnes de chiffres
    X_num_transforme = scaler.fit_transform(X[cols_num_presentes])
    # On reconstruit un joli tableau propre avec ces chiffres normalisés
    df_num = pd.DataFrame(X_num_transforme, columns=cols_num_presentes, index=X.index)
    
    
    # --- ÉTAPE EXPLICITE 10 : ENCODAGE DES VARIABLES CATÉGORIELLES ---
    # --- À AJOUTER JUSTE AVANT LA LIGNE 102 ---
    
    # 1. On s'assure que toutes les colonnes sont bien numériques (convertit le texte indésirable en NaN)
    for col in cols_num_presentes:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # 2. On remplace toutes les valeurs manquantes (NaN) par la médiane de leur colonne
    X[cols_num_presentes] = X[cols_num_presentes].fillna(X[cols_num_presentes].median())

    # --- TA LIGNE ACTUELLE (Ligne 102) ---
    X_num_transforme = scaler.fit_transform(X[cols_num_presentes])
    # On initialise l'outil OneHotEncoder pour créer des colonnes remplies de 0 et de 1
    encodeur_categories = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # On applique cet encodage sur nos colonnes de texte
    X_cat_transforme = encodeur_categories.fit_transform(X[cols_cat_presentes])
    # On demande à l'encodeur de nous donner les nouveaux noms des colonnes créées
    nouveaux_noms = encodeur_categories.get_feature_names_out(cols_cat_presentes)
    # On reconstruit un joli tableau propre avec ces colonnes de 0 et de 1
    df_cat = pd.DataFrame(X_cat_transforme, columns=nouveaux_noms, index=X.index)
    
    
    # --- ÉTAPE EXPLICITE 11 : RECOMBINAISON FINALE ---
    # On colle horizontalement (axis=1) le tableau des chiffres normés et le tableau des catégories en 0 et 1
    X_final = pd.concat([df_num, df_cat], axis=1)
    
    
    # --- ÉTAPE EXPLICITE 12 : SAUVEGARDE SUR TON DISQUE DUR (REQUIS EN ROUGE) ---
    # On enregistre l'outil de normalisation dans un fichier externe
    joblib.dump(scaler, 'modele_scaler.pkl')
    # On enregistre l'outil d'encodage des caractéristiques dans un fichier externe
    joblib.dump(encodeur_categories, 'modele_encodeur_cat.pkl')
    # On enregistre l'outil d'encodage de ta cible dans un fichier externe
    joblib.dump(encodeur_cible, 'modele_encodeur_cible.pkl')
    
    # On affiche un message clair indiquant que l'exécution s'est terminée sans encombres
    print(f"--> Préparation terminée ! Nombre de lignes conservées : {len(X_final)}")
    
    # La fonction se ferme en renvoyant les deux matrices X et y prêtes pour ton modèle de classification
    return X_final, y_encode

    # =========================================================
# ÉTAPE 4 : NETTOYAGE FINAL ET SÉPARATION DES DONNÉES
# =========================================================

print("\n4. Démarrage de l'optimisation et de la séparation...")

# 4a. Suppression des colonnes avec une variance de 0 (Évite les divisions par zéro)
print(" -> Nettoyage des colonnes sans variance...")
selector = VarianceThreshold(threshold=0.0)
X_pret_clean = selector.fit_transform(X_pret)
print(f" -> Colonnes conservées : {X_pret_clean.shape[1]} (sur {X_pret.shape[1]} initialement)")

# 4b. Optimisation de la mémoire (Évite le MemoryError)
print(" -> Conversion des données en float32 pour économiser la RAM...")
X_pret_clean = X_pret_clean.astype('float32')

# 4c. Création de X_train, X_test, y_train, y_test (Évite le NameError)
print(" -> Séparation des données en Train/Test (80% / 20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_pret_clean, 
    y_pret, 
    test_size=0.2, 
    random_state=42
)


# =========================================================
# ÉTAPE 5 : ENTRAÎNEMENT DU MODÈLE
# =========================================================

print("\n5. Entraînement du modèle de Classification (Random Forest)...")

# Initialisation "sécurisée" du modèle pour ne pas faire exploser la RAM
modele = RandomForestClassifier(
    n_estimators=50,         # 50 arbres pour commencer sans saturer la mémoire
    max_depth=15,            # Limite la profondeur pour éviter un arbre infini
    min_samples_split=10,    # Nombre minimum de données pour créer une branche
    n_jobs=1,                # 1 seul cœur utilisé pour stabiliser la RAM
    random_state=42
)

# Lancement de l'entraînement
modele.fit(X_train, y_train)

print("✅ Entraînement terminé avec succès !")


# =========================================================
# ÉTAPE 6 : ÉVALUATION
# =========================================================

print("\n6. Évaluation du modèle...")
score_train = modele.score(X_train, y_train)
score_test = modele.score(X_test, y_test)

print(f" -> Précision sur les données d'entraînement : {score_train * 100:.2f}%")
print(f" -> Précision sur les données de test        : {score_test * 100:.2f}%")


# --- BLOC D'ÉXÉCUTION PRINCIPAL (LE MAIN) ---
if __name__ == "__main__":
    # On affiche un message d'avancement dans la console
    print("1. Chargement des données...")
    # On charge le fichier CSV (low_memory=False évite à Pandas de râler sur les types de données)
    df = pd.read_csv(chemin_fichier, low_memory=False)
    
    # On affiche l'étape suivante
    print("2. Lancement du nettoyage et de la préparation explicite...")
    # On appelle notre fonction en lui passant notre tableau brut "df"
    X_train, y_train = preparer_donnees_implantation_v2(df)
    
    # On valide la fin du script avec succès
    print("3. ✅ Succès total ! Variables X_pret et y_pret prêtes pour ton modèle.")

    
    print("5. Entraînement du modèle de Classification (Random Forest)...")
    # On crée le modèle (random_state=42 fige le hasard pour tes tests)
    modele = RandomForestClassifier(random_state=42)

    # L'IA s'entraîne sur les 80% de données (Train)
    modele.fit(X_train, y_train)
    print("✅ Apprentissage terminé !")

    # --- L'HEURE DE VÉRITÉ ---
    print("\n6. Évaluation du modèle sur les données de test (les 20% cachés)...")
    # L'IA passe son examen : on lui donne les caractéristiques (X_test) et elle doit deviner les réponses
    predictions = modele.predict(X_test)

    # On calcule la note de l'examen en comparant ses prédictions aux vraies réponses (y_test)
    precision = accuracy_score(y_test, predictions)
    print(f"🎯 Précision globale de l'IA : {precision * 100:.2f} %")

    # Affichage du bulletin de notes détaillé
    print("\n📊 Rapport détaillé par catégorie :")
    print(classification_report(y_test, predictions))