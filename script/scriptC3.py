import pandas as pd
import joblib
import tkinter as tk
from tkinter import ttk, messagebox

# --- 1. Chargement des modèles ---
print("Chargement des outils et des modèles...")
try:
    scaler = joblib.load('modele_scaler.pkl')
    encodeur_cat = joblib.load('modele_encodeur_cat.pkl')
    encodeur_cible = joblib.load('modele_encodeur_cible.pkl')
    selector = joblib.load('modele_selector.pkl')
    
    # Chargement des deux modèles d'entraînement
    modele_rf = joblib.load('modele_random_forest.pkl')
    modele_gb = joblib.load('modele_gradient_boosting.pkl')
    print("Modèles chargés avec succès !")
except Exception as e:
    print("Erreur de chargement : {0}".format(e))
    exit()


# --- 2. Fonction de prédiction ---
def lancer_prediction():
    try:
        # Récupération des données depuis le dictionnaire d'inputs
        donnees_saisies = {
            'nbre_pdc': [float(inputs['nbre_pdc'].get())],
            'puissance_nominale': [float(inputs['puissance_nominale'].get())],
            'consolidated_longitude': [float(inputs['consolidated_longitude'].get())],
            'consolidated_latitude': [float(inputs['consolidated_latitude'].get())],
            'condition_acces': [inputs['condition_acces'].get()],
            'reservation': [inputs['reservation'].get()],
            'paiement_cb': [inputs['paiement_cb'].get()],
            'prise_type_2': [inputs['prise_type_2'].get()],
            'accessibilite_pmr': [inputs['accessibilite_pmr'].get()]
        }
        
        df = pd.DataFrame(donnees_saisies)

        # Préparation (Scaler sur le numérique)
        cols_num = ['nbre_pdc', 'puissance_nominale', 'consolidated_longitude', 'consolidated_latitude']
        num_transforme = scaler.transform(df[cols_num])
        df_num = pd.DataFrame(num_transforme, columns=cols_num)

        # Préparation (Encodeur sur le catégoriel)
        cols_cat = ['condition_acces', 'reservation', 'paiement_cb', 'prise_type_2', 'accessibilite_pmr']
        cat_transforme = encodeur_cat.transform(df[cols_cat].astype(str))
        nouveaux_noms = encodeur_cat.get_feature_names_out(cols_cat)
        df_cat = pd.DataFrame(cat_transforme, columns=nouveaux_noms)

        # Recombinaison
        X_final = pd.concat([df_num, df_cat], axis=1)
        X_final = selector.transform(X_final)

        # Sélection du modèle en fonction du choix de l'utilisateur
        choix_modele = menu_modele.get()
        if choix_modele == 'Random Forest':
            modele_actif = modele_rf
        else:
            modele_actif = modele_gb

        # Prédiction
        prediction_num = modele_actif.predict(X_final)
        prediction_txt = encodeur_cible.inverse_transform(prediction_num)

        # Mise à jour UI (sans f-string)
        resultat_texte = "RECOMMANDATION ({0}) :\n{1}".format(choix_modele, prediction_txt[0])
        label_resultat.config(text=resultat_texte, fg="#059669")

    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des nombres valides (utilisez un point pour les décimales).")
    except Exception as e:
        messagebox.showerror("Erreur", "Vérifiez vos saisies : {0}".format(e))


# --- 3. Interface graphique (Style Procédural) ---
root = tk.Tk()
root.title("IRVE Station Predictor - IA")
# Taille légèrement augmentée pour s'adapter au nouveau champ
root.geometry("500x650")
root.configure(bg="#f0f2f5")

# Dictionnaire global pour stocker les éléments de saisie
inputs = {}

# Frame principal pour regrouper les éléments
main_frame = tk.Frame(root, bg="#f0f2f5", padx=20, pady=10)
main_frame.pack(expand=True, fill="both")

tk.Label(main_frame, text="Paramètres de la Station", font=("Arial", 16, "bold"), bg="#f0f2f5", fg="#1a202c").pack(pady=5)

# --- Choix du Modèle ---
frame_modele = tk.Frame(main_frame, bg="#e2e8f0", padx=10, pady=5)
frame_modele.pack(fill="x", pady=10)
tk.Label(frame_modele, text="Modèle prédictif :", width=20, anchor="w", bg="#e2e8f0", font=("Arial", 10, "bold")).pack(side="left")
menu_modele = ttk.Combobox(frame_modele, values=['Random Forest', 'Gradient Boosting'], state="readonly")
menu_modele.current(0)  # Random Forest sélectionné par défaut
menu_modele.pack(side="right", expand=True, fill="x")

# --- Ligne de séparation ---
tk.Frame(main_frame, bg="#cbd5e1", height=1).pack(fill="x", pady=5)

# --- Champs Numériques ---
num_fields = [
    ('nbre_pdc', 'Nombre de points de charge', '2'),
    ('puissance_nominale', 'Puissance en kW', '22.0'),
    ('consolidated_longitude', 'Longitude (ex: 2.35)', '2.3522'),
    ('consolidated_latitude', 'Latitude (ex: 48.85)', '48.8566')
]

for key, label_text, default in num_fields:
    frame = tk.Frame(main_frame, bg="#f0f2f5")
    frame.pack(fill="x", pady=2)
    tk.Label(frame, text=label_text, width=22, anchor="w", bg="#f0f2f5").pack(side="left")
    entry = tk.Entry(frame)
    entry.insert(0, default)
    entry.pack(side="right", expand=True, fill="x")
    inputs[key] = entry

# Ligne de séparation
tk.Frame(main_frame, bg="#cbd5e1", height=1).pack(fill="x", pady=10)

# --- Champs Catégoriels ---
cat_fields = [
    ('condition_acces', 'Condition d\'accès', ['Accès libre', 'Accès réservé', 'Payant']),
    ('reservation', 'Réservation', ['Faux', 'Vrai']),
    ('paiement_cb', 'Paiement CB', ['Vrai', 'Faux']),
    ('prise_type_2', 'Prise Type 2', ['Vrai', 'Faux']),
    ('accessibilite_pmr', 'Accessibilité PMR', ['Accessibilité inconnue', 'Accessible', 'Non accessible'])
]

for key, label_text, options in cat_fields:
    frame = tk.Frame(main_frame, bg="#f0f2f5")
    frame.pack(fill="x", pady=2)
    tk.Label(frame, text=label_text, width=22, anchor="w", bg="#f0f2f5").pack(side="left")
    combo = ttk.Combobox(frame, values=options, state="readonly")
    combo.current(0)
    combo.pack(side="right", expand=True, fill="x")
    inputs[key] = combo

# --- Bouton et Résultat ---
# Bouton plus compact
tk.Button(main_frame, text="LANCER LA PRÉDICTION", command=lancer_prediction, bg="#38bdf8", fg="white", font=("Arial", 12, "bold"), relief="flat").pack(fill="x", pady=15)

# Label de résultat visible et remonté
label_resultat = tk.Label(main_frame, text="Attente de saisie...", font=("Arial", 14, "bold"), bg="#ffffff", fg="#0284c7", pady=10, relief="groove")
label_resultat.pack(fill="x", pady=5)

root.mainloop()