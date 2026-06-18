import os
import sys
import joblib
import pandas as pd

fichier_modeles = "models/pipeline_bornes.joblib"
if not os.path.exists(fichier_modeles):
    print("le fichier  est introuvable.")
    print("d'abord exécuter le script pour avoir les modèles.")
    sys.exit()

artefacts = joblib.load(fichier_modeles)

# recup
best_rf = artefacts["best_rf"]
best_dt = artefacts["best_dt"]
preprocessor = artefacts["preprocessor"]
enco = artefacts["enco"]


def predire_nouvelle_borne(borne_brute: dict, choix_modele: str = "random_forest") -> dict:

    # Séelection du modèle sauv
    if choix_modele == "random_forest":
        clf = best_rf
    elif choix_modele == "decision_tree":
        clf = best_dt
    else:
        raise ValueError("Modèle inconnu random_forest ou decision_tree")


    df_borne = pd.DataFrame([borne_brute])
    X_processed = preprocessor.transform(df_borne)

    # 4predire de la classe
    classe_code = clf.predict(X_processed)[0]

    # enco pour le remetre en texte
    classe_texte = enco.inverse_transform([classe_code])[0]

    #probabilités de confiance du modèle
    proba_dict = {}
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X_processed)[0]
        for i, p in enumerate(probas):
            nom_classe = enco.inverse_transform([i])[0]
            proba_dict[nom_classe] = f"{p * 100:.2f}%"

    return {
        "classe_predite": classe_texte,
        "certitude_du_modele": proba_dict,
        "algorithme_utilise": choix_modele
    }

# choix des caracteristique
implantation = input("type d'implantation, ex: Voirie, Station d'autoroute :")
acces = input("condition d'acces, ex: Accessibilité gratuite, Payant : ")

while True:
    saisie_pdc = input("Nombre de points de charge: ").strip()

    if saisie_pdc.isdigit():
        nbre_pdc = int(saisie_pdc)
        if nbre_pdc > 0:
            break
        else:
            print("au moins 1 point de charge.")
    else:
        print("invalide.")

#savoir si c'est en france
while True:
    lat = float(input("Latitude de la borne, entre 41.0 et 51.1: "))
    if 41.0 <= lat <= 51.1:
        break
    print("pas en France !")
while True:
    lon = float(input("Longitude de la borne, entre -5.0 et 10.0 : "))
    if -5.0 <= lon <= 10.0:
        break
    print("pas en France !")


def demander_oui_non(question):
    while True:
        reponse = input(question + " oui/non : ").strip().lower()
        if (reponse =='oui'):
            return True
        elif (reponse=='non'):
            return False
        else:
            print("que oui ou non")


print("choisie ta borne")
t2 = demander_oui_non("Présence d'une prise Type 2 ?")
ccs = demander_oui_non("Présence d'une prise Combo CCS (recharge rapide) ?")
chademo = demander_oui_non("Présence d'une prise CHAdeMO ?")
ef = demander_oui_non("Présence d'une prise EF (Prise domestique) ?")
autre_prise = demander_oui_non("Présence d'un autre type de prise ?")
acte = demander_oui_non("Paiement à l'acte possible ?")
cb = demander_oui_non("Paiement par Carte Bancaire direct ?")
autre_paiement = demander_oui_non("Autre moyen de paiement (badge/appli) ?")
reservation = demander_oui_non("Réservation de la borne possible ?")


#dictionnaire avec les réponses de l'utilisateur
nouvelle_borne_test = {
    "implantation_station": implantation,
    "condition_acces": acces,
    "nbre_pdc": nbre_pdc,
    "consolidated_latitude": lat,
    "consolidated_longitude": lon,
    "prise_type_2": t2,
    "prise_type_combo_ccs": ccs,
    "prise_type_chademo": chademo,
    "prise_type_ef": ef,
    "prise_type_autre": autre_prise,
    "paiement_acte": acte,
    "paiement_cb": cb,
    "paiement_autre": autre_paiement,
    "reservation": reservation
}
print("\n prediction")


#test avec le Random Forest
resultat_rf = predire_nouvelle_borne(nouvelle_borne_test, choix_modele="random_forest")
print(f"\n resultat random forest")
print(f" Puissance estimée : {resultat_rf['classe_predite']}")
print(f" Détail des probabilités : {resultat_rf['certitude_du_modele']}")

#test avec decision tree
resultat_dt = predire_nouvelle_borne(nouvelle_borne_test, choix_modele="decision_tree")
print(f"\n Resultat Decision tree")
print(f" Puissance estimée : {resultat_dt['classe_predite']}")
print(f" Détail des probabilités : {resultat_dt['certitude_du_modele']}")

