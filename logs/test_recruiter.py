import logging
from agents.vector.state_schema import GraphState
from agents.vector.retrieve_project import create_project_graph

def test_recruiter_direct():
    graph = create_project_graph()
    compiled_graph = graph.compile()

    test_query = "Nous recherchons un développeur backend à Lyon avec un budget de 4000 euros et un démarrage rapide."
    fake_data_analytics = {
        "bassin_emploi": "Lyon",
        "disponibilite_profils": "Élevée",
        "tendances_marche": ["Demande croissante de profils backend"],
        "budget_moyen": 4500,
        "delai_moyen_lancement_projet": 30,
        "capacites_disponibles": [
            {"competence": "Python", "effectif": 3},
            {"competence": "Django", "effectif": 2}
        ],
        "budget_utilisateur": 4000,
        "delai_utilisateur_jours": 15,
        "localisation": "Lyon",
        "erreur": None
    }

    state = GraphState(query=test_query, data_analytics=fake_data_analytics)

    recruiter_output = compiled_graph.invoke("recruiter", state)

    print("\n=== Résultat de l'agent Recruiter ===")
    print(recruiter_output.get("recruiter_output"))  # ✅ clé correcte

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_recruiter_direct()
