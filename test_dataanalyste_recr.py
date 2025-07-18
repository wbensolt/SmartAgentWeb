from agents.nodes.hr_agents.recruiter_agent import recruit_candidates
from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst

def test_analyse_data():
    query = "Nous souhaitons lancer un projet RH à Toulouse avec un budget de 100000 euros et un délai de 3 mois."
    result = analyse_data_analyst(query)
    print("Résultat Data Analyst:")
    print(result)

def test_recruit_candidates():
    query = "Recherche développeur Python avec expérience SQL à Lille"
    data_analytics = {
        "localisation": "Lille",
        "capacites_disponibles": [
            {"competence": "python", "effectif": 3},
            {"competence": "sql", "effectif": 2},
            {"competence": "java", "effectif": 10}
        ],
        "budget_utilisateur": 50000,
        "delai_utilisateur_jours": 60,
        "competences_manquantes": ["python", "sql"]
    }
    
    #result = recruit_candidates(query, data=data_analytics)
    result = recruit_candidates.invoke({"query": query, "data": data_analytics})
    print(result)

if __name__ == "__main__":
    test_recruit_candidates()


"""if __name__ == "__main__":
    test_analyse_data()"""
