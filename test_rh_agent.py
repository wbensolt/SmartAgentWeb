"""import json
from agents.nodes.hr_agents.rhagent import check_labor_law

if __name__ == "__main__":
    query = (
        "Nous prévoyons un projet RH avec un budget de 50 000 euros, "
        "un délai de lancement de 3 mois, et des compétences en gestion, recrutement et formation."
    )
    data = {
        "budget_moyen": 50000,
        "delai_moyen_lancement_projet": 90,
        "capacites_disponibles": [
            {"competence": "gestion", "effectif": 3},
            {"competence": "recrutement", "effectif": 2},
            {"competence": "formation", "effectif": 1}
        ],
        "context": {
            "convention_collective": "IDCC 2098"
        }
    }

    input_str = json.dumps({"query": query, "data": data})

    result = check_labor_law.invoke(input_str)

    print("=== Résultat RHAgent ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
"""
"""import json
from agents.nodes.hr_agents.talentManagerrh import talent_manager_tool

if __name__ == "__main__":
    query = (
        "Nous prévoyons un projet RH avec un budget de 50 000 euros, "
        "un délai de lancement de 3 mois, et des compétences en gestion, recrutement et formation."
    )
    data = {
        "budget_moyen": 50000,
        "delai_moyen_lancement_projet": 90,
        "competences_couvertes": [
            {"competence": "gestion", "effectif": 3},
            {"competence": "recrutement", "effectif": 2},
            {"competence": "formation", "effectif": 1}
        ],
        "competences_manquantes": ["communication", "leadership"],
        "context": {
            "convention_collective": "IDCC 2098"
        }
    }

    # Préparation de la chaîne JSON à passer à l'agent
    input_str = json.dumps({
        "query": query,
        "state": {
            "data_analytics": data
        }
    }, ensure_ascii=False)

    # Invocation du tool avec une string JSON
    result = talent_manager_tool.invoke(input_str)

    print("=== Résultat TalentManagerAgent ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
"""

import json
from agents.nodes.hr_agents.onboarding_agent import onboarding_agent_tool

if __name__ == "__main__":
    query = "Projet d'intégration de 10 nouveaux collaborateurs pour un service IT."
    data = {
        "capacites_disponibles": [
            {"competence": "formation", "effectif": 2},
            {"competence": "tutorat", "effectif": 1}
        ]
    }
    state = {"data_analytics": data}

    input_str = json.dumps({
        "query": query,
        "team_size": 10,
        "start_date": "2025-07-01",
        "state": state
    })

    result = onboarding_agent_tool.invoke(input_str)

    print("=== Résultat OnboardingAgent ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
