"""from agents.nodes.hr_agents.RecruiterAgent import RecruiterAgent

if __name__ == "__main__":
    agent = RecruiterAgent()

    query = "Je cherche un développeur Python senior disponible immédiatement à Lille en alternance."

    result = agent.invoke({"query": query})
    profils = result.get("profils", [])

    print(f"Résultats pour la requête : {query}\n")
    for p in profils:
        print(f"- {p['nom']} | {p['competences']} | {p['localisation']} | {p['experience_niveau']} | {p['statut']} | dispo: {p['disponibilite']}")"""