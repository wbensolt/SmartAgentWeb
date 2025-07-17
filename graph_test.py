from agents.vector.retrieve_project_mcp import create_project_graph


def main():
    # Crée et compile le graphe
    graph = create_project_graph().compile()

    # Requête d'exemple
    query = "Analyse de faisabilité pour un centre logistique à Lille dans 3 mois avec budgets de 30000 euros"

    # Création de l'état initial
    initial_state = {"query": query}

    # Appel du graphe
    print("\n>>> Requête envoyée :", query)
    result = graph.invoke(initial_state)

    # Affichage du résultat
    print("\n--- Résultat complet ---")
    for key, value in result.items():
        print(f"\n[{key}]")
        print(value)
    
    print("\n--- Résultat RECRUITER ---")
    print(result.get("recruiter", {}))

    print("\n--- Résultat RH ---")
    print(result.get("rh", {}))

    print("\n--- Résultat ONBOARDING ---")
    print(result.get("onboarding", {}))

if __name__ == "__main__":
    main()
