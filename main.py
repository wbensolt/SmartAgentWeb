# === FICHIER: main.py ===

import sys
from agents.vector.retrieve_project import create_project_graph
from agents.vector.retrieve import create_dynamic_rh_graph, ensure_index

def clean_response(text):
    cleaned = text.strip()
    cleaned = re.sub(r"```json|```", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned

def run_smart_agent():
    print("🧠 Lancement du Smart RH Project Agent...")
    graph = create_project_graph().compile()
    while True:
        query = input("\nPose ta question (ou 'exit') > ")
        if query.lower() == "exit":
            break
        result = graph.invoke({"query": query})

        raw_response = result.get("final_answer", result)
        print("[DEBUG] Réponse brute avant nettoyage :")
        print(raw_response)

        if isinstance(raw_response, str):
            # Si c'est une chaîne, on nettoie et parse JSON
            cleaned = clean_response(raw_response)
            try:
                parsed = json.loads(cleaned)
                print("\n=== ✅ Réponse Smart Agent (JSON validé) ===")
                print(parsed)
            except Exception as e:
                print(f"Réponse JSON invalide : {str(e)}")
                print("Tentative d'afficher la réponse brute :")
                print(raw_response)
        elif isinstance(raw_response, dict):
            # Déjà un dict, on peut juste afficher
            print("\n=== ✅ Réponse Smart Agent (dict déjà parsé) ===")
            print(raw_response)
        else:
            # Type inattendu
            print(f"Réponse inattendue de type {type(raw_response)} :")
            print(raw_response)

        print("=================================\n")


def run_rh_agent():
    print("👔 Préparation du Retriever RH dynamique...")
    ensure_index()
    graph = create_dynamic_rh_graph().compile()
    while True:
        query = input("\nPose ta question RH (ou 'exit') > ")
        if query.lower() == "exit":
            break
        result = graph.invoke({"query": query})
        print("\n=== ✅ Réponse Finale de l'Agent RH ===")
        print(result.get("final_answer", "Aucune réponse générée."))
        print("=======================================\n")

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "project"

    if mode == "rh":
        run_rh_agent()
    else:
        run_smart_agent()
