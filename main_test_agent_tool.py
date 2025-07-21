# main.py
import json
import traceback
from agents.nodes.hr_agents.hr_workflow import create_hr_workflow
from agents.nodes.hr_agents.workflow_intial_agent import workflow_intial_agent

def display_step_output(step_name, output_data):
    """Affiche joliment les résultats d'une étape"""
    print(f"\n{'='*50}")
    print(f"📋 {step_name.upper()} RESULTS")
    print("="*50)
    if isinstance(output_data, dict):
        print(json.dumps(output_data, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"Type de sortie inattendu: {type(output_data)}")
        print(output_data)
    print("="*50 + "\n")

def main():
    # Initialisation du workflow
    workflow = create_hr_workflow()
    
    print("🚀 HR Workflow Testing Tool")
    print("--------------------------")
    
    while True:
        # Saisie utilisateur
        user_input = input("\nEntrez votre requête HR (ou 'quit' pour sortir):\n> ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Au revoir !")
            break
            
        if not user_input.strip():
            print("Veuillez entrer une requête valide.")
            continue
            
        # Exécution du workflow
        print(f"\n⚡ Traitement de la requête: '{user_input}'...")
        
        inputs = {"input": user_input}
        
        try:
            workflow_output = workflow.stream(inputs)
            
            # Nouvelle méthode plus robuste pour itérer sur les résultats
            for output in workflow_output:
                if output:  # Vérifie si l'output n'est pas vide
                    for node_name, node_output in output.items():
                        display_step_output(node_name, node_output)
                else:
                    print("Aucun résultat retourné pour cette étape.")
                    
            print("✅ Workflow terminé avec succès !")
            
        except Exception as e:
            print(f"\n❌ ERREUR CRITIQUE ❌")
            print(f"Type d'erreur: {type(e).__name__}")
            print(f"Message: {str(e)}")
            print("\nStack trace:")
            traceback.print_exc()
            print("\nConseil: Vérifiez que tous les nœuds retournent bien des dictionnaires.")

if __name__ == "__main__":
    main()