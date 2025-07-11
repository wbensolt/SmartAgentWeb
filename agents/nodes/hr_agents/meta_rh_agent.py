from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
import json
import logging

class MetaAgent(Runnable):
    def __init__(self, llm=None, agents_map=None, max_iterations=2):
        """
        llm : LLM pour analyser les réponses et détecter incohérences
        agents_map : dict {agent_key: agent_instance} pour relancer questions ciblées
        max_iterations : nombre max d'itérations pour la boucle critique
        """
        self.llm = llm or LLMManager().get_llm()
        self.agents_map = agents_map or {}
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(__name__)

    def invoke(self, input: dict) -> dict:
        # input doit contenir les réponses initiales des agents : {agent_key: response}
        state = input.get("state", {})
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            self.logger.info(f"[MetaAgent] Itération {iteration} de détection incohérences")

            # Préparer prompt avec toutes les réponses agents
            prompt = f"""
            Tu es un méta-agent qui analyse les réponses suivantes d'agents spécialisés dans différents domaines.
            Identifie toutes incohérences, contradictions ou informations manquantes.
            Pour chaque problème détecté, génère une ou plusieurs questions ciblées à poser à l'agent concerné pour clarifier ou corriger.
            Si aucune incohérence n'est détectée, répond "Aucune incohérence détectée".

            Réponses des agents (JSON) :
            {json.dumps(state, indent=2, ensure_ascii=False)}

            Réponds strictement en JSON au format :
            {{
            "incoherences": [
                {{
                "agent": "finance",
                "probleme": "Le budget demandé dépasse le budget moyen.",
                "questions": ["Peux-tu expliquer ce dépassement ?"]
                }},
                ...
            ]
            }}
                        """

            try:
                raw_response = self.llm.invoke(prompt)
                response_clean = raw_response.strip()
                # Nettoyage simple
                if response_clean.startswith("```json"):
                    response_clean = response_clean[7:]
                if response_clean.endswith("```"):
                    response_clean = response_clean[:-3]
                response_clean = response_clean.strip()

                meta_data = json.loads(response_clean)

                self.logger.info(f"[MetaAgent] Analyse incohérences : {meta_data}")

                if not meta_data.get("incoherences"):
                    # Pas d'incohérence, on sort de la boucle
                    self.logger.info("[MetaAgent] Aucune incohérence détectée, boucle terminée.")
                    break

                # Pour chaque incohérence, poser questions aux agents concernés
                for issue in meta_data["incoherences"]:
                    agent_key = issue.get("agent")
                    questions = issue.get("questions", [])
                    agent = self.agents_map.get(agent_key)
                    if not agent:
                        self.logger.warning(f"[MetaAgent] Agent '{agent_key}' introuvable, on saute.")
                        continue

                    for question in questions:
                        self.logger.info(f"[MetaAgent] Question à '{agent_key}': {question}")

                        # Préparer input adapté, ici on passe la question + état actuel
                        agent_input = {
                            "query": question,
                            "state": state
                        }
                        # Appel synchrone invoke
                        answer = agent.invoke(agent_input)
                        self.logger.info(f"[MetaAgent] Réponse agent '{agent_key}': {answer}")

                        # Met à jour l'état avec la réponse affinée
                        if isinstance(answer, dict):
                            # On merge soigneusement selon ta structure
                            if agent_key in state and isinstance(state[agent_key], dict):
                                state[agent_key].update(answer)
                            else:
                                state[agent_key] = answer
                        else:
                            # Si réponse simple, on remplace
                            state[agent_key] = answer

                # boucle continue (max_iterations)
            except Exception as e:
                self.logger.error(f"[MetaAgent] Erreur durant l'analyse : {str(e)}", exc_info=True)
                break

        # On renvoie l'état final, enrichi et corrigé
        return state

    async def ainvoke(self, input: dict) -> dict:
        return self.invoke(input)
