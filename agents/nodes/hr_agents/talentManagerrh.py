from langchain_core.runnables import Runnable
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from core.formation_database import FormationDatabase
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import TalentPlan
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

class TalentManagerAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.parser = PydanticOutputParser(pydantic_object=TalentPlan)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.logger = logging.getLogger(__name__)
        self.formation_db = FormationDatabase()

    def _analyze_formation_feasibility(self, competences_manquantes, budget_moyen, delai_moyen):
        formations_trouvees, cout_total, duree_max_semaines = [], 0, 0
        try:
            budget_num = float(str(budget_moyen).replace("€", "").replace(",", "").replace(" ", "")) if budget_moyen != "inconnu" else 50000
            delai_num = float(str(delai_moyen).replace("jours", "").replace(" ", "")) if delai_moyen != "inconnu" else 90
        except:
            budget_num, delai_num = 50000, 90

        for comp in competences_manquantes:
            formation = self.formation_db.find_formation(comp)
            if formation:
                formations_trouvees.append({"competence": comp, "formation": formation})
                cout_total += formation["cout_moyen"]
                try:
                    duree_max = int(formation["duree"].split("-")[-1].split()[0])
                    duree_max_semaines = max(duree_max_semaines, duree_max)
                except:
                    duree_max_semaines = max(duree_max_semaines, 8)

        ratio_budget = (cout_total / budget_num * 100) if budget_num > 0 else 0
        duree_jours = duree_max_semaines * 7
        ratio_delai = (duree_jours / delai_num * 100) if delai_num > 0 else 0

        faisable = ratio_budget <= 25 and ratio_delai <= 60 and len(formations_trouvees) >= len(competences_manquantes) * 0.7
        return {
            "formations_trouvees": formations_trouvees,
            "cout_total": cout_total,
            "duree_max_semaines": duree_max_semaines,
            "ratio_budget": ratio_budget,
            "ratio_delai": ratio_delai,
            "faisable": faisable,
            "nb_competences_formables": len(formations_trouvees)
        }

    def _generate_formation_plan(self, analysis):
        if not analysis["formations_trouvees"]:
            return "Aucune formation spécifique identifiée"

        plan_parts = [
            f"• {item['competence']}: {item['formation']['duree']} via {', '.join(item['formation']['organismes'][:2])} (certification: {item['formation']['certifications'][0]}, taux réussite: {item['formation']['taux_reussite']}%)"
            for item in analysis["formations_trouvees"]
        ]

        organismes = list({org for item in analysis["formations_trouvees"] for org in item["formation"]["organismes"][:2]})
        return (
            "Plan de formation personnalisé:\n"
            + "\n".join(plan_parts)
            + f"\n\nSynthèse:\n- Durée: {analysis['duree_max_semaines']} semaines\n"
            + f"- Coût: {analysis['cout_total']:,} € ({analysis['ratio_budget']:.1f}% budget)\n"
            + f"- Compétences couvertes: {analysis['nb_competences_formables']}\n"
            + f"- Organismes: {', '.join(organismes)}"
        )

    def invoke(self, input):
        query = str(input.get("query", ""))[:2000]
        state = input.get("state", {})
        data = state.get("data_analytics", {})

        budget, delai = data.get("budget_moyen", "inconnu"), data.get("delai_moyen_lancement_projet", "inconnu")
        comp_couv = data.get("competences_couvertes", [])
        comp_manq = data.get("competences_manquantes", [])

        analysis = self._analyze_formation_feasibility(comp_manq, budget, delai)
        plan = self._generate_formation_plan(analysis)
        referentiels = self.formation_db.get_referentiel_info()

        contexte_entreprise = f"Budget: {budget} €, Délai: {delai} jours, Compétences: {', '.join([c.get('competence','') for c in comp_couv]) or 'Aucune'}, Manquantes: {', '.join(comp_manq) or 'Aucune'}"

        exemple = json.dumps({
            "upskill_possible": analysis["faisable"],
            "duree_formation": f"{analysis['duree_max_semaines']} semaines",
            "referentiel_competences": "oui",
            "plan_global": f"Formation via organismes spécialisés pour {analysis['nb_competences_formables']} compétences"
        }, indent=2, ensure_ascii=False)

        prompt = f"""
Tu es un expert RH spécialisé dans la montée en compétences.

Contexte du projet:
{query}

Contexte entreprise:
{contexte_entreprise}

Analyse:
{plan}

Réponds avec un JSON conforme:
{exemple}

Réponse:
""".strip()

        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, dict):
                response = response.get("response", "")
            cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = self.fixing_parser.parse(cleaned)
            result = parsed.dict()
            result["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "formations": analysis["formations_trouvees"],
                "cout_total": analysis["cout_total"],
                "referentiels": list(referentiels.keys())
            }
            return result
        except Exception as e:
            self.logger.error(f"Erreur TalentManagerAgent: {e}", exc_info=True)
            return {
                "upskill_possible": False,
                "duree_formation": "Non estimable",
                "referentiel_competences": "oui",
                "plan_global": f"Erreur: {str(e)}",
                "metadata": {"timestamp": datetime.now().isoformat(), "error": str(e)}
            }

    async def ainvoke(self, input):
        return self.invoke(input)
