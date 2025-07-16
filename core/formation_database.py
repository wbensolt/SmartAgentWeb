from typing import Dict, Optional, List
import random
from datetime import datetime, timedelta

class FormationDatabase:
    """Base de données des formations disponibles en France"""

    def __init__(self):
        self.formations = self._generate_formations()
        self.referentiels = self._generate_referentiels()

    def _generate_formations(self) -> Dict:
        """Génère un dictionnaire de formations fictives"""
        formations = {}
        domains = ["Informatique", "Santé", "Ingénierie", "Marketing", "Finance", "Art", "Éducation", "Droit", "Environnement", "Tourisme"]
        skills = ["Développement", "Design", "Gestion", "Analyse", "Communication", "Planification", "Création", "Stratégie", "Innovation", "Recherche"]

        for i in range(1, 101):
            domain = random.choice(domains)
            skill = random.choice(skills)
            name = f"{skill} {domain} {i}"
            key = name.lower().replace(" ", "_")

            formations[key] = {
                "duree": f"{random.randint(2, 20)} semaines",
                "cout_moyen": random.randint(1000, 10000),
                "organismes": [f"Organisme {random.randint(1, 20)}", f"Centre {random.randint(1, 20)}"],
                "certifications": [f"Certificat {skill}", f"Diplôme {domain}"],
                "taux_reussite": random.randint(70, 99),
                "prerequis": f"Connaissances en {domain}",
                "modalite": random.choice(["Présentiel", "Distanciel", "Hybride"])
            }

        return formations

    def _generate_referentiels(self) -> Dict:
        """Génère un dictionnaire de référentiels fictifs"""
        referentiels = {}
        for i in range(1, 101):
            referentiels[f"referentiel_{i}"] = {
                "description": f"Description du référentiel {i}",
                "fiches_metiers": random.randint(100, 2000),
                "url": f"https://example.com/referentiel{i}",
                "derniere_maj": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
            }
        return referentiels

    def find_formation(self, competence: str) -> Optional[Dict]:
        """Trouve une formation correspondant à une compétence"""
        competence_lower = competence.lower().replace(" ", "_")

        if competence_lower in self.formations:
            return self.formations[competence_lower]

        return None

    def get_referentiel_info(self) -> Dict:
        """Retourne les informations sur les référentiels disponibles"""
        return self.referentiels

# Exemple d'utilisation
formation_db = FormationDatabase()
print(formation_db.formations)
print(formation_db.referentiels)
