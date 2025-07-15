from agents.nodes.hr_agents.recruiter_agent import RecruiterAgent


recruiter_agent = RecruiterAgent()

result = recruiter_agent.invoke({
    "query": "Peut-on construire une usine pour avion militaire à Toulouse dans 3 mois pour un budget de 30000 euros ?",
    "data": {
        "localisation": "Toulouse",
        "budget_utilisateur": 30000,
        "delai_utilisateur_jours": 90,
        "competences_manquantes": [
            "Aerospace Engineering",
            "Project Management",
            "Construction Management",
            "Supply Chain Management",
            "Logistics",
            "Budgeting",
            "Cost Estimation",
            "Scheduling",
            "Risk Management",
            "Quality Control",
            "Aviation Regulations Compliance",
            "Site Management",
            "Civil Engineering",
            "Mechanical Engineering",
            "Electrical Engineering",
            "Industrial Automation",
            "Security Clearance",
            "Local Authority Liaison",
            "Environmental Impact Assessment",
            "Health and Safety Management"
        ],
        "capacites_disponibles": []
    }
})

print(result)
