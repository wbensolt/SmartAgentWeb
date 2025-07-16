import gradio as gr
import requests

# Appel à l'API FastAPI
API_URL = "http://127.0.0.1:8000/feasibility/invoke"

def analyser_faisabilite(query: str):
    try:
        response = requests.post(API_URL, json={"query": query})
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {"Erreur": f"Impossible d'appeler l'API : {str(e)}"}

    cards = {}

    # 🧠 Data Analytics
    da = data.get("data_analytics", {})
    cards["🧠 Data Analyst"] = f"""
### <b>Conclusion</b> :
<b>{ 'Faisable' if len(da.get('competences_manquantes', [])) == 0 else 'Très peu faisable' }</b>

<b>Budget utilisateur :</b> {da.get('budget_utilisateur')} €  
<b>Budget moyen observé :</b> {da.get('budget_moyen')} €  
<b>Délai utilisateur :</b> {da.get('delai_utilisateur_jours')} jours  
<b>Délai moyen observé :</b> {da.get('delai_moyen_lancement_projet')} jours  

<b>Compétences couvertes :</b> {[c['competence'] for c in da.get('competences_couvertes', [])]}  
<b>Compétences manquantes :</b> {da.get('competences_manquantes')}  

🔺 {da.get('warnings')}
"""

    # 👥 Recruiter
    recruiter = data.get("recruiter", {})
    cards["👥 Recruteur"] = f"""
<b>Profils trouvés :</b> {recruiter.get('count')} profils  
<b>Compétences recherchées :</b> {recruiter.get('competences_recherchees_finales')}  
<b>Compétences non trouvées :</b> {recruiter.get('competences_non_trouvees')}  

🔺 {recruiter.get('message')}
"""

    # ⚖️ RH
    rh = data.get("rh", {})
    cards["⚖️ RH"] = f"""
<b>Conformité juridique :</b> {'✅ Conforme' if rh.get('compliance_status') == 'conforme' else '❌ Non conforme'}  

{rh.get('response')}
"""

    # 🎓 Talent
    talent = data.get("talent", {})
    cards["🎓 Talent"] = f"""
<b>Upskill possible :</b> {'✅ Oui' if talent.get('upskill_possible') else '❌ Non'}  

{talent.get('plan_global')}
"""

    # 👋 Onboarding
    onboarding = data.get("onboarding", {}).get("accueil", {})
    cards["👋 Onboarding"] = f"""
<b>Ressources disponibles :</b>  
- RH : {onboarding.get('resources', {}).get('rh')}  
- Tuteurs : {onboarding.get('resources', {}).get('tutors')}  

<b>Checklist :</b> {onboarding.get('checklist')}  
<b>Risques :</b> {onboarding.get('risks')}
"""

    # 💰 Payroll
    payroll = data.get("payroll", {})
    cards["💰 Coûts"] = f"""
<b>Coût mensuel estimé :</b> {payroll.get('cout_mensuel')}  
<b>Budget suffisant ?</b> {'✅ Oui' if payroll.get('budget_suffisant') == 'oui' else '❌ Non'}  

{payroll.get('analyse_cout')}
"""

    # 🧪 Critique
    critique = data.get("critique", {})
    cards["🧪 Critique"] = f"""
<b>Points forts :</b> {critique.get('points_forts')}  
<b>Points faibles :</b> {critique.get('points_faibles')}  
<b>Suggestions :</b> {critique.get('suggestions')}  
<b>Note de cohérence :</b> {critique.get('note_coherence')}/5
"""

    # ✅ Synthèse
    final = data.get("final_answer", {})
    cards["✅ Synthèse finale"] = f"""
<b>Faisabilité :</b> {final.get('faisabilite')}  
<b>Score de confiance :</b> {final.get('score_confiance')}  

<b>Conditions de réussite :</b> {final.get('conditions_reussite')}  
<b>Risques principaux :</b> {final.get('risques_principaux')}
"""

    # 📊 Rapport global (onglet supplémentaire)
    cards["📊 Rapport Global"] = """
<b>Ce rapport est généré automatiquement à partir de l’analyse croisée de tous les agents RH.</b>  
Il fournit une synthèse transversale de la viabilité du projet avec mise en perspective stratégique.

🚧 À personnaliser selon vos besoins.
"""

    return cards

# --- CSS STYLE ---
css_style = """
body, .gradio-container {
    background-color: #f5e6da !important;
}

h1, h2, h3, p, div, span, label {
    font-size: 20px !important;
    font-weight: bold;
    color: #222222 !important;
}

.gradio-container {
    padding: 20px !important;
}

/* Onglets */
button.svelte-1ipelgc, .tabitem, .tabitem span {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-weight: bold;
    font-size: 16px !important;
    border: 1px solid #ffffff !important;
}

/* Onglet actif */
button.selected.svelte-1ipelgc {
    background-color: #222222 !important;
}
"""

# --- INTERFACE GRADIO ---css=css_style,
with gr.Blocks(title="Smart RH Project Agent") as demo:
    gr.Markdown("""# 🧠 <b>Smart RH Project Agent</b>
<b>Entrez un projet pour analyser sa faisabilité par nos agents spécialisés.</b>""")

    with gr.Row():
        query_input = gr.Textbox(label="Votre projet", placeholder="Ex : Peut-on construire une usine pour avions militaires à Toulouse dans 3 mois ?", lines=4)
        bouton = gr.Button("🔍 Investiguer")

    output_tabs = gr.Tabs()
    output_cards = {}

    all_tabs = [
        "🧠 Data Analyst", "👥 Recruteur", "⚖️ RH", "🎓 Talent",
        "👋 Onboarding", "💰 Coûts", "🧪 Critique", "✅ Synthèse finale",
        "📊 Rapport Global"  # nouvel onglet ajouté ici
    ]

    for name in all_tabs:
        with output_tabs:
            with gr.Tab(name):
                output_cards[name] = gr.Markdown("En attente...")

    def maj_interface(query):
        result = analyser_faisabilite(query)
        if "Erreur" in result:
            return [result["Erreur"]] * len(output_cards)
        return [result.get(k, "(Aucune donnée)") for k in output_cards]

    bouton.click(fn=maj_interface, inputs=[query_input], outputs=list(output_cards.values()))

demo.launch()
