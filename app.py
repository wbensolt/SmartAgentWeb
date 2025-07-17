import streamlit as st
import requests
import pandas as pd
import ast

API_URL = "http://127.0.0.1:8000/feasibility/invoke"

def safe_convert_to_list(data):
    """Convertit en toute sécurité une chaîne en liste"""
    if isinstance(data, str):
        try:
            # Essaye d'évaluer la chaîne comme une liste Python
            if data.startswith('[') and data.endswith(']'):
                return ast.literal_eval(data)
            # Si ce n'est pas une liste mais séparée par des virgules
            return [item.strip() for item in data.split(',') if item.strip()]
        except:
            return [data]
    elif isinstance(data, list):
        return data
    return []

def format_as_table(items, title=""):
    """Transforme une liste ou une chaîne en DataFrame pour affichage sous forme de tableau"""
    items = safe_convert_to_list(items)
    if not items:
        return None
    return pd.DataFrame(items, columns=[title])

def analyser_faisabilite(query: str):
    try:
        response = requests.post(API_URL, json={"query": query})
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {"Erreur": f"Impossible d'appeler l'API : {str(e)}"}

    cards = {}

    # 🧠 Data Analyst
    da = data.get("data_analytics", {})
    
    # Compétences couvertes
    competences_couvertes = da.get("competences_couvertes", [])
    if isinstance(competences_couvertes, list) and competences_couvertes and isinstance(competences_couvertes[0], dict):
        competences_couvertes = [f"✅ {c['competence']}" for c in competences_couvertes]
    elif isinstance(competences_couvertes, str):
        competences_couvertes = [f"✅ {c.strip()}" for c in competences_couvertes.split('✅') if c.strip()]
    
    df_competences_couvertes = format_as_table(competences_couvertes, "Compétences")

    # Compétences manquantes
    competences_manquantes = da.get("competences_manquantes", [])
    if isinstance(competences_manquantes, str):
        competences_manquantes = [f"❌ {c.strip()}" for c in competences_manquantes.replace("❌ Compétence manquante :", "").split('❌') if c.strip()]
    
    df_competences_manquantes = format_as_table(competences_manquantes, "Compétences")

    # Alertes
    warnings = da.get("warnings", [])
    if isinstance(warnings, str):
        warnings = [f"⚠️ {w.strip()}" for w in warnings.split('⚠️') if w.strip()]
    
    df_warnings = format_as_table(warnings, "Alertes")

    # Construction de la carte Data Analyst
    data_analyst_content = f"""
### 💬 **Conclusion :**  
**{'Faisable' if not competences_manquantes else 'Très peu faisable'}**

💰 **Budget utilisateur :** {da.get('budget_utilisateur', 'N/A')} €  
📊 **Budget moyen observé :** {da.get('budget_moyen', 'N/A')} €  
⏳ **Délai utilisateur :** {da.get('delai_utilisateur_jours', 'N/A')} jours  
⌛ **Délai moyen observé :** {da.get('delai_moyen_lancement_projet', 'N/A')} jours  
"""

    cards["🧠 Data Analyst"] = {
        "content": data_analyst_content,
        "tables": {
            "Compétences couvertes": df_competences_couvertes,
            "Compétences manquantes": df_competences_manquantes,
            "Alertes": df_warnings
        }
    }

    # 👥 Recruiter
    recruiter = data.get("recruiter", {})

    # Compétences recherchées
    competences_recherchees = recruiter.get("competences_recherchees_finales", [])
    if isinstance(competences_recherchees, str):
        competences_recherchees = [f"🔍 {c.strip()}" for c in competences_recherchees.split('🔍') if c.strip()]
    
    df_competences_recherchees = format_as_table(competences_recherchees, "Compétences")

    # Compétences non trouvées
    competences_non_trouvees = recruiter.get("competences_non_trouvees", [])
    if isinstance(competences_non_trouvees, str):
        competences_non_trouvees = [f"🚫 {c.strip()}" for c in competences_non_trouvees.split('🚫') if c.strip()]
    
    df_competences_non_trouvees = format_as_table(competences_non_trouvees, "Compétences")

    recruiter_content = f"""
👤 **Profils trouvés :** {recruiter.get('count', 0)} profils  
⚠️ **Message :** {recruiter.get('message', 'Aucun message disponible')}
"""

    cards["👥 Recruteur"] = {
        "content": recruiter_content,
        "tables": {
            "Compétences recherchées": df_competences_recherchees,
            "Compétences non trouvées": df_competences_non_trouvees
        }
    }

    # ⚖️ RH
    rh = data.get("rh", {})
    cards["⚖️ RH"] = {
        "content": f"""
⚖️ **Conformité juridique :** {'✅ Conforme' if rh.get('compliance_status') == 'conforme' else '❌ Non conforme'}  
{rh.get('response', 'Aucune réponse disponible')}
        """
    }

    # 🎓 Talent
    talent = data.get("talent", {})
    cards["🎓 Talent"] = {
        "content": f"""
🎯 **Upskill possible :** {'✅ Oui' if talent.get('upskill_possible') else '❌ Non'}  
{talent.get('plan_global', 'Aucun plan disponible')}
        """
    }

    # 👋 Onboarding - Version améliorée
    onboarding = data.get("onboarding", {}).get("accueil", {})
    checklist = safe_convert_to_list(onboarding.get("checklist", []))
    risks = safe_convert_to_list(onboarding.get("risks", []))
    
    df_checklist = format_as_table(checklist, "Étapes")
    df_risks = format_as_table(risks, "Risques")

    cards["👋 Onboarding"] = {
        "content": f"""
👥 **RH :** {onboarding.get('resources', {}).get('rh', 'N/A')}  
👨‍🏫 **Tuteurs :** {onboarding.get('resources', {}).get('tutors', 'N/A')}  
        """,
        "tables": {
            "📋 Checklist": df_checklist,
            "⚠️ Risques": df_risks
        },
        "layout": "columns"  # Marqueur pour un affichage spécial
    }

    # 💰 Payroll
    payroll = data.get("payroll", {})
    cards["💰 Coûts"] = {
        "content": f"""
💸 **Coût mensuel estimé :** {payroll.get('cout_mensuel', 'N/A')}  
💵 **Budget suffisant ?** {'✅ Oui' if payroll.get('budget_suffisant') == 'oui' else '❌ Non'}  
{payroll.get('analyse_cout', 'Aucune analyse disponible')}
        """
    }

    # 🧪 Critique
    critique = data.get("critique", {})
    df_points_forts = format_as_table(critique.get("points_forts", []), "Points forts")
    df_points_faibles = format_as_table(critique.get("points_faibles", []), "Points faibles")
    df_suggestions = format_as_table(critique.get("suggestions", []), "Suggestions")

    cards["🧪 Critique"] = {
        "content": f"⭐ **Note de cohérence :** {critique.get('note_coherence', 'N/A')}/5",
        "tables": {
            "👍 Points forts": df_points_forts,
            "👎 Points faibles": df_points_faibles,
            "💡 Suggestions": df_suggestions
        }
    }

    # ✅ Synthèse finale
    final = data.get("final_answer", {})
    df_conditions = format_as_table(final.get("conditions_reussite", []), "Conditions")
    df_risques = format_as_table(final.get("risques_principaux", []), "Risques")

    cards["✅ Synthèse finale"] = {
        "content": f"""
📈 **Faisabilité :** {final.get('faisabilite', 'N/A')}  
📊 **Score de confiance :** {final.get('score_confiance', 'N/A')}  
        """,
        "tables": {
            "🏆 Conditions de réussite": df_conditions,
            "⚠️ Risques principaux": df_risques
        }
    }

    # 📊 Rapport global
    cards["📊 Rapport Global"] = {
        "content": """
Ce rapport est généré automatiquement à partir de l'analyse croisée de tous les agents RH.  
Il fournit une synthèse transversale de la viabilité du projet avec mise en perspective stratégique.
        """,
        "image": "data/ArchitectureSmartWebAgentRH.png"
    }

    return cards

# --------------------
# Interface Streamlit
# --------------------

# Configuration de la page avec un logo RH personnalisé
st.set_page_config(
    page_title="Smart RH Project Agent",
    layout="wide",
    page_icon="🧑‍💼"  # Icône RH dans l'onglet navigateur (peut aussi être une URL vers une image)
)

# Titre principal avec couleur jaune et texte en gras
st.markdown("<h1 style='color:#FFA500; font-weight:bold;'>📊 Smart RH Project Agent</h1>", unsafe_allow_html=True)

# Sous-titre
st.markdown("**Entrez un projet pour analyser sa faisabilité par nos agents spécialisés.**")

query = st.text_area(
    "Votre projet :",
    height=120,
    placeholder="Ex : Peut-on construire une usine pour avions militaires à Toulouse dans 3 mois ?"
)

if st.button("🔍 Investiguer"):
    with st.spinner("Analyse en cours..."):
        result = analyser_faisabilite(query)
        if "Erreur" in result:
            st.error(result["Erreur"])
        else:
            tabs = st.tabs(list(result.keys()))
            for tab, key in zip(tabs, result.keys()):
                with tab:
                    card = result[key]
                    st.markdown(card["content"], unsafe_allow_html=True)
                    
                    # Affichage spécial pour l'onglet Onboarding
                    if key == "👋 Onboarding":
                        col1, col2 = st.columns(2)
                        with col1:
                            if card["tables"].get("📋 Checklist") is not None:
                                st.subheader("📋 Checklist")
                                st.dataframe(
                                    card["tables"]["📋 Checklist"],
                                    use_container_width=True,
                                    hide_index=True
                                )
                        with col2:
                            if card["tables"].get("⚠️ Risques") is not None:
                                st.subheader("⚠️ Risques")
                                st.dataframe(
                                    card["tables"]["⚠️ Risques"],
                                    use_container_width=True,
                                    hide_index=True
                                )
                    # Affichage standard pour les autres onglets
                    elif "tables" in card:
                        for table_name, table_data in card["tables"].items():
                            if table_data is not None and not table_data.empty:
                                st.subheader(table_name)
                                st.dataframe(
                                    table_data,
                                    use_container_width=True,
                                    hide_index=True
                                )
                    
                    if key == "📊 Rapport Global" and "image" in card:
                        st.image(card["image"], use_container_width=True)