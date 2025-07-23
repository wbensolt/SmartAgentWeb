Voici le **template prêt à être publié** dans ton fichier `README.md` GitHub pour ton projet **Smart RH Agent Web** :

---

```markdown
# 🤖 Smart RH Agent Web – Multi-Agent LLM RH Project

## 🧠 Contexte

Ce projet a été réalisé dans le cadre d’un travail de recherche et développement autour des **Systèmes Multi-Agents (SMA)** orchestrés par un **LLM central**, dans un contexte **Ressources Humaines (RH)**.

L’objectif : concevoir une **application web intelligente** capable de traiter diverses tâches RH en simulant des compétences humaines à l’aide d’agents spécialisés (data analyst, recruteur, onboarding, RH généraliste...).

Le projet s’appuie sur les dernières avancées en IA générative, vectorisation de documents, chaînes de raisonnement et orchestration d’agents via **LangGraph**.

---

## 🎯 Objectif

Développer un **assistant RH intelligent** avec :

- Un orchestrateur LLM qui répartit les requêtes vers les agents RH.
- Des agents spécialisés pour chaque tâche (recrutement, intégration, paie, etc.).
- Une architecture modulaire avec LangGraph.
- Une interface web conviviale avec Chainlit.
- Un système de vectorisation de documents RH (ChromaDB).
- Une réponse contextualisée, interprétable et traçable.

---

## 🧰 Stack Technique

- Python 3.12
- LangChain / LangGraph
- Ollama / Groq (LLMs)
- Chainlit
- ChromaDB
- SQLite (base locale)
- Docker
- Git / GitHub

---

## 📁 Structure du Projet

```

SmartAgentWeb/
├── agents/
│   ├── vector/
│   │   ├── retrieve\_project.py
│   │   └── state\_schema.py
│   └── nodes/hr\_agents/
│       ├── tools/data\_analyst.py
│       ├── recruiter\_agent.py
│       ├── onboarding\_agent.py
│       ├── rhagent.py
│       ├── talent\_agent.py
│       └── payroll\_agent.py
├── core/
│   ├── llm\_providers.py
│   └── router.py
├── vectorstore/
│   └── chroma\_index/
├── chainlit\_app.py
├── requirements.txt
└── README.md

````

---

## 🏗️ Architectures SMA RH Adoptées

### ✅ Version 1 : SMA Orchestré par LLM Central

- LLM central (Groq / Ollama) qui route les requêtes.
- Agents outils sans mémoire ni logique propre.
- Avantage : simplicité ; Inconvénient : dépendance forte au LLM.

### ✅ Version 2 : Agents autonomes via LangGraph

- Orchestration basée sur l'état partagé (`GraphState`)
- Agents avec prompts spécialisés, logique métier, raisonnement local
- Possibilité de chaînage conditionnel et vectorisation RH via ChromaDB

---

## 🧠 Agents RH intégrés

| Agent            | Compétence simulée                 |
|------------------|------------------------------------|
| `data_analyst`   | Analyse RH, reporting              |
| `recruiter`      | Matching candidat / poste          |
| `onboarding`     | Parcours d’intégration             |
| `rh`             | Questions RH classiques            |
| `talent`         | Évolution, formation, mobilité     |
| `payroll`        | Rémunération, fiche de paie        |

---

## 🚀 Lancer l’application

1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
````

2. **Lancer l’app Chainlit** :

   ```bash
   chainlit run chainlit_app.py
   ```

3. **(Optionnel) Rendre l’application publique** :

   ```bash
   chainlit run chainlit_app.py --public
   ```

---

## 🧪 Exemple d'utilisation

* L’utilisateur envoie une requête RH (ex : *"Quels profils sont adaptés au poste de data analyst ?"*)
* Le `router.py` l'analyse et la redirige vers les bons agents
* Les agents coopèrent pour fournir une réponse complète et contextualisée

---

## 📊 Suivi & Interprétation

* Logs interactifs via Chainlit
* Visualisation de la chaîne d’agents activés
* Affichage transparent du raisonnement et des données utilisées
* Possibilité de debugging en mode développeur

---

## 🔁 Reproductibilité

* Agents isolés et testables indépendamment
* Seeds fixés si usage d’éléments stochastiques
* Index vectoriels reproductibles
* Compatible Docker

---

## 👥 Auteurs

* **Wael Bensoltana**
  [GitHub](https://github.com/waelbensoltana)

---

## 🔗 Liens Utiles

* [LangChain](https://www.langchain.com/)
* [LangGraph](https://www.langchain.com/langgraph)
* [Chainlit](https://www.chainlit.io/)
* [ChromaDB](https://docs.trychroma.com/)
* [Groq](https://groq.com/)
* [Ollama](https://ollama.com/)

---

## 🏁 Recommandations d’utilisation

| Action                  | Commande                                                 |
| ----------------------- | -------------------------------------------------------- |
| Lancer l'entraînement   | *(non applicable ici)*                                   |
| Lancer l’application RH | `chainlit run chainlit_app.py`                           |
| Voir les logs           | Intégrés directement dans l'interface Chainlit           |
| Ajouter un nouvel agent | Créer dans `agents/nodes/hr_agents/` + ajouter au router |

```

---
```
