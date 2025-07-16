class SmartRHAgent {
    constructor() {
        this.apiUrl = 'http://127.0.0.1:8000/feasibility/invoke';
        this.agents = [
            { key: 'data_analytics', title: '🧠 Data Analyst', id: 'data-analyst' },
            { key: 'recruiter', title: '👥 Recruteur', id: 'recruiter' },
            { key: 'rh', title: '⚖️ RH', id: 'rh' },
            { key: 'talent', title: '🎓 Talent', id: 'talent' },
            { key: 'onboarding', title: '👋 Onboarding', id: 'onboarding' },
            { key: 'payroll', title: '💰 Coûts', id: 'payroll' },
            { key: 'critique', title: '🧪 Critique', id: 'critique' },
            { key: 'final_answer', title: '✅ Synthèse finale', id: 'final-answer' }
        ];
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        const projectInput = document.getElementById('projectInput');

        analyzeBtn.addEventListener('click', () => this.analyzeProject());
        
        projectInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                this.analyzeProject();
            }
        });

        // Auto-resize textarea
        projectInput.addEventListener('input', () => {
            projectInput.style.height = 'auto';
            projectInput.style.height = (projectInput.scrollHeight) + 'px';
        });
    }

    async analyzeProject() {
        const query = document.getElementById('projectInput').value.trim();
        
        if (!query) {
            this.showError('Veuillez entrer un projet à analyser.');
            return;
        }

        this.setLoading(true);
        this.hideError();
        this.showResultsSection();
        this.createAgentCards();

        try {
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Erreur lors de l'analyse : ${error.message}`);
        } finally {
            this.setLoading(false);
        }
    }

    setLoading(isLoading) {
        const btn = document.getElementById('analyzeBtn');
        const btnText = btn.querySelector('.btn-text');
        const spinner = btn.querySelector('.loading-spinner');

        if (isLoading) {
            btn.disabled = true;
            btnText.style.display = 'none';
            spinner.style.display = 'inline-block';
        } else {
            btn.disabled = false;
            btnText.style.display = 'inline-block';
            spinner.style.display = 'none';
        }
    }

    showResultsSection() {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';
    }

    createAgentCards() {
        const agentsGrid = document.getElementById('agentsGrid');
        agentsGrid.innerHTML = '';

        this.agents.forEach(agent => {
            const card = document.createElement('div');
            card.className = 'agent-card loading';
            card.id = agent.id;
            
            card.innerHTML = `
                <div class="agent-header">
                    <h2 class="agent-title">${agent.title}</h2>
                </div>
                <div class="agent-content">
                    <div class="skeleton skeleton-text medium"></div>
                    <div class="skeleton skeleton-text long"></div>
                    <div class="skeleton skeleton-text short"></div>
                    <div class="skeleton skeleton-text long"></div>
                    <div class="skeleton skeleton-text medium"></div>
                </div>
            `;
            
            agentsGrid.appendChild(card);
        });
    }

    displayResults(data) {
        this.agents.forEach(agent => {
            const card = document.getElementById(agent.id);
            if (card) {
                card.classList.remove('loading');
                const content = card.querySelector('.agent-content');
                content.innerHTML = this.formatAgentResponse(agent.key, data[agent.key] || {});
            }
        });
    }

    formatAgentResponse(agentKey, data) {
        switch (agentKey) {
            case 'data_analytics':
                return this.formatDataAnalytics(data);
            case 'recruiter':
                return this.formatRecruiter(data);
            case 'rh':
                return this.formatRH(data);
            case 'talent':
                return this.formatTalent(data);
            case 'onboarding':
                return this.formatOnboarding(data);
            case 'payroll':
                return this.formatPayroll(data);
            case 'critique':
                return this.formatCritique(data);
            case 'final_answer':
                return this.formatFinalAnswer(data);
            default:
                return '<p>Aucune donnée disponible</p>';
        }
    }

    formatDataAnalytics(data) {
        const competencesMissing = data.competences_manquantes || [];
        const isFeasible = competencesMissing.length === 0;
        
        return `
            <div class="status-badge ${isFeasible ? 'status-faisable' : 'status-non-faisable'}">
                ${isFeasible ? 'Faisable' : 'Très peu faisable'}
            </div>
            <h3>Analyse budgétaire</h3>
            <p><strong>Budget utilisateur :</strong> ${data.budget_utilisateur || 'N/A'} €</p>
            <p><strong>Budget moyen observé :</strong> ${data.budget_moyen || 'N/A'} €</p>
            
            <h3>Délais</h3>
            <p><strong>Délai utilisateur :</strong> ${data.delai_utilisateur_jours || 'N/A'} jours</p>
            <p><strong>Délai moyen observé :</strong> ${data.delai_moyen_lancement_projet || 'N/A'} jours</p>
            
            <h3>Compétences</h3>
            <p><strong>Compétences couvertes :</strong> ${this.formatCompetences(data.competences_couvertes)}</p>
            <p><strong>Compétences manquantes :</strong> ${this.formatList(competencesMissing)}</p>
            
            ${data.warnings ? `<p style="color: #e74c3c;">⚠️ ${data.warnings}</p>` : ''}
        `;
    }

    formatRecruiter(data) {
        return `
            <h3>Résultats de recherche</h3>
            <p><strong>Profils trouvés :</strong> ${data.count || 0} profils</p>
            <p><strong>Compétences recherchées :</strong> ${this.formatList(data.competences_recherchees_finales)}</p>
            <p><strong>Compétences non trouvées :</strong> ${this.formatList(data.competences_non_trouvees)}</p>
            
            ${data.message ? `<p style="color: #e74c3c;">⚠️ ${data.message}</p>` : ''}
        `;
    }

    formatRH(data) {
        const isCompliant = data.compliance_status === 'conforme';
        
        return `
            <div class="status-badge ${isCompliant ? 'status-conforme' : 'status-non-conforme'}">
                ${isCompliant ? '✅ Conforme' : '❌ Non conforme'}
            </div>
            <h3>Analyse de conformité</h3>
            <p>${data.response || 'Aucune information disponible'}</p>
        `;
    }

    formatTalent(data) {
        return `
            <h3>Possibilité d'upskilling</h3>
            <div class="status-badge ${data.upskill_possible ? 'status-faisable' : 'status-non-faisable'}">
                ${data.upskill_possible ? '✅ Possible' : '❌ Non possible'}
            </div>
            <h3>Plan de développement</h3>
            <p>${data.plan_global || 'Aucun plan disponible'}</p>
        `;
    }

    formatOnboarding(data) {
        const accueil = data.accueil || {};
        const resources = accueil.resources || {};
        
        return `
            <h3>Ressources disponibles</h3>
            <p><strong>RH :</strong> ${resources.rh || 'N/A'}</p>
            <p><strong>Tuteurs :</strong> ${resources.tutors || 'N/A'}</p>
            
            <h3>Checklist</h3>
            <p>${this.formatList(accueil.checklist)}</p>
            
            <h3>Risques identifiés</h3>
            <p>${this.formatList(accueil.risks)}</p>
        `;
    }

    formatPayroll(data) {
        const isBudgetSufficient = data.budget_suffisant === 'oui';
        
        return `
            <h3>Estimation des coûts</h3>
            <p><strong>Coût mensuel estimé :</strong> ${data.cout_mensuel || 'N/A'}</p>
            <div class="status-badge ${isBudgetSufficient ? 'status-faisable' : 'status-non-faisable'}">
                ${isBudgetSufficient ? '✅ Budget suffisant' : '❌ Budget insuffisant'}
            </div>
            
            <h3>Analyse détaillée</h3>
            <p>${data.analyse_cout || 'Aucune analyse disponible'}</p>
        `;
    }

    formatCritique(data) {
        return `
            <h3>Points forts</h3>
            <p>${this.formatList(data.points_forts)}</p>
            
            <h3>Points faibles</h3>
            <p>${this.formatList(data.points_faibles)}</p>
            
            <h3>Suggestions</h3>
            <p>${this.formatList(data.suggestions)}</p>
            
            <h3>Note de cohérence</h3>
            <p><strong>${data.note_coherence || 'N/A'}/5</strong></p>
        `;
    }

    formatFinalAnswer(data) {
        return `
            <h3>Faisabilité globale</h3>
            <p><strong>${data.faisabilite || 'N/A'}</strong></p>
            
            <h3>Score de confiance</h3>
            <p><strong>${data.score_confiance || 'N/A'}</strong></p>
            
            <h3>Conditions de réussite</h3>
            <p>${this.formatList(data.conditions_reussite)}</p>
            
            <h3>Risques principaux</h3>
            <p>${this.formatList(data.risques_principaux)}</p>
        `;
    }

    formatList(items) {
        if (!items) return 'Aucun élément';
        if (Array.isArray(items)) {
            return items.length > 0 ? items.join(', ') : 'Aucun élément';
        }
        return items.toString();
    }

    formatCompetences(competences) {
        if (!competences || !Array.isArray(competences)) return 'Aucune compétence';
        return competences.map(c => c.competence || c).join(', ');
    }

    showError(message) {
        const errorElement = document.getElementById('errorMessage');
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        const errorElement = document.getElementById('errorMessage');
        errorElement.style.display = 'none';
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new SmartRHAgent();
});