// Memory Curation Interface JavaScript

class MemoryCurationApp {
    constructor() {
        this.rules = [];
        this.filteredRules = [];
        this.conflicts = [];
        this.stats = {};
        this.enums = {};
        this.currentPage = 1;
        this.pageSize = 12;
        this.viewMode = 'card';
        
        this.init();
    }
    
    async init() {
        await this.loadEnums();
        await this.loadRules();
        await this.loadConflicts();
        await this.loadStats();
        
        this.setupEventListeners();
        this.initializeView();
    }
    
    async loadEnums() {
        try {
            const response = await fetch('/api/enums');
            this.enums = await response.json();
            this.populateSelectOptions();
        } catch (error) {
            console.error('Failed to load enums:', error);
        }
    }
    
    async loadRules() {
        try {
            const response = await fetch('/api/rules');
            const data = await response.json();
            this.rules = data.rules;
            this.filteredRules = [...this.rules];
            this.renderRules();
            this.updateRulesCount();
        } catch (error) {
            console.error('Failed to load rules:', error);
            this.showError('Failed to load memory rules');
        }
    }
    
    async loadConflicts() {
        try {
            const response = await fetch('/api/conflicts');
            const data = await response.json();
            this.conflicts = data.conflicts;
            this.renderConflicts();
        } catch (error) {
            console.error('Failed to load conflicts:', error);
        }
    }
    
    async loadStats() {
        try {
            const response = await fetch('/api/stats');
            this.stats = await response.json();
            this.renderAnalytics();
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }
    
    populateSelectOptions() {
        // Category filters
        const categoryFilter = document.getElementById('category-filter');
        const ruleCategorySelect = document.getElementById('rule-category');
        
        this.enums.categories.forEach(category => {
            const option1 = new Option(this.capitalizeFirst(category), category);
            const option2 = new Option(this.capitalizeFirst(category), category);
            categoryFilter.appendChild(option1);
            ruleCategorySelect.appendChild(option2);
        });
        
        // Authority filters
        const authorityFilter = document.getElementById('authority-filter');
        const ruleAuthoritySelect = document.getElementById('rule-authority');
        
        this.enums.authorities.forEach(authority => {
            const option1 = new Option(this.capitalizeFirst(authority), authority);
            const option2 = new Option(this.capitalizeFirst(authority), authority);
            authorityFilter.appendChild(option1);
            ruleAuthoritySelect.appendChild(option2);
        });
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('[data-view]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchView(e.target.getAttribute('data-view'));
            });
        });
        
        // View mode toggle
        document.querySelectorAll('input[name="view-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.viewMode = e.target.id.replace('-view', '');
                this.renderRules();
            });
        });
        
        // Filters
        document.getElementById('search-input').addEventListener('input', 
            this.debounce(() => this.applyFilters(), 300));
        document.getElementById('category-filter').addEventListener('change', () => this.applyFilters());
        document.getElementById('authority-filter').addEventListener('change', () => this.applyFilters());
        document.getElementById('scope-filter').addEventListener('input', 
            this.debounce(() => this.applyFilters(), 300));
        document.getElementById('clear-filters').addEventListener('click', () => this.clearFilters());
        
        // Actions
        document.getElementById('add-rule-btn').addEventListener('click', () => this.showRuleModal());
        document.getElementById('import-rules-btn').addEventListener('click', () => this.showImportDialog());
        document.getElementById('export-rules-btn').addEventListener('click', () => this.exportRules());
        
        // Rule form
        document.getElementById('rule-form').addEventListener('submit', (e) => this.handleRuleSubmit(e));
    }
    
    initializeView() {
        this.switchView('rules');
    }
    
    switchView(viewName) {
        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-view="${viewName}"]`).classList.add('active');
        
        // Update view
        document.querySelectorAll('.view').forEach(view => {
            view.classList.remove('active');
        });
        document.getElementById(`${viewName}-view`).classList.add('active');
        
        // Load data if needed
        if (viewName === 'conflicts' && this.conflicts.length === 0) {
            this.loadConflicts();
        } else if (viewName === 'analytics') {
            this.renderAnalytics();
        }
    }
    
    applyFilters() {
        const search = document.getElementById('search-input').value.toLowerCase();
        const category = document.getElementById('category-filter').value;
        const authority = document.getElementById('authority-filter').value;
        const scope = document.getElementById('scope-filter').value.toLowerCase();
        
        this.filteredRules = this.rules.filter(rule => {
            const matchesSearch = !search || 
                rule.name.toLowerCase().includes(search) ||
                rule.rule.toLowerCase().includes(search);
                
            const matchesCategory = !category || rule.category === category;
            const matchesAuthority = !authority || rule.authority === authority;
            const matchesScope = !scope || 
                (rule.scope && rule.scope.some(s => s.toLowerCase().includes(scope)));
                
            return matchesSearch && matchesCategory && matchesAuthority && matchesScope;
        });
        
        this.currentPage = 1;
        this.renderRules();
        this.updateRulesCount();
    }
    
    clearFilters() {
        document.getElementById('search-input').value = '';
        document.getElementById('category-filter').value = '';
        document.getElementById('authority-filter').value = '';
        document.getElementById('scope-filter').value = '';
        
        this.filteredRules = [...this.rules];
        this.currentPage = 1;
        this.renderRules();
        this.updateRulesCount();
    }
    
    renderRules() {
        const container = document.getElementById('rules-container');
        const startIndex = (this.currentPage - 1) * this.pageSize;
        const endIndex = startIndex + this.pageSize;
        const pageRules = this.filteredRules.slice(startIndex, endIndex);
        
        if (pageRules.length === 0) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i class="fas fa-search fa-3x text-muted mb-3"></i>
                    <h5 class="text-muted">No rules found</h5>
                    <p class="text-muted">Try adjusting your filters or add a new rule.</p>
                </div>
            `;
            return;
        }
        
        if (this.viewMode === 'card') {
            this.renderCardView(container, pageRules);
        } else {
            this.renderTableView(container, pageRules);
        }
        
        this.renderPagination();
    }
    
    renderCardView(container, rules) {
        const cardHtml = rules.map(rule => `
            <div class="col-md-6 col-lg-4">
                <div class="card rule-card fade-in" data-rule-id="${rule.id}">
                    <div class="card-header">
                        <div>
                            <h6 class="rule-title">${this.escapeHtml(rule.name)}</h6>
                            <div class="mt-1">
                                <span class="category-badge category-${rule.category}">
                                    ${this.capitalizeFirst(rule.category)}
                                </span>
                                <span class="authority-badge authority-${rule.authority}">
                                    ${this.capitalizeFirst(rule.authority)}
                                </span>
                            </div>
                        </div>
                        <div class="rule-actions">
                            <button class="btn btn-sm btn-outline-primary" onclick="app.editRule('${rule.id}')">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-danger" onclick="app.deleteRule('${rule.id}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="rule-content" id="content-${rule.id}">
                            <p class="card-text">${this.escapeHtml(rule.rule)}</p>
                        </div>
                        ${rule.rule.length > 150 ? `
                            <button class="expand-button" onclick="app.toggleRuleContent('${rule.id}')">
                                Show more...
                            </button>
                        ` : ''}
                        ${rule.scope && rule.scope.length > 0 ? `
                            <div class="scope-tags">
                                ${rule.scope.map(tag => `
                                    <span class="scope-tag">${this.escapeHtml(tag)}</span>
                                `).join('')}
                            </div>
                        ` : ''}
                        <small class="text-muted">
                            ${rule.source || 'Unknown source'} • 
                            ${rule.created_at ? new Date(rule.created_at).toLocaleDateString() : 'Unknown date'}
                        </small>
                    </div>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = `<div class="row">${cardHtml}</div>`;
    }
    
    renderTableView(container, rules) {
        const tableHtml = `
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Category</th>
                            <th>Authority</th>
                            <th>Rule</th>
                            <th>Scope</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rules.map(rule => `
                            <tr data-rule-id="${rule.id}">
                                <td>
                                    <strong>${this.escapeHtml(rule.name)}</strong>
                                    <br><small class="text-muted">${rule.source || 'Unknown'}</small>
                                </td>
                                <td>
                                    <span class="category-badge category-${rule.category}">
                                        ${this.capitalizeFirst(rule.category)}
                                    </span>
                                </td>
                                <td>
                                    <span class="authority-badge authority-${rule.authority}">
                                        ${this.capitalizeFirst(rule.authority)}
                                    </span>
                                </td>
                                <td>
                                    <div class="text-truncate-2" style="max-width: 300px;">
                                        ${this.escapeHtml(rule.rule)}
                                    </div>
                                </td>
                                <td>
                                    ${rule.scope && rule.scope.length > 0 ? 
                                        rule.scope.slice(0, 2).map(tag => `
                                            <span class="scope-tag">${this.escapeHtml(tag)}</span>
                                        `).join(' ') + 
                                        (rule.scope.length > 2 ? ` +${rule.scope.length - 2}` : '')
                                        : '-'
                                    }
                                </td>
                                <td>
                                    <div class="btn-group btn-group-sm">
                                        <button class="btn btn-outline-primary" onclick="app.editRule('${rule.id}')">
                                            <i class="fas fa-edit"></i>
                                        </button>
                                        <button class="btn btn-outline-danger" onclick="app.deleteRule('${rule.id}')">
                                            <i class="fas fa-trash"></i>
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        
        container.innerHTML = tableHtml;
    }
    
    renderPagination() {
        const totalPages = Math.ceil(this.filteredRules.length / this.pageSize);
        const paginationContainer = document.getElementById('pagination');
        const paginationNav = document.getElementById('pagination-nav');
        
        if (totalPages <= 1) {
            paginationNav.style.display = 'none';
            return;
        }
        
        paginationNav.style.display = 'block';
        
        let paginationHtml = '';
        
        // Previous button
        paginationHtml += `
            <li class="page-item ${this.currentPage === 1 ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="app.changePage(${this.currentPage - 1})">
                    <i class="fas fa-chevron-left"></i>
                </a>
            </li>
        `;
        
        // Page numbers
        const startPage = Math.max(1, this.currentPage - 2);
        const endPage = Math.min(totalPages, startPage + 4);
        
        for (let i = startPage; i <= endPage; i++) {
            paginationHtml += `
                <li class="page-item ${i === this.currentPage ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="app.changePage(${i})">${i}</a>
                </li>
            `;
        }
        
        // Next button
        paginationHtml += `
            <li class="page-item ${this.currentPage === totalPages ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="app.changePage(${this.currentPage + 1})">
                    <i class="fas fa-chevron-right"></i>
                </a>
            </li>
        `;
        
        paginationContainer.innerHTML = paginationHtml;
    }
    
    changePage(page) {
        const totalPages = Math.ceil(this.filteredRules.length / this.pageSize);
        if (page >= 1 && page <= totalPages) {
            this.currentPage = page;
            this.renderRules();
        }
    }
    
    updateRulesCount() {
        const countElement = document.getElementById('rules-count');
        countElement.textContent = `Showing ${this.filteredRules.length} of ${this.rules.length} rules`;
    }
    
    toggleRuleContent(ruleId) {
        const contentElement = document.getElementById(`content-${ruleId}`);
        const button = contentElement.nextElementSibling;
        
        if (contentElement.classList.contains('expanded')) {
            contentElement.classList.remove('expanded');
            button.textContent = 'Show more...';
        } else {
            contentElement.classList.add('expanded');
            button.textContent = 'Show less';
        }
    }
    
    showRuleModal(rule = null) {
        const modal = new bootstrap.Modal(document.getElementById('rule-modal'));
        const modalTitle = document.getElementById('rule-modal-title');
        const form = document.getElementById('rule-form');
        
        if (rule) {
            modalTitle.textContent = 'Edit Rule';
            form.elements.name.value = rule.name;
            form.elements.category.value = rule.category;
            form.elements.authority.value = rule.authority;
            form.elements.rule.value = rule.rule;
            form.elements.scope.value = rule.scope ? rule.scope.join(', ') : '';
            form.dataset.ruleId = rule.id;
        } else {
            modalTitle.textContent = 'Add Rule';
            form.reset();
            delete form.dataset.ruleId;
        }
        
        modal.show();
    }
    
    async handleRuleSubmit(e) {
        e.preventDefault();
        
        const form = e.target;
        const formData = new FormData(form);
        const ruleId = form.dataset.ruleId;
        
        try {
            const url = ruleId ? `/api/rules/${ruleId}` : '/api/rules';
            const method = ruleId ? 'PUT' : 'POST';
            
            const response = await fetch(url, {
                method: method,
                body: formData
            });
            
            if (response.ok) {
                const modal = bootstrap.Modal.getInstance(document.getElementById('rule-modal'));
                modal.hide();
                
                await this.loadRules();
                await this.loadStats();
                
                this.showSuccess(ruleId ? 'Rule updated successfully' : 'Rule created successfully');
            } else {
                const error = await response.json();
                this.showError(error.detail || 'Failed to save rule');
            }
        } catch (error) {
            this.showError('Failed to save rule: ' + error.message);
        }
    }
    
    async editRule(ruleId) {
        try {
            const response = await fetch(`/api/rules/${ruleId}`);
            const rule = await response.json();
            this.showRuleModal(rule);
        } catch (error) {
            this.showError('Failed to load rule for editing');
        }
    }
    
    async deleteRule(ruleId) {
        if (!confirm('Are you sure you want to delete this rule?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/rules/${ruleId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                await this.loadRules();
                await this.loadStats();
                this.showSuccess('Rule deleted successfully');
            } else {
                this.showError('Failed to delete rule');
            }
        } catch (error) {
            this.showError('Failed to delete rule: ' + error.message);
        }
    }
    
    renderConflicts() {
        const container = document.getElementById('conflicts-container');
        
        if (this.conflicts.length === 0) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i class="fas fa-check-circle fa-3x text-success mb-3"></i>
                    <h5 class="text-success">No conflicts detected</h5>
                    <p class="text-muted">Your memory rules are working in harmony.</p>
                </div>
            `;
            return;
        }
        
        const conflictsHtml = this.conflicts.map(conflict => `
            <div class="card conflict-card">
                <div class="card-header">
                    <h6 class="mb-0">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        ${this.capitalizeFirst(conflict.conflict_type)} Conflict
                        <span class="badge bg-warning text-dark ms-2">
                            ${Math.round(conflict.confidence * 100)}% confidence
                        </span>
                    </h6>
                </div>
                <div class="card-body">
                    <p class="text-muted">${conflict.description}</p>
                    
                    <div class="conflict-comparison">
                        <div class="conflict-rule border-start-primary">
                            <h6>${this.escapeHtml(conflict.rule1.name)}</h6>
                            <p class="small mb-2">${this.escapeHtml(conflict.rule1.rule)}</p>
                            <span class="authority-badge authority-${conflict.rule1.authority}">
                                ${this.capitalizeFirst(conflict.rule1.authority)}
                            </span>
                        </div>
                        
                        <div class="conflict-rule border-start-warning">
                            <h6>${this.escapeHtml(conflict.rule2.name)}</h6>
                            <p class="small mb-2">${this.escapeHtml(conflict.rule2.rule)}</p>
                            <span class="authority-badge authority-${conflict.rule2.authority}">
                                ${this.capitalizeFirst(conflict.rule2.authority)}
                            </span>
                        </div>
                    </div>
                    
                    <div class="conflict-actions">
                        ${conflict.resolution_options.map((option, index) => `
                            <button class="btn btn-outline-primary btn-sm" 
                                    onclick="app.resolveConflict('${conflict.rule1.id}', '${conflict.rule2.id}', ${index})">
                                ${option}
                            </button>
                        `).join('')}
                    </div>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = conflictsHtml;
    }
    
    async resolveConflict(rule1Id, rule2Id, optionIndex) {
        // This would need to be implemented with the actual conflict resolution API
        this.showInfo('Conflict resolution not yet implemented');
    }
    
    renderAnalytics() {
        this.renderTokenUsage();
        this.renderCategoryChart();
        this.renderAuthorityChart();
    }
    
    renderTokenUsage() {
        const container = document.getElementById('token-usage-content');
        const { estimated_tokens, total_rules } = this.stats;
        
        const maxTokens = 4000; // Assumed limit
        const percentage = (estimated_tokens / maxTokens) * 100;
        let level = 'low';
        if (percentage > 70) level = 'high';
        else if (percentage > 40) level = 'medium';
        
        container.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="stat-card">
                        <span class="stat-value">${total_rules}</span>
                        <span class="stat-label">Total Rules</span>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="stat-card">
                        <span class="stat-value">${estimated_tokens}</span>
                        <span class="stat-label">Estimated Tokens</span>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="stat-card">
                        <span class="stat-value">${Math.round(percentage)}%</span>
                        <span class="stat-label">Memory Usage</span>
                    </div>
                </div>
            </div>
            
            <div class="progress-wrapper">
                <div class="progress-info">
                    <span>Token Usage</span>
                    <span>${estimated_tokens} / ${maxTokens}</span>
                </div>
                <div class="token-meter">
                    <div class="token-fill ${level}" style="width: ${Math.min(percentage, 100)}%"></div>
                    <div class="token-info">${Math.round(percentage)}%</div>
                </div>
            </div>
            
            ${percentage > 80 ? `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    High token usage detected. Consider optimizing your rules.
                </div>
            ` : ''}
        `;
    }
    
    renderCategoryChart() {
        const ctx = document.getElementById('category-chart').getContext('2d');
        const data = this.stats.rules_by_category;
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(data).map(cat => this.capitalizeFirst(cat)),
                datasets: [{
                    data: Object.values(data),
                    backgroundColor: [
                        '#0dcaf0', // info
                        '#198754', // success
                        '#ffc107'  // warning
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    renderAuthorityChart() {
        const ctx = document.getElementById('authority-chart').getContext('2d');
        const data = this.stats.rules_by_authority;
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(data).map(auth => this.capitalizeFirst(auth)),
                datasets: [{
                    label: 'Rules',
                    data: Object.values(data),
                    backgroundColor: [
                        '#dc3545', // absolute - danger
                        '#ffc107', // high - warning
                        '#0d6efd', // default - primary
                        '#6c757d'  // low - secondary
                    ]
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    async exportRules() {
        try {
            const response = await fetch('/api/rules');
            const data = await response.json();
            
            const blob = new Blob([JSON.stringify(data.rules, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `memory-rules-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showSuccess('Rules exported successfully');
        } catch (error) {
            this.showError('Failed to export rules');
        }
    }
    
    showImportDialog() {
        // Create a file input dialog
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            try {
                const text = await file.text();
                const rules = JSON.parse(text);
                
                // Validate the imported data
                if (!Array.isArray(rules)) {
                    throw new Error('Invalid file format: Expected an array of rules');
                }
                
                this.showInfo(`Found ${rules.length} rules. Import functionality not yet implemented.`);
            } catch (error) {
                this.showError('Failed to parse import file: ' + error.message);
            }
        };
        
        input.click();
    }
    
    // Utility functions
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showSuccess(message) {
        this.showToast(message, 'success');
    }
    
    showError(message) {
        this.showToast(message, 'danger');
    }
    
    showInfo(message) {
        this.showToast(message, 'info');
    }
    
    showToast(message, type = 'info') {
        // Create a simple toast notification
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        toast.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }
}

// Initialize the application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MemoryCurationApp();
});