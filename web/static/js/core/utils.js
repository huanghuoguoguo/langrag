/**
 * Utility functions for LangRAG Web Console
 */

// Toast Notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Format search type for display
function formatSearchType(searchType) {
    const parts = searchType.split('+');
    const labels = {
        'hybrid': '混合检索',
        'vector': '向量检索',
        'keyword': '关键词',
        'rerank': '重排序',
        'cached': '缓存'
    };
    return parts.map(p => labels[p] || p).join(' + ');
}

// Make formatSearchType available globally for Alpine.js
window.formatSearchType = formatSearchType;
