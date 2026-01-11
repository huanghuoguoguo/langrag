/**
 * API Client for LangRAG Web Console
 */

window.api = {
    async get(url) {
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed: ${res.status}`);
        }
        return res.json();
    },

    async post(url, data, isFormData = false) {
        const options = {
            method: 'POST',
            body: isFormData ? data : JSON.stringify(data),
        };
        if (!isFormData) {
            options.headers = { 'Content-Type': 'application/json' };
        }
        const res = await fetch(url, options);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed: ${res.status}`);
        }
        return res.json();
    },

    async delete(url) {
        const res = await fetch(url, { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed: ${res.status}`);
        }
        return res.json();
    },

    // Knowledge Base
    listKBs: () => window.api.get('/api/kb'),
    getKB: (id) => window.api.get(`/api/kb/${id}`),
    createKB: (data) => window.api.post('/api/kb', data),
    deleteKB: (id) => window.api.delete(`/api/kb/${id}`),

    // Documents
    listDocuments: (kbId) => window.api.get(`/api/upload/documents/${kbId}`),
    uploadDocuments: (formData) => window.api.post('/api/upload', formData, true),

    // Search
    search: (data) => window.api.post('/api/search', data),

    // Chat
    chat: (data) => window.api.post('/api/chat', data),
    evaluate: (data) => window.api.post('/api/chat/evaluate', data),

    // Config
    listEmbedders: () => window.api.get('/api/config/embedders'),
    saveEmbedder: (data) => window.api.post('/api/config/embedder', data),
    activateEmbedder: (name) => window.api.post('/api/config/embedder/activate', { name }),
    listLLMs: () => window.api.get('/api/config/llms'),
    saveLLM: (data) => window.api.post('/api/config/llm', data),
    activateLLM: (name) => window.api.post('/api/config/llm/activate', { name }),

    // Playground
    compareSearchModes: (data) => window.api.post('/api/playground/search-compare', data),
    testQueryRewrite: (data) => window.api.post('/api/playground/query-rewrite', data),
    compareReranking: (data) => window.api.post('/api/playground/rerank-compare', data),
    getCacheStats: () => window.api.get('/api/playground/cache-stats'),
    testCache: (data) => window.api.post('/api/playground/cache-test', data),
    clearCache: () => window.api.post('/api/playground/cache-clear', {}),
};
