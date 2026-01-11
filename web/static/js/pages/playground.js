/**
 * Playground Page Component
 * Feature visualization and comparison tools
 */

function playgroundPage() {
    return {
        // State
        selectedKB: '',
        query: '',

        // Search Compare
        searchCompareLoading: false,
        searchCompareResults: null,

        // Query Rewrite
        rewriteQuery: '',
        rewriteLoading: false,
        rewriteResult: null,

        // Rerank Compare
        rerankLoading: false,
        rerankResults: null,

        // Cache
        cacheStats: null,
        cacheLoading: false,
        cacheTestQuery: '',
        cacheTestResult: null,

        init() {
            Alpine.store('kbs').load();
            this.loadCacheStats();
        },

        get kbs() {
            return Alpine.store('kbs').list;
        },

        // ==================== Search Mode Comparison ====================
        async compareSearchModes() {
            if (!this.selectedKB || !this.query.trim()) {
                showToast('请选择知识库并输入查询', 'error');
                return;
            }
            this.searchCompareLoading = true;
            this.searchCompareResults = null;
            try {
                this.searchCompareResults = await api.compareSearchModes({
                    kb_id: this.selectedKB,
                    query: this.query,
                    top_k: 5
                });
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                this.searchCompareLoading = false;
            }
        },

        getModeLabel(mode) {
            const labels = {
                'hybrid': '混合检索',
                'vector': '向量检索',
                'keyword': '关键词检索'
            };
            return labels[mode] || mode;
        },

        getModeIcon(mode) {
            const icons = { 'hybrid': '🔀', 'vector': '🧠', 'keyword': '🔤' };
            return icons[mode] || '📊';
        },

        // ==================== Query Rewrite ====================
        async testRewrite() {
            if (!this.rewriteQuery.trim()) {
                showToast('请输入查询', 'error');
                return;
            }
            this.rewriteLoading = true;
            this.rewriteResult = null;
            try {
                this.rewriteResult = await api.testQueryRewrite({ query: this.rewriteQuery });
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                this.rewriteLoading = false;
            }
        },

        // ==================== Rerank Comparison ====================
        async compareReranking() {
            if (!this.selectedKB || !this.query.trim()) {
                showToast('请选择知识库并输入查询', 'error');
                return;
            }
            this.rerankLoading = true;
            this.rerankResults = null;
            try {
                this.rerankResults = await api.compareReranking({
                    kb_id: this.selectedKB,
                    query: this.query,
                    top_k: 5
                });
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                this.rerankLoading = false;
            }
        },

        getRankChangeClass(change) {
            if (change > 0) return 'rank-up';
            if (change < 0) return 'rank-down';
            return 'rank-same';
        },

        getRankChangeIcon(change) {
            if (change > 0) return '↑';
            if (change < 0) return '↓';
            return '→';
        },

        // ==================== Cache Analysis ====================
        async loadCacheStats() {
            try {
                this.cacheStats = await api.getCacheStats();
            } catch (e) {
                console.error('Failed to load cache stats:', e);
            }
        },

        async testCacheHit() {
            if (!this.selectedKB || !this.cacheTestQuery.trim()) {
                showToast('请选择知识库并输入查询', 'error');
                return;
            }
            this.cacheLoading = true;
            this.cacheTestResult = null;
            try {
                this.cacheTestResult = await api.testCache({
                    kb_id: this.selectedKB,
                    query: this.cacheTestQuery
                });
                await this.loadCacheStats();
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                this.cacheLoading = false;
            }
        },

        async clearCache() {
            try {
                await api.clearCache();
                showToast('缓存已清空', 'success');
                await this.loadCacheStats();
                this.cacheTestResult = null;
            } catch (e) {
                showToast(e.message, 'error');
            }
        },

        formatHitRate(rate) {
            return (rate * 100).toFixed(1) + '%';
        }
    };
}

window.playgroundPage = playgroundPage;
