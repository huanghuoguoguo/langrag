/**
 * KB List Page Component
 * 支持知识库级别的检索配置（Reranker、Rewriter 等）
 */

function kbListPage() {
    return {
        showModal: false,
        form: {
            name: '',
            description: '',
            vdb_type: 'seekdb',
            embedder_name: '',
            chunk_size: 1000,
            chunk_overlap: 100,
            // 检索配置
            search_mode: 'hybrid',
            top_k: 5,
            score_threshold: 0.0,
            // Reranker 配置
            reranker: {
                enabled: false,
                reranker_type: '',
                model: '',
                api_key: '',
                top_k: null
            },
            // Rewriter 配置
            rewriter: {
                enabled: false,
                llm_name: ''
            }
        },
        creating: false,

        init() {
            Alpine.store('kbs').load();
            Alpine.store('models').load();
        },

        get kbs() {
            return Alpine.store('kbs').list;
        },

        get loading() {
            return Alpine.store('kbs').loading;
        },

        get embedders() {
            return Alpine.store('models').embedders;
        },

        get llms() {
            return Alpine.store('models').llms;
        },

        get isWebSearch() {
            return this.form.vdb_type === 'web_search';
        },

        openModal() {
            this.form = {
                name: '',
                description: '',
                vdb_type: 'seekdb',
                embedder_name: '',
                chunk_size: 1000,
                chunk_overlap: 100,
                // 检索配置
                search_mode: 'hybrid',
                top_k: 5,
                score_threshold: 0.0,
                // Reranker 配置
                reranker: {
                    enabled: false,
                    reranker_type: '',
                    model: '',
                    api_key: '',
                    top_k: null
                },
                // Rewriter 配置
                rewriter: {
                    enabled: false,
                    llm_name: ''
                }
            };
            this.showModal = true;
        },

        async createKB() {
            if (!this.form.name) {
                showToast('请输入知识库名称', 'error');
                return;
            }
            if (!this.isWebSearch && !this.form.embedder_name) {
                showToast('请选择 Embedder', 'error');
                return;
            }

            // 构建请求数据
            const requestData = {
                name: this.form.name,
                description: this.form.description,
                vdb_type: this.form.vdb_type,
                embedder_name: this.form.embedder_name,
                chunk_size: this.form.chunk_size,
                chunk_overlap: this.form.chunk_overlap,
                search_mode: this.form.search_mode,
                top_k: this.form.top_k,
                score_threshold: this.form.score_threshold
            };

            // 添加 Reranker 配置（如果启用）
            if (this.form.reranker.enabled) {
                requestData.reranker = {
                    enabled: true,
                    reranker_type: this.form.reranker.reranker_type || null,
                    model: this.form.reranker.model || null,
                    api_key: this.form.reranker.api_key || null,
                    top_k: this.form.reranker.top_k || null
                };
            }

            // 添加 Rewriter 配置（如果启用）
            if (this.form.rewriter.enabled) {
                requestData.rewriter = {
                    enabled: true,
                    llm_name: this.form.rewriter.llm_name || null
                };
            }

            this.creating = true;
            const success = await Alpine.store('kbs').create(requestData);
            this.creating = false;
            if (success) {
                this.showModal = false;
            }
        },

        selectKB(kb) {
            Alpine.store('kbs').loadDetail(kb.id);
            Alpine.store('nav').goto('kb-detail', kb);
        }
    };
}

window.kbListPage = kbListPage;
