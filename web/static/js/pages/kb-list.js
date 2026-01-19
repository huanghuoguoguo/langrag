/**
 * KB List Page Component
 *
 * 创建知识库时配置离线/索引相关的选项：
 * - embedder: 决定向量维度，创建后不可变
 * - chunk_size/overlap: 文档切分方式
 * - vdb_type: 向量库类型
 * - indexing_technique: 索引策略 (paragraph/qa/raptor)
 * - indexing_llm_name: QA/RAPTOR 索引使用的 LLM
 *
 * 检索配置 (reranker/rewriter/router) 是在线动态的，在聊天时选择
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
            indexing_technique: 'paragraph',
            indexing_llm_name: ''
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

        get needsLLM() {
            return this.form.indexing_technique === 'qa' || this.form.indexing_technique === 'raptor';
        },

        openModal() {
            this.form = {
                name: '',
                description: '',
                vdb_type: 'seekdb',
                embedder_name: '',
                chunk_size: 1000,
                chunk_overlap: 100,
                indexing_technique: 'paragraph',
                indexing_llm_name: ''
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
            if (this.needsLLM && !this.form.indexing_llm_name) {
                showToast(`${this.form.indexing_technique.toUpperCase()} 索引需要选择 LLM`, 'error');
                return;
            }

            // 构建请求数据 - 只包含离线配置
            const requestData = {
                name: this.form.name,
                description: this.form.description,
                vdb_type: this.form.vdb_type,
                embedder_name: this.form.embedder_name,
                chunk_size: this.form.chunk_size,
                chunk_overlap: this.form.chunk_overlap,
                indexing_technique: this.form.indexing_technique,
                indexing_llm_name: this.needsLLM ? this.form.indexing_llm_name : null
            };

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
        },

        // Helper: Get indexing technique display name
        getIndexingName(technique) {
            const names = {
                'paragraph': '段落索引',
                'qa': 'QA 索引',
                'raptor': 'RAPTOR 索引'
            };
            return names[technique] || technique;
        }
    };
}

window.kbListPage = kbListPage;
