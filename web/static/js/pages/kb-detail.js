/**
 * KB Detail Page Component
 * 支持知识库级别的检索配置编辑
 */

function kbDetailPage() {
    return {
        filesToUpload: [],
        uploading: false,
        searchQuery: '',
        showEditModal: false,
        saving: false,
        editForm: {
            name: '',
            description: '',
            search_mode: 'hybrid',
            top_k: 5,
            score_threshold: 0.0,
            reranker: {
                enabled: false,
                reranker_type: '',
                model: '',
                api_key: '',
                top_k: null
            },
            rewriter: {
                enabled: false,
                llm_name: ''
            }
        },

        get kb() {
            return Alpine.store('kbs').current;
        },

        get documents() {
            return Alpine.store('kbs').documents;
        },

        get searchResults() {
            return Alpine.store('kbs').searchResults;
        },

        get searchLoading() {
            return Alpine.store('kbs').searchLoading;
        },

        get searchMode() {
            return Alpine.store('kbs').searchMode;
        },

        set searchMode(val) {
            Alpine.store('kbs').searchMode = val;
        },

        get useRerank() {
            return Alpine.store('kbs').useRerank;
        },

        set useRerank(val) {
            Alpine.store('kbs').useRerank = val;
        },

        get useRewrite() {
            return Alpine.store('kbs').useRewrite;
        },

        set useRewrite(val) {
            Alpine.store('kbs').useRewrite = val;
        },

        get rewrittenQuery() {
            return Alpine.store('kbs').rewrittenQuery;
        },

        get llms() {
            return Alpine.store('models').llms;
        },

        init() {
            // 加载 LLM 列表供 Rewriter 选择
            Alpine.store('models').load();
            
            // 监听 kb 变化，初始化编辑表单
            this.$watch('kb', (newKb) => {
                if (newKb) {
                    this.initEditForm(newKb);
                }
            });
            
            // 如果 kb 已存在，立即初始化
            if (this.kb) {
                this.initEditForm(this.kb);
            }
        },

        initEditForm(kb) {
            this.editForm = {
                name: kb.name || '',
                description: kb.description || '',
                search_mode: kb.search_mode || 'hybrid',
                top_k: kb.top_k || 5,
                score_threshold: kb.score_threshold || 0.0,
                reranker: {
                    enabled: kb.reranker?.enabled || false,
                    reranker_type: kb.reranker?.reranker_type || '',
                    model: kb.reranker?.model || '',
                    api_key: '',  // 不回显 API Key
                    top_k: kb.reranker?.top_k || null
                },
                rewriter: {
                    enabled: kb.rewriter?.enabled || false,
                    llm_name: kb.rewriter?.llm_name || ''
                }
            };
        },

        goBack() {
            Alpine.store('kbs').load();
            Alpine.store('nav').goto('kb-list');
        },

        handleFiles(event) {
            const files = event.target.files || event.dataTransfer?.files;
            if (files) {
                this.filesToUpload = [...this.filesToUpload, ...Array.from(files)];
            }
        },

        removeFile(index) {
            this.filesToUpload.splice(index, 1);
        },

        async upload() {
            if (this.filesToUpload.length === 0) {
                showToast('请选择要上传的文件', 'error');
                return;
            }

            this.uploading = true;
            const formData = new FormData();
            formData.append('kb_id', this.kb.id);
            this.filesToUpload.forEach(f => formData.append('files', f));

            try {
                await api.uploadDocuments(formData);
                showToast('文档处理完成', 'success');
                this.filesToUpload = [];
                Alpine.store('kbs').documents = await api.listDocuments(this.kb.id);
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                this.uploading = false;
            }
        },

        async search() {
            await Alpine.store('kbs').search(this.searchQuery);
        },

        async deleteKB() {
            await Alpine.store('kbs').deleteKB(this.kb.id);
        },

        async saveConfig() {
            this.saving = true;
            try {
                // 构建更新请求
                const updateData = {
                    name: this.editForm.name,
                    description: this.editForm.description,
                    search_mode: this.editForm.search_mode,
                    top_k: this.editForm.top_k,
                    score_threshold: this.editForm.score_threshold
                };

                // 添加 Reranker 配置
                if (this.editForm.reranker.enabled) {
                    updateData.reranker = {
                        enabled: true,
                        reranker_type: this.editForm.reranker.reranker_type || null,
                        model: this.editForm.reranker.model || null,
                        api_key: this.editForm.reranker.api_key || null,
                        top_k: this.editForm.reranker.top_k || null
                    };
                } else {
                    updateData.reranker = { enabled: false };
                }

                // 添加 Rewriter 配置
                if (this.editForm.rewriter.enabled) {
                    updateData.rewriter = {
                        enabled: true,
                        llm_name: this.editForm.rewriter.llm_name || null
                    };
                } else {
                    updateData.rewriter = { enabled: false };
                }

                // 调用 API 更新
                const updated = await api.updateKB(this.kb.id, updateData);
                
                // 更新本地状态
                Alpine.store('kbs').current = updated;
                
                showToast('配置已保存', 'success');
                this.showEditModal = false;
            } catch (e) {
                showToast(e.message || '保存失败', 'error');
            } finally {
                this.saving = false;
            }
        },

        formatSearchType(type) {
            const typeMap = {
                'hybrid': '混合',
                'vector': '向量',
                'keyword': '关键词',
                'hybrid+rerank': '混合+重排',
                'vector+rerank': '向量+重排',
                'semantic_search': '语义'
            };
            return typeMap[type] || type;
        }
    };
}

window.kbDetailPage = kbDetailPage;
