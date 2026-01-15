/**
 * Alpine.js Stores for LangRAG Web Console
 */

document.addEventListener('alpine:init', () => {

    // Navigation Store
    Alpine.store('nav', {
        currentView: 'kb-list',
        currentKB: null,

        goto(view, kb = null) {
            this.currentView = view;
            if (kb) this.currentKB = kb;
        },

        isActive(view) {
            return this.currentView === view;
        }
    });

    // Knowledge Base Store
    Alpine.store('kbs', {
        list: [],
        loading: false,
        current: null,
        documents: [],
        searchResults: [],
        searchLoading: false,
        searchMode: 'auto',
        useRerank: null,
        useRewrite: false,
        rewrittenQuery: null,

        async load() {
            this.loading = true;
            try {
                this.list = await api.listKBs();
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                this.loading = false;
            }
        },

        async loadDetail(id) {
            try {
                this.current = await api.getKB(id);
                this.documents = await api.listDocuments(id);
                this.searchResults = [];
            } catch (e) {
                showToast(e.message, 'error');
            }
        },

        async create(data) {
            try {
                await api.createKB(data);
                showToast('知识库创建成功', 'success');
                await this.load();
                return true;
            } catch (e) {
                showToast(e.message, 'error');
                return false;
            }
        },

        async deleteKB(id) {
            // User requested immediate deletion without confirmation
            // if (!confirm('确定要删除这个知识库吗？')) return;
            try {
                await api.deleteKB(id);
                showToast('知识库已删除', 'success');
                await this.load();
                Alpine.store('nav').goto('kb-list');
            } catch (e) {
                showToast(e.message, 'error');
            }
        },

        async search(query) {
            if (!query.trim() || !this.current) return;
            this.searchLoading = true;
            this.rewrittenQuery = null;
            try {
                const result = await api.search({
                    kb_id: this.current.id,
                    query: query,
                    top_k: 5,
                    search_mode: this.searchMode === 'auto' ? null : this.searchMode,
                    use_rerank: this.useRerank,
                    use_rewrite: this.useRewrite
                });
                this.searchResults = result.results || [];
                this.rewrittenQuery = result.rewritten_query;
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                this.searchLoading = false;
            }
        }
    });

    // Models Store
    Alpine.store('models', {
        embedders: [],
        llms: [],
        activeEmbedder: null,  // Keep for backward compatibility, but not enforced
        activeLLM: null,       // Keep for backward compatibility, but not enforced

        get list() {
            return this.llms || [];
        },

        async load() {
            try {
                const embeddersRes = await api.listEmbedders();
                this.embedders = embeddersRes.embedders || [];

                const llmsRes = await api.listLLMs();
                this.llms = llmsRes.llms || [];

                // Still set active for backward compatibility, but don't enforce single active
                this.activeEmbedder = this.embedders.find(e => e.is_active) || this.embedders[0] || null;
                this.activeLLM = this.llms.find(l => l.is_active) || this.llms[0] || null;
            } catch (e) {
                showToast(e.message, 'error');
            }
        },

        async saveEmbedder(data) {
            try {
                await api.saveEmbedder(data);
                showToast('Embedder 保存成功', 'success');
                await this.load();
                return true;
            } catch (e) {
                showToast(e.message, 'error');
                return false;
            }
        },


        async saveLLM(data) {
            try {
                await api.saveLLM(data);
                showToast('LLM 保存成功', 'success');
                await this.load();
                return true;
            } catch (e) {
                showToast(e.message, 'error');
                return false;
            }
        },

        async deleteEmbedder(name) {
            try {
                await api.deleteEmbedder(name);
                showToast('Embedder 删除成功', 'success');
                await this.load();
                return true;
            } catch (e) {
                showToast(e.message, 'error');
                return false;
            }
        },

        async deleteLLM(name) {
            try {
                await api.deleteLLM(name);
                showToast('LLM 删除成功', 'success');
                await this.load();
                return true;
            } catch (e) {
                showToast(e.message, 'error');
                return false;
            }
        },

    });

    // Chat Store
    Alpine.store('chat', {
        messages: [
            { role: 'assistant', content: '你好！', sources: [] }
        ],
        selectedKBs: [],
        loading: false,

        toggleKB(kbId) {
            const idx = this.selectedKBs.indexOf(kbId);
            if (idx === -1) {
                this.selectedKBs.push(kbId);
            } else {
                this.selectedKBs.splice(idx, 1);
            }
        },

        isKBSelected(kbId) {
            return this.selectedKBs.includes(kbId);
        },

        selectAllKBs(allIds) {
            this.selectedKBs = [...allIds];
        },

        deselectAllKBs() {
            this.selectedKBs = [];
        },

        async send(input, modelName = null, retrievalConfig = {}) {
            if (!input.trim()) return;

            const userMessage = { role: 'user', content: input, sources: [] };
            this.messages.push(userMessage);
            this.loading = true;

            try {
                const result = await api.chat({
                    kb_ids: this.selectedKBs,
                    query: input,
                    history: this.messages.slice(0, -1).map(m => ({
                        role: m.role,
                        content: m.content
                    })),
                    stream: false,
                    model_name: modelName,
                    // 检索配置参数
                    use_rerank: retrievalConfig.use_rerank || false,
                    reranker_type: retrievalConfig.reranker_type || null,
                    reranker_model: retrievalConfig.reranker_model || null,
                    use_router: retrievalConfig.use_router || false,
                    router_model: retrievalConfig.router_model || null,
                    use_rewriter: retrievalConfig.use_rewriter || false,
                    rewriter_model: retrievalConfig.rewriter_model || null
                });

                // 优先显示 LLM 生成的答案，否则显示检索信息
                let content = result.answer || result.message || `检索完成，找到 ${result.sources?.length || 0} 个相关文档`;


                this.messages.push({
                    role: 'assistant',
                    content: content,
                    sources: result.sources || [],
                    question: input,
                    retrieval_stats: result.retrieval_stats,
                    rewritten_query: result.rewritten_query
                });
            } catch (e) {
                showToast(e.message, 'error');
                this.messages.push({
                    role: 'assistant',
                    content: `错误: ${e.message}`,
                    sources: []
                });
            } finally {
                this.loading = false;
            }
        },

        async evaluate(msgIndex) {
            const msg = this.messages[msgIndex];
            if (!msg || msg.role !== 'assistant' || !msg.sources?.length) return null;

            // Set evaluating state
            msg.evaluating = true;

            try {
                const result = await api.evaluate({
                    question: msg.question || '',
                    answer: msg.content,
                    contexts: msg.sources.map(s => s.content)
                });
                return result;
            } catch (e) {
                showToast(e.message, 'error');
                return null;
            } finally {
                msg.evaluating = false;
            }
        }
    });
});
