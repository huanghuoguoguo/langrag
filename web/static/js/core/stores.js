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
            if (!confirm('确定要删除这个知识库吗？')) return;
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
            try {
                const result = await api.search({
                    kb_id: this.current.id,
                    query: query,
                    top_k: 5,
                    search_mode: this.searchMode === 'auto' ? null : this.searchMode,
                    use_rerank: this.useRerank
                });
                this.searchResults = result.results || [];
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
        activeEmbedder: null,
        activeLLM: null,

        async load() {
            try {
                const embeddersRes = await api.listEmbedders();
                this.embedders = embeddersRes.embedders || [];

                const llmsRes = await api.listLLMs();
                this.llms = llmsRes.llms || [];

                this.activeEmbedder = this.embedders.find(e => e.is_active) || null;
                this.activeLLM = this.llms.find(l => l.is_active) || null;
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

        async activateEmbedder(name) {
            try {
                await api.activateEmbedder(name);
                showToast('Embedder 已激活', 'success');
                await this.load();
            } catch (e) {
                showToast(e.message, 'error');
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

        async activateLLM(name) {
            try {
                await api.activateLLM(name);
                showToast('LLM 已激活', 'success');
                await this.load();
            } catch (e) {
                showToast(e.message, 'error');
            }
        }
    });

    // Chat Store
    Alpine.store('chat', {
        messages: [
            { role: 'assistant', content: '你好！请先在左侧选择一个或多个知识库，然后开始提问。', sources: [] }
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

        async send(input) {
            if (!input.trim() || this.selectedKBs.length === 0) {
                showToast('请选择至少一个知识库', 'error');
                return;
            }

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
                    stream: false
                });

                this.messages.push({
                    role: 'assistant',
                    content: result.answer,
                    sources: result.sources || [],
                    question: input
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
            }
        }
    });
});
