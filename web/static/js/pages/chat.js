/**
 * Chat Page Component
 */

function chatPage() {
    return {
        input: '',
        selectedLLM: null,

        // 检索配置
        useRerank: false,
        selectedReranker: '',
        selectedRerankerLLM: '',
        useRouter: false,
        selectedRouter: '',
        useRewriter: false,
        selectedRewriter: '',

        init() {
            console.log('Chat page initializing...');
            Alpine.store('kbs').load();
            Alpine.store('models').load().then(() => {
                console.log('Models loaded:', Alpine.store('models').llms);
                console.log('Available LLMs:', this.availableLLMs);
                // Auto-select first available LLM if none selected
                if (!this.selectedLLM && this.availableLLMs.length > 0) {
                    this.selectedLLM = this.availableLLMs[0].name;
                }
                // Auto-select first available LLM for reranker if none selected
                if (!this.selectedRerankerLLM && this.availableLLMs.length > 0) {
                    this.selectedRerankerLLM = this.availableLLMs[0].name;
                }
            });
        },

        get availableLLMs() {
            return Alpine.store('models').list || [];
        },

        get kbs() {
            return Alpine.store('kbs').list;
        },

        get messages() {
            return Alpine.store('chat').messages;
        },

        get loading() {
            return Alpine.store('chat').loading;
        },


        isSelected(kbId) {
            return Alpine.store('chat').isKBSelected(kbId);
        },

        toggleKB(kbId) {
            Alpine.store('chat').toggleKB(kbId);
        },

        toggleAll() {
            const allIds = this.kbs.map(k => k.id);
            const store = Alpine.store('chat');
            if (store.selectedKBs.length === allIds.length) {
                store.deselectAllKBs();
            } else {
                store.selectAllKBs(allIds);
            }
        },

        get isAllSelected() {
            return this.kbs.length > 0 && Alpine.store('chat').selectedKBs.length === this.kbs.length;
        },

        get isNoneSelected() {
            return Alpine.store('chat').selectedKBs.length === 0;
        },

        async send() {
            if (!this.input.trim()) return;
            const msg = this.input;
            this.input = '';

            // 构建检索配置参数
            const retrievalConfig = {
                use_rerank: this.useRerank,
                reranker_type: this.selectedReranker,
                reranker_model: this.selectedRerankerLLM,
                use_router: this.useRouter,
                router_model: this.selectedRouter,
                use_rewriter: this.useRewriter,
                rewriter_model: this.selectedRewriter
            };

            await Alpine.store('chat').send(msg, this.selectedLLM, retrievalConfig);
            // Scroll to bottom
            this.$nextTick(() => {
                const container = document.getElementById('chat-messages');
                if (container) container.scrollTop = container.scrollHeight;
            });
        },

        async evaluate(index) {
            const result = await Alpine.store('chat').evaluate(index);
            if (result) {
                this.messages[index].evaluation = result;
            }
        }
    };
}

window.chatPage = chatPage;
