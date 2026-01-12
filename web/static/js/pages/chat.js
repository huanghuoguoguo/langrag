/**
 * Chat Page Component
 */

function chatPage() {
    return {
        input: '',
        selectedLLM: null,

        init() {
            console.log('Chat page initializing...');
            Alpine.store('kbs').load();
            Alpine.store('models').load().then(() => {
                console.log('Models loaded:', Alpine.store('models').llms);
                console.log('Active LLM:', Alpine.store('models').activeLLM);
                console.log('Available LLMs:', this.availableLLMs);
            });
            this.$watch('activeLLM', (val) => {
                console.log('Active LLM changed:', val);
                if (val && !this.selectedLLM) this.selectedLLM = val.name;
            });
            // Init selected if available
            if (this.activeLLM && !this.selectedLLM) this.selectedLLM = this.activeLLM.name;
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

        get activeLLM() {
            return Alpine.store('models').activeLLM;
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
            await Alpine.store('chat').send(msg, this.selectedLLM);
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
