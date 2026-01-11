/**
 * Chat Page Component
 */

function chatPage() {
    return {
        input: '',

        init() {
            Alpine.store('kbs').load();
            Alpine.store('models').load();
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

        async send() {
            if (!this.input.trim()) return;
            const msg = this.input;
            this.input = '';
            await Alpine.store('chat').send(msg);
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
