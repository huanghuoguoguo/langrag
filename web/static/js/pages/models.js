/**
 * Models Config Page Component
 */

function modelsPage() {
    return {
        // Embedder form
        embForm: {
            name: '',
            embedder_type: 'openai',
            base_url: 'https://api.openai.com/v1',
            api_key: '',
            model: 'text-embedding-3-small'
        },
        embSaving: false,

        // LLM form
        llmForm: {
            name: '',
            provider: 'kimi',
            base_url: 'https://api.moonshot.cn/v1',
            api_key: '',
            model: 'moonshot-v1-8k',
            temperature: 0.7,
            max_tokens: 2048
        },
        llmSaving: false,

        init() {
            Alpine.store('models').load();
        },

        get embedders() {
            return Alpine.store('models').embedders;
        },

        get llms() {
            return Alpine.store('models').llms;
        },

        get isSeekDB() {
            return this.embForm.embedder_type === 'seekdb';
        },

        onEmbTypeChange() {
            if (this.isSeekDB) {
                this.embForm.base_url = '';
                this.embForm.api_key = '';
                this.embForm.model = 'all-MiniLM-L6-v2';
            } else {
                this.embForm.base_url = 'https://api.openai.com/v1';
                this.embForm.model = 'text-embedding-3-small';
            }
        },

        onLLMProviderChange() {
            if (this.llmForm.provider === 'kimi') {
                this.llmForm.base_url = 'https://api.moonshot.cn/v1';
                this.llmForm.model = 'moonshot-v1-8k';
            } else {
                this.llmForm.base_url = '';
                this.llmForm.model = '';
            }
        },

        async saveEmbedder() {
            if (!this.embForm.name) {
                showToast('请输入配置名称', 'error');
                return;
            }
            this.embSaving = true;
            const success = await Alpine.store('models').saveEmbedder(this.embForm);
            this.embSaving = false;
            if (success) {
                this.embForm = {
                    name: '',
                    embedder_type: 'openai',
                    base_url: 'https://api.openai.com/v1',
                    api_key: '',
                    model: 'text-embedding-3-small'
                };
            }
        },

        async activateEmbedder(name) {
            await Alpine.store('models').activateEmbedder(name);
        },

        async saveLLM() {
            if (!this.llmForm.name) {
                showToast('请输入配置名称', 'error');
                return;
            }
            this.llmSaving = true;
            const success = await Alpine.store('models').saveLLM(this.llmForm);
            this.llmSaving = false;
            if (success) {
                this.llmForm = {
                    name: '',
                    provider: 'kimi',
                    base_url: 'https://api.moonshot.cn/v1',
                    api_key: '',
                    model: 'moonshot-v1-8k',
                    temperature: 0.7,
                    max_tokens: 2048
                };
            }
        },

        async activateLLM(name) {
            await Alpine.store('models').activateLLM(name);
        }
    };
}

window.modelsPage = modelsPage;
