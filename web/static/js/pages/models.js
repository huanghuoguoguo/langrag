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
                this.llmForm.model_path = '';
            } else if (this.llmForm.provider === 'local') {
                this.llmForm.base_url = '';
                this.llmForm.api_key = '';
                this.llmForm.model = 'qwen2.5-7b-instruct';
                this.llmForm.model_path = '';  // 留空，后端会使用默认路径
            }  else {
                this.llmForm.base_url = '';
                this.llmForm.model = '';
                this.llmForm.model_path = '';
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


        async saveLLM() {
            if (!this.llmForm.name) {
                showToast('请输入配置名称', 'error');
                return;
            }
            this.llmSaving = true;
            const payload = { ...this.llmForm };

            // Defaults for local model
            if (payload.provider === 'local') {
                // 总是使用默认路径和配置（优先使用小模型）
                payload.model_path = "/home/yhh/models/qwen2-0_5b-instruct-q4_k_m.gguf";
                payload.model = "qwen2-0_5b-instruct";
                if (!payload.name) {
                    payload.name = "qwen-local";
                }
            }

            const success = await Alpine.store('models').saveLLM(payload);
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

    };
}

window.modelsPage = modelsPage;
