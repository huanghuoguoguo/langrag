
import * as api from '../api.js';
import * as ui from '../ui.js';
import { setLoading, showToast } from '../utils.js';

export function init() {
    refreshEmbedders();
    refreshLLMs();
    setupEmbedderForm();
    setupLLMForm();
}

export async function refreshEmbedders() {
    try {
        const data = await api.fetchEmbedders();
        ui.renderEmbedders(data.embedders || [], document.getElementById('embedder-list'), document.getElementById('modal-kb-embedder'));

        // Re-attach listeners is handled by ui helpers usually, but here main logic did it inline.
        // ui.renderEmbedders renders items. We need to attach click listeners here.
        document.getElementById('embedder-list').querySelectorAll('.file-item').forEach(el => {
            el.addEventListener('click', () => activateEmbedder(el.dataset.name));
        });
    } catch (e) {
        console.error('Load embedders error:', e);
    }
}

async function activateEmbedder(name) {
    try {
        await api.activateEmbedderConfig(name);
        showToast('已激活 Embedder: ' + name);
        refreshEmbedders();
    } catch (e) {
        showToast('激活失败', 'error');
    }
}

export async function refreshLLMs() {
    try {
        const data = await api.fetchLLMs();
        ui.renderLLMs(data.llms || [], document.getElementById('llm-list'));

        const activeLLM = data.llms.find(l => l.is_active);
        const display = document.getElementById('active-llm-display');
        if (activeLLM) {
            display.textContent = `${activeLLM.name} (${activeLLM.model})`;
            display.style.color = 'var(--success)';
        } else {
            display.textContent = '未激活';
            display.style.color = 'var(--text-muted)';
        }
    } catch (e) {
        console.error('Load LLMs error', e);
    }
}

function setupEmbedderForm() {
    const embTypeSelect = document.getElementById('emb-type');
    const baseUrlGroup = document.getElementById('emb-base-url-group');
    const apiKeyGroup = document.getElementById('emb-api-key-group');
    const modelGroup = document.getElementById('emb-model-group');
    const seekdbInfo = document.getElementById('seekdb-info');

    if (embTypeSelect) {
        embTypeSelect.addEventListener('change', () => {
            const type = embTypeSelect.value;
            if (type === 'seekdb') {
                if (baseUrlGroup) baseUrlGroup.style.display = 'none';
                if (apiKeyGroup) apiKeyGroup.style.display = 'none';
                if (modelGroup) modelGroup.style.display = 'none';
                if (seekdbInfo) seekdbInfo.style.display = 'block';
            } else {
                if (baseUrlGroup) baseUrlGroup.style.display = 'block';
                if (apiKeyGroup) apiKeyGroup.style.display = 'block';
                if (modelGroup) modelGroup.style.display = 'block';
                if (seekdbInfo) seekdbInfo.style.display = 'none';
            }
        });
    }

    const btnSaveConfig = document.getElementById('btn-save-config');
    if (btnSaveConfig) {
        btnSaveConfig.addEventListener('click', async () => {
            const btn = btnSaveConfig;
            const name = document.getElementById('emb-name')?.value;
            const embType = document.getElementById('emb-type')?.value;
            const baseUrl = document.getElementById('emb-base-url')?.value;
            const apiKey = document.getElementById('emb-api-key')?.value;
            let model = document.getElementById('emb-model')?.value;

            if (!name) return showToast('请填写配置名称', 'error');
            if (embType === 'openai') {
                if (!baseUrl || !apiKey || !model) {
                    return showToast('OpenAI 类型需要 Base URL、API Key 和 Model Name', 'error');
                }
            } else if (embType === 'seekdb') {
                model = 'all-MiniLM-L6-v2';
            }

            setLoading(btn, true);
            try {
                await api.saveEmbedderConfig({
                    name,
                    embedder_type: embType,
                    model,
                    base_url: baseUrl,
                    api_key: apiKey
                });
                showToast('配置已保存！');
                refreshEmbedders();

                // Clear fields
                if (document.getElementById('emb-name')) document.getElementById('emb-name').value = '';
                if (document.getElementById('emb-base-url')) document.getElementById('emb-base-url').value = '';
                if (document.getElementById('emb-api-key')) document.getElementById('emb-api-key').value = '';
                if (document.getElementById('emb-model')) document.getElementById('emb-model').value = '';
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                setLoading(btn, false);
            }
        });
    }
}

function setupLLMForm() {
    const llmProviderSelect = document.getElementById('llm-provider');
    const llmBaseUrlInput = document.getElementById('llm-base-url');
    const llmModelInput = document.getElementById('llm-model');

    const PROVIDERS = {
        'kimi': { baseUrl: 'https://api.moonshot.cn/v1', model: 'kimi-k2-turbo-preview' },
        'custom': { baseUrl: '', model: '' }
    };

    if (llmProviderSelect && llmBaseUrlInput && llmModelInput) {
        llmProviderSelect.addEventListener('change', () => {
            const provider = llmProviderSelect.value;
            const config = PROVIDERS[provider];

            if (provider !== 'custom') {
                llmBaseUrlInput.value = config.baseUrl;
                llmModelInput.value = config.model;
                llmBaseUrlInput.readOnly = true;
                llmModelInput.readOnly = true;
                llmBaseUrlInput.style.backgroundColor = 'var(--bg-secondary)';
                llmModelInput.style.backgroundColor = 'var(--bg-secondary)';
            } else {
                llmBaseUrlInput.value = '';
                llmModelInput.value = '';
                llmBaseUrlInput.readOnly = false;
                llmModelInput.readOnly = false;
                llmBaseUrlInput.style.backgroundColor = '';
                llmModelInput.style.backgroundColor = '';
            }
        });
        llmProviderSelect.dispatchEvent(new Event('change'));
    }

    const btnSaveLLM = document.getElementById('btn-save-llm');
    if (btnSaveLLM) {
        btnSaveLLM.addEventListener('click', async () => {
            const btn = btnSaveLLM;
            const name = document.getElementById('llm-name')?.value;
            const baseUrl = document.getElementById('llm-base-url')?.value;
            const apiKey = document.getElementById('llm-api-key')?.value;
            const model = document.getElementById('llm-model')?.value;
            const temp = parseFloat(document.getElementById('llm-temp')?.value || 0.7);
            const tokens = parseInt(document.getElementById('llm-tokens')?.value || 2048);

            if (!name || !baseUrl || !apiKey || !model) {
                return showToast('请填写完整的 LLM 配置信息', 'error');
            }

            setLoading(btn, true);
            try {
                await api.saveLLMConfig({
                    name,
                    base_url: baseUrl,
                    api_key: apiKey,
                    model: model,
                    temperature: temp,
                    max_tokens: tokens
                });
                showToast('LLM 配置已保存并激活！');
                refreshLLMs();

                if (document.getElementById('llm-name')) document.getElementById('llm-name').value = '';
                if (document.getElementById('llm-api-key')) document.getElementById('llm-api-key').value = '';
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                setLoading(btn, false);
            }
        });
    }
}
