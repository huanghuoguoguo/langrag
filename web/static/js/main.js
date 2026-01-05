
import * as api from './api.js';
import * as ui from './ui.js';
import { setLoading, showToast } from './utils.js';

document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let currentKbId = null;
    let filesToUpload = [];

    // --- Elements ---
    const modal = document.getElementById('create-kb-modal');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const fileListForUpload = document.getElementById('file-list');
    const kbListContainer = document.getElementById('kb-list-container');
    const documentsContainer = document.getElementById('documents-list');
    const embedderListContainer = document.getElementById('embedder-list');
    const embedderSelect = document.getElementById('modal-kb-embedder');
    const llmListContainer = document.getElementById('llm-list');
    const chatMessages = document.getElementById('chat-messages');
    const chatKBList = document.getElementById('chat-kb-list');
    const activeLLMDisplay = document.getElementById('active-llm-display');

    // --- Initialization ---
    init();

    async function init() {
        await refreshKBList();
        await refreshEmbedders();
        await refreshLLMs();
    }

    // --- Actions (Data Fetch + UI Update) ---

    async function refreshKBList() {
        try {
            const data = await api.fetchKnowledgeBases();
            ui.renderKBList(data, kbListContainer);
            ui.renderKBCheckboxes(data, chatKBList);

            // Re-attach click listeners for KB cards
            document.querySelectorAll('.kb-card').forEach(card => {
                card.addEventListener('click', () => selectKB(card.dataset.id));
            });
        } catch (e) {
            showToast('加载知识库失败', 'error');
            console.error(e);
        }
    }

    async function selectKB(kbId) {
        currentKbId = kbId;
        try {
            const kb = await api.fetchKBDetail(kbId);
            ui.renderKBDetail(kb);

            await refreshDocuments(kbId);
            showKBDetailView();
        } catch (e) {
            showToast('加载知识库详情失败', 'error');
            console.error(e);
        }
    }

    async function refreshDocuments(kbId) {
        try {
            const docs = await api.fetchDocuments(kbId);
            ui.renderDocuments(docs, documentsContainer);
        } catch (e) {
            console.error('Load documents error:', e);
        }
    }

    async function refreshEmbedders() {
        try {
            const data = await api.fetchEmbedders();
            ui.renderEmbedders(data.embedders || [], embedderListContainer, embedderSelect);

            // Re-attach click listeners for embedder items to activate
            embedderListContainer.querySelectorAll('.file-item').forEach(el => {
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

    async function refreshLLMs() {
        try {
            const data = await api.fetchLLMs();
            ui.renderLLMs(data.llms || [], llmListContainer);

            // Check for active LLM
            const activeLLM = data.llms.find(l => l.is_active);
            if (activeLLM) {
                activeLLMDisplay.textContent = `${activeLLM.name} (${activeLLM.model})`;
                activeLLMDisplay.style.color = 'var(--success)';
            } else {
                activeLLMDisplay.textContent = '未激活';
                activeLLMDisplay.style.color = 'var(--text-muted)';
            }
        } catch (e) {
            console.error('Load LLMs error', e);
        }
    }

    // --- Navigation & View Switching ---

    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const target = item.dataset.target;
            if (!target) return;

            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            document.querySelectorAll('.section').forEach(s => {
                s.classList.remove('active');
                if (s.id === target) {
                    s.classList.add('active');
                }
            });
        });
    });

    document.getElementById('btn-back-to-list').addEventListener('click', showKBListView);
    document.getElementById('btn-show-create-modal').addEventListener('click', () => modal.classList.add('active'));

    function showKBDetailView() {
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById('kb-detail').classList.add('active');

        // Update Nav
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        const detailNav = document.querySelector('[data-target="kb-detail"]');
        if (detailNav) {
            detailNav.style.display = 'flex';
            detailNav.classList.add('active');
        }
    }

    function showKBListView() {
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById('kb-list').classList.add('active');

        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.querySelector('[data-target="kb-list"]').classList.add('active');

        refreshKBList();
    }

    // --- Modal Controls ---
    document.querySelectorAll('.modal-close').forEach(el => {
        el.addEventListener('click', () => modal.classList.remove('active'));
    });
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });

    // --- Form Handlers ---

    // VDB Type Change Handler - Show/Hide fields based on type
    const vdbTypeSelect = document.getElementById('modal-kb-vdb');
    const embedderGroup = document.getElementById('modal-kb-embedder-group');
    const chunkSizeGroup = document.getElementById('modal-kb-chunk-size-group');
    const chunkOverlapGroup = document.getElementById('modal-kb-chunk-overlap-group');

    if (vdbTypeSelect) {
        vdbTypeSelect.addEventListener('change', () => {
            const vdbType = vdbTypeSelect.value;
            if (vdbType === 'web_search') {
                // Hide embedder and chunk config for web search
                if (embedderGroup) embedderGroup.style.display = 'none';
                if (chunkSizeGroup) chunkSizeGroup.style.display = 'none';
                if (chunkOverlapGroup) chunkOverlapGroup.style.display = 'none';
            } else {
                // Show all fields for local VDBs
                if (embedderGroup) embedderGroup.style.display = 'block';
                if (chunkSizeGroup) chunkSizeGroup.style.display = 'block';
                if (chunkOverlapGroup) chunkOverlapGroup.style.display = 'block';
            }
        });
    }

    // Create KB
    document.getElementById('btn-create-kb-confirm').addEventListener('click', async () => {
        const name = document.getElementById('modal-kb-name').value;
        const desc = document.getElementById('modal-kb-desc').value;
        const vdbType = document.getElementById('modal-kb-vdb').value;
        const embedderName = document.getElementById('modal-kb-embedder').value;
        const chunkSize = parseInt(document.getElementById('modal-kb-chunk-size').value);
        const chunkOverlap = parseInt(document.getElementById('modal-kb-chunk-overlap').value);
        const btn = document.getElementById('btn-create-kb-confirm');

        if (!name) return showToast('请输入知识库名称', 'error');

        // Skip embedder validation for web_search
        if (vdbType !== 'web_search' && !embedderName) {
            return showToast('请选择 Embedding 模型', 'error');
        }

        setLoading(btn, true);
        try {
            const payload = {
                name,
                description: desc,
                vdb_type: vdbType
            };

            // Only include embedder and chunk config for non-web types
            if (vdbType !== 'web_search') {
                payload.embedder_name = embedderName;
                payload.chunk_size = chunkSize;
                payload.chunk_overlap = chunkOverlap;
            }

            await api.createKB(payload);
            showToast('知识库创建成功！');
            modal.classList.remove('active');

            // Clean inputs
            document.getElementById('modal-kb-name').value = '';
            document.getElementById('modal-kb-desc').value = '';

            refreshKBList();
        } catch (e) {
            showToast(e.message, 'error');
        } finally {
            setLoading(btn, false);
        }
    });

    // File Upload Drag & Drop
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    function handleFiles(files) {
        Array.from(files).forEach(file => filesToUpload.push(file));
        ui.renderFilesToUpload(files, fileListForUpload);
    }

    // Upload Action
    document.getElementById('btn-upload').addEventListener('click', async () => {
        if (!currentKbId) return showToast('请先选择知识库', 'error');
        if (filesToUpload.length === 0) return showToast('没有可上传的文件', 'error');

        const btn = document.getElementById('btn-upload');
        setLoading(btn, true);

        const formData = new FormData();
        formData.append('kb_id', currentKbId);
        filesToUpload.forEach(file => formData.append('files', file));

        try {
            const data = await api.uploadDocuments(formData);
            showToast(`成功！处理了 ${data.processed_files} 个文件，共 ${data.total_chunks} 个块。`);
            if (data.failed_files.length > 0) {
                showToast('部分文件失败: ' + data.failed_files.join(', '), 'error');
            }
            filesToUpload = [];
            fileListForUpload.innerHTML = '';
            refreshDocuments(currentKbId);
        } catch (e) {
            showToast(e.message, 'error');
        } finally {
            setLoading(btn, false);
        }
    });

    // Search
    document.getElementById('btn-search').addEventListener('click', async () => {
        if (!currentKbId) return showToast('请先选择知识库', 'error');
        const query = document.getElementById('search-query').value;
        const btn = document.getElementById('btn-search');
        const resultsDiv = document.getElementById('search-results');

        if (!query) return;

        setLoading(btn, true);
        resultsDiv.innerHTML = '';

        try {
            const data = await api.searchKB({
                kb_id: currentKbId,
                query: query,
                top_k: 5
            });
            ui.renderSearchResults(data.results, resultsDiv);
        } catch (e) {
            showToast(e.message, 'error');
        } finally {
            setLoading(btn, false);
        }
    });

    // Chat
    document.getElementById('btn-send').addEventListener('click', sendMessage);
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    async function sendMessage() {
        // Collect selected KB IDs
        // Mode: Auto Select All (Smart Routing)
        const selectedKBs = []; // Empty -> Auto Select All in Backend

        // Validations
        // if (selectedKBs.length === 0) { ... } // Allow empty -> Auto Select All

        const input = document.getElementById('chat-input');
        const query = input.value.trim();
        const btn = document.getElementById('btn-send');

        if (!query) return;

        // Add user message
        ui.appendMessage(chatMessages, 'user', query);
        input.value = '';

        setLoading(btn, true);

        try {
            const history = []; // TODO: Extract from UI if needed

            const data = await api.chat({
                kb_ids: selectedKBs,
                query: query,
                history: history
            });

            ui.appendMessage(chatMessages, 'assistant', data.answer, data.sources);

        } catch (e) {
            ui.appendMessage(chatMessages, 'assistant', `Error: ${e.message}`);
        } finally {
            setLoading(btn, false);
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    // Embedder Config Form
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

                // Clear specific fields
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

    // LLM Config Form
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
        // Trigger once
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
});
