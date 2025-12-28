
document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentKbId = null;
    let filesToUpload = [];

    // Elements
    const modal = document.getElementById('create-kb-modal');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');

    // Initialize
    loadKnowledgeBases();
    loadEmbedders();

    // Navigation
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

    // Modal Controls
    document.getElementById('btn-show-create-modal').addEventListener('click', () => {
        modal.classList.add('active');
    });

    document.querySelectorAll('.modal-close').forEach(el => {
        el.addEventListener('click', () => {
            modal.classList.remove('active');
        });
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });

    // Back to list
    document.getElementById('btn-back-to-list').addEventListener('click', () => {
        showKBList();
    });

    // File Upload
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        Array.from(files).forEach(file => {
            filesToUpload.push(file);
            const div = document.createElement('div');
            div.className = 'file-item';
            div.innerHTML = `
                <span>${file.name}</span>
                <span class="badge">${(file.size / 1024).toFixed(1)} KB</span>
            `;
            fileList.appendChild(div);
        });
    }

    // === API Actions ===

    // Load Knowledge Bases
    async function loadKnowledgeBases() {
        try {
            const res = await fetch('/api/kb');
            const kbs = await res.json();

            const container = document.getElementById('kb-list-container');
            if (kbs.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <p>还没有知识库</p>
                        <p style="font-size:0.9rem; margin-top:0.5rem;">点击右上角"创建知识库"开始</p>
                    </div>
                `;
            } else {
                container.innerHTML = kbs.map(kb => `
                    <div class="card kb-card" onclick="window.selectKB('${kb.id}')">
                        <div class="kb-card-header">
                            <div>
                                <div class="kb-card-title">${kb.name}</div>
                                <p style="color:var(--text-muted); font-size:0.9rem; margin-top:0.25rem;">${kb.description || '无描述'}</p>
                            </div>
                        </div>
                        <div class="kb-card-meta">
                            <span class="badge primary">${kb.vdb_type}</span>
                            ${kb.embedder_name ? `<span class="badge success">${kb.embedder_name}</span>` : '<span class="badge">无 Embedder</span>'}
                            <span class="badge">Chunk: ${kb.chunk_size}</span>
                        </div>
                    </div>
                `).join('');
            }
        } catch (e) {
            showToast('加载知识库失败', 'error');
        }
    }

    // Select KB
    window.selectKB = async function (kbId) {
        currentKbId = kbId;

        try {
            const res = await fetch(`/api/kb/${kbId}`);
            const kb = await res.json();

            // Update UI
            document.getElementById('kb-detail-name').textContent = kb.name;
            document.getElementById('kb-detail-desc').textContent = kb.description || '无描述';
            document.getElementById('kb-info-id').textContent = kb.id;
            document.getElementById('kb-info-vdb').textContent = kb.vdb_type;
            document.getElementById('kb-info-embedder').textContent = kb.embedder_name || '未配置';
            document.getElementById('kb-info-chunk').textContent = `${kb.chunk_size} / ${kb.chunk_overlap}`;

            // Load documents
            await loadDocuments(kbId);

            // Show detail view
            showKBDetail();

        } catch (e) {
            showToast('加载知识库详情失败', 'error');
        }
    };

    // Show KB List
    function showKBList() {
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById('kb-list').classList.add('active');

        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.querySelector('[data-target="kb-list"]').classList.add('active');

        loadKnowledgeBases();
    }

    // Show KB Detail
    function showKBDetail() {
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById('kb-detail').classList.add('active');

        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        const detailNav = document.querySelector('[data-target="kb-detail"]');
        if (detailNav) {
            detailNav.style.display = 'flex';
            detailNav.classList.add('active');
        }
    }

    // Load Documents
    async function loadDocuments(kbId) {
        try {
            const res = await fetch(`/api/upload/documents/${kbId}`);
            const docs = await res.json();

            const container = document.getElementById('documents-list');
            if (docs.length === 0) {
                container.innerHTML = '<p style="color:var(--text-muted);">还没有上传文档</p>';
            } else {
                container.innerHTML = docs.map(doc => `
                    <div class="file-item">
                        <div>
                            <div>${doc.filename}</div>
                            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.25rem;">
                                ${(doc.file_size / 1024).toFixed(1)} KB · ${doc.chunk_count} chunks
                            </div>
                        </div>
                        <span class="badge ${doc.status === 'completed' ? 'success' : ''}">${doc.status}</span>
                    </div>
                `).join('');
            }
        } catch (e) {
            console.error('Load documents error:', e);
        }
    }

    // Load Embedders
    async function loadEmbedders() {
        try {
            const res = await fetch('/api/config/embedders');
            const data = await res.json();

            const select = document.getElementById('modal-kb-embedder');
            const listContainer = document.getElementById('embedder-list');

            // Clear existing options (except the first "不使用" option)
            while (select.options.length > 1) {
                select.remove(1);
            }

            if (data.embedders && data.embedders.length > 0) {
                // Add all embedders to dropdown
                data.embedders.forEach(emb => {
                    const option = document.createElement('option');
                    option.value = emb.name;
                    option.textContent = `${emb.name} (${emb.model})`;
                    select.appendChild(option);
                });

                // Display embedders list
                listContainer.innerHTML = data.embedders.map(emb => `
                    <div class="file-item">
                        <div>
                            <div><strong>${emb.name}</strong> <span class="badge">${emb.embedder_type}</span></div>
                            <div style="font-size:0.9rem; color:var(--text-muted); margin-top:0.25rem;">${emb.model}</div>
                        </div>
                        ${emb.is_active ? '<span class="badge success">激活</span>' : '<span class="badge">未激活</span>'}
                    </div>
                `).join('');
            } else {
                listContainer.innerHTML = '<p style="color:var(--text-muted);">还没有配置模型，请在下方添加</p>';
            }
        } catch (e) {
            console.error('Load embedders error:', e);
        }
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
        if (!embedderName) return showToast('请选择 Embedding 模型（在"模型配置"页面添加）', 'error');

        setLoading(btn, true);
        try {
            const res = await fetch('/api/kb', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name,
                    description: desc,
                    vdb_type: vdbType,
                    embedder_name: embedderName,
                    chunk_size: chunkSize,
                    chunk_overlap: chunkOverlap
                })
            });
            const data = await res.json();
            if (res.ok) {
                showToast('知识库创建成功！');
                modal.classList.remove('active');
                // Clear form
                document.getElementById('modal-kb-name').value = '';
                document.getElementById('modal-kb-desc').value = '';
                // Reload list
                loadKnowledgeBases();
            } else {
                showToast(data.detail || '创建失败', 'error');
            }
        } catch (e) {
            showToast('网络错误', 'error');
        } finally {
            setLoading(btn, false);
        }
    });

    // Upload Documents
    document.getElementById('btn-upload').addEventListener('click', async () => {
        if (!currentKbId) return showToast('请先选择知识库', 'error');
        if (filesToUpload.length === 0) return showToast('没有可上传的文件', 'error');

        const btn = document.getElementById('btn-upload');
        setLoading(btn, true);

        const formData = new FormData();
        formData.append('kb_id', currentKbId);
        filesToUpload.forEach(file => formData.append('files', file));

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                const data = await res.json();
                showToast(`成功！处理了 ${data.processed_files} 个文件，共 ${data.total_chunks} 个块。`);
                if (data.failed_files.length > 0) {
                    showToast('部分文件失败: ' + data.failed_files.join(', '), 'error');
                }
                filesToUpload = [];
                fileList.innerHTML = '';
                // Reload documents
                await loadDocuments(currentKbId);
            } else {
                const err = await res.json();
                showToast('上传失败: ' + err.detail, 'error');
            }
        } catch (e) {
            showToast('网络错误', 'error');
            console.error(e);
        } finally {
            setLoading(btn, false);
        }
    });

    // Search
    document.getElementById('btn-search').addEventListener('click', async () => {
        if (!currentKbId) return showToast('请先选择知识库', 'error');

        const query = document.getElementById('search-query').value;
        const resultsDiv = document.getElementById('search-results');
        const btn = document.getElementById('btn-search');

        if (!query) return;

        setLoading(btn, true);
        resultsDiv.innerHTML = '';

        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    kb_id: currentKbId,
                    query,
                    top_k: 5
                })
            });
            const data = await res.json();

            if (res.ok) {
                if (data.results.length === 0) {
                    resultsDiv.innerHTML = '<div style="color:var(--text-muted); text-align:center;">未找到结果。</div>';
                } else {
                    data.results.forEach(item => {
                        const el = document.createElement('div');
                        el.className = 'result-item';
                        const typeLabel = item.search_type === 'vector'
                            ? '<span class="badge success">向量检索</span>'
                            : '<span class="badge">关键词</span>';

                        el.innerHTML = `
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                                <div class="result-score">Score: ${item.score.toFixed(3)}</div>
                                ${typeLabel}
                            </div>
                            <div class="result-content">${item.content}</div>
                            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.5rem;">来源: ${item.source || '未知'}</div>
                        `;
                        resultsDiv.appendChild(el);
                    });
                }
            } else {
                const err = await res.json();
                showToast('搜索失败: ' + err.detail, 'error');
            }
        } catch (e) {
            showToast('网络错误', 'error');
        } finally {
            setLoading(btn, false);
        }
    });

    // Embedder Type Toggle
    const embTypeSelect = document.getElementById('emb-type');
    const baseUrlGroup = document.getElementById('emb-base-url-group');
    const apiKeyGroup = document.getElementById('emb-api-key-group');
    const modelGroup = document.getElementById('emb-model-group');
    const seekdbInfo = document.getElementById('seekdb-info');

    embTypeSelect.addEventListener('change', () => {
        const type = embTypeSelect.value;
        if (type === 'seekdb') {
            baseUrlGroup.style.display = 'none';
            apiKeyGroup.style.display = 'none';
            modelGroup.style.display = 'none';
            seekdbInfo.style.display = 'block';
        } else {
            baseUrlGroup.style.display = 'block';
            apiKeyGroup.style.display = 'block';
            modelGroup.style.display = 'block';
            seekdbInfo.style.display = 'none';
        }
    });

    // Save Embedder Config
    document.getElementById('btn-save-config').addEventListener('click', async () => {
        const btn = document.getElementById('btn-save-config');
        const name = document.getElementById('emb-name').value;
        const embType = document.getElementById('emb-type').value;
        const baseUrl = document.getElementById('emb-base-url').value;
        const apiKey = document.getElementById('emb-api-key').value;
        let model = document.getElementById('emb-model').value;

        if (!name) return showToast('请填写配置名称', 'error');

        // Validate and set model based on type
        if (embType === 'openai') {
            if (!baseUrl || !apiKey || !model) {
                return showToast('OpenAI 类型需要 Base URL、API Key 和 Model Name', 'error');
            }
        } else if (embType === 'seekdb') {
            // SeekDB 使用固定的模型名称
            model = 'all-MiniLM-L6-v2';
        }

        setLoading(btn, true);
        try {
            const res = await fetch('/api/config/embedder', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name,
                    embedder_type: embType,
                    model: model,
                    base_url: baseUrl,
                    api_key: apiKey
                })
            });
            if (res.ok) {
                showToast('配置已保存！');
                loadEmbedders();
                // Clear form
                document.getElementById('emb-name').value = '';
                document.getElementById('emb-base-url').value = '';
                document.getElementById('emb-api-key').value = '';
                document.getElementById('emb-model').value = '';
            } else {
                const err = await res.json();
                showToast('保存失败: ' + err.detail, 'error');
            }
        } catch (e) {
            showToast('网络错误', 'error');
        } finally {
            setLoading(btn, false);
        }
    });

    // Utils
    function setLoading(btn, isLoading) {
        const loader = btn.querySelector('.loader');
        if (loader) {
            if (isLoading) loader.classList.add('active');
            else loader.classList.remove('active');
        }
        btn.disabled = isLoading;
    }

    function showToast(msg, type = 'success') {
        const container = document.getElementById('toast-container');
        const el = document.createElement('div');
        el.className = 'toast';
        el.style.borderLeft = `4px solid ${type === 'error' ? 'var(--error)' : 'var(--success)'}`;
        el.innerHTML = `<span>${msg}</span>`;
        container.appendChild(el);
        setTimeout(() => {
            el.remove();
        }, 3000);
    }
});
