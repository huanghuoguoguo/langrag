
export function renderKBList(kbs, container) {
    if (kbs.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <p>还没有知识库</p>
                <p style="font-size:0.9rem; margin-top:0.5rem;">点击右上角"创建知识库"开始</p>
            </div>
        `;
    } else {
        container.innerHTML = kbs.map(kb => `
            <div class="card kb-card" data-id="${kb.id}">
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
}

export function renderKBCheckboxes(kbs, container) {
    if (kbs.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted); font-size:0.9rem;">暂无可用知识库</p>';
    } else {
        container.innerHTML = kbs.map(kb => `
            <div style="display:flex; align-items:center; gap:0.75rem; padding:0.75rem 0.5rem; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="width:8px; height:8px; background:var(--success); border-radius:50%; box-shadow: 0 0 5px var(--success);"></span>
                <div>
                    <div style="font-size:0.9rem; font-weight:500;">${kb.name}</div>
                    <div style="font-size:0.8rem; color:var(--text-muted);">${kb.vdb_type} · Ready</div>
                </div>
            </div>
        `).join('');
    }
}

export function renderKBDetail(kb) {
    document.getElementById('kb-detail-name').textContent = kb.name;
    document.getElementById('kb-detail-desc').textContent = kb.description || '无描述';
    document.getElementById('kb-info-id').textContent = kb.id;
    document.getElementById('kb-info-vdb').textContent = kb.vdb_type;
    document.getElementById('kb-info-embedder').textContent = kb.embedder_name || '未配置';
    document.getElementById('kb-info-chunk').textContent = `${kb.chunk_size} / ${kb.chunk_overlap}`;
}

export function renderDocuments(docs, container) {
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
}

export function renderEmbedders(embedders, listContainer, selectElement) {
    // Render List
    if (embedders.length === 0) {
        listContainer.innerHTML = '<p style="color:var(--text-muted); font-size:0.9rem;">暂无已配置的模型</p>';
    } else {
        listContainer.innerHTML = '';
        embedders.forEach(emb => {
            const el = document.createElement('div');
            el.className = 'kb-item'; // Note: Reusing a class that might need to be defined in CSS if it was ad-hoc
            // Actually in original code it said "Reuse style" but there was no .kb-item class in CSS, it used card or similar? 
            // In original script.js: el.className = 'kb-item'; 
            // But looking at CSS file, there IS NO .kb-item class. 
            // It might have relied on generic div styling or inherited.
            // Let's use 'file-item' which exists and looks similar (flex row).
            el.className = 'file-item';
            el.style.cursor = 'pointer';

            el.innerHTML = `
                <div>
                    <div><strong>${emb.name}</strong> <span class="badge">${emb.embedder_type}</span></div>
                    <div style="font-size:0.9rem; color:var(--text-muted); margin-top:0.25rem;">${emb.model}</div>
                </div>
                ${emb.is_active ? '<span class="badge success">Active</span>' : ''}
            `;
            el.dataset.name = emb.name; // Store name for click handler
            listContainer.appendChild(el);
        });
    }

    // Render Select options
    // Clear existing options (except the first "Please select" option if we want to keep it, but here we rebuild)
    selectElement.innerHTML = '<option value="">请选择 Embedder...</option>';
    embedders.forEach(emb => {
        const opt = document.createElement('option');
        opt.value = emb.name;
        opt.textContent = `${emb.name} (${emb.embedder_type})`;
        selectElement.appendChild(opt);
    });
}

export function renderLLMs(llms, container) {
    container.innerHTML = '';
    if (llms.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted); font-size:0.9rem;">暂无已配置的 LLM</p>';
        return;
    }

    llms.forEach(llm => {
        const el = document.createElement('div');
        el.className = 'file-item'; // reusing file-item for consistent look
        el.innerHTML = `
            <div>
                <div><strong>${llm.name}</strong></div>
                <div style="font-size:0.9rem; color:var(--text-muted); margin-top:0.25rem;">${llm.model}</div>
            </div>
            ${llm.is_active ? '<span class="badge success">Active</span>' : ''}
        `;
        // Click handler logic will be attached in main.js if needed
        container.appendChild(el);
    });
}

export function renderSearchResults(results, container) {
    if (results.length === 0) {
        container.innerHTML = '<div style="color:var(--text-muted); text-align:center;">未找到结果。</div>';
    } else {
        container.innerHTML = '';
        results.forEach(item => {
            const el = document.createElement('div');
            el.className = 'result-item';
            const typeLabel = item.search_type === 'vector'
                ? '<span class="badge success">向量检索</span>'
                : (item.search_type === 'hybrid' ? '<span class="badge" style="background:purple;color:white;">混合检索</span>' : '<span class="badge">关键词</span>');

            // Detect if this is a web source
            const isWebSource = item.type === 'web_search' || (item.link && item.link.startsWith('http'));
            const icon = isWebSource ? '🌐' : '📄';
            const sourceDisplay = isWebSource && item.link
                ? `<a href="${item.link}" target="_blank" rel="noopener noreferrer" 
                     style="color:var(--primary); text-decoration:none;">
                     ${item.title || item.source || '互联网来源'}
                     <span style="margin-left:0.25rem;">↗</span>
                   </a>`
                : `${item.source || '未知'}`;

            el.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <div class="result-score">Score: ${item.score.toFixed(3)}</div>
                    ${typeLabel}
                </div>
                <div class="result-content">${item.content}</div>
                <div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.5rem; display:flex; align-items:center; gap:0.5rem;">
                    <span style="font-size:1rem;">${icon}</span>
                    <span>来源: ${sourceDisplay}</span>
                </div>
            `;
            container.appendChild(el);
        });
    }
}

export function appendMessage(container, role, content, sources = []) {
    const el = document.createElement('div');
    el.className = `message ${role}`;

    let contentHtml = content.replace(/\n/g, '<br>');
    if (sources && sources.length > 0) {
        contentHtml += '<div style="margin-top:1rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.1); font-size:0.85rem;">';
        contentHtml += '<strong>参考来源:</strong><br>';
        sources.forEach((s, i) => {
            // Detect if this is a web source
            const isWebSource = s.type === 'web_search' || (s.link && s.link.startsWith('http'));
            const icon = isWebSource ? '🌐' : '📄';
            // Use kb_name if available, otherwise fallback to kb_id
            const kbText = s.kb_name || s.kb_id;
            const kbLabel = kbText ? ` <span class="badge" style="font-size:0.75rem;">${kbText}</span>` : '';

            if (isWebSource && s.link) {
                // Web source with clickable link
                contentHtml += `
                    <div style="margin-top:0.5rem; padding:0.5rem; background:rgba(255,255,255,0.03); border-radius:4px;">
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="font-size:1.2rem;">${icon}</span>
                            <div style="flex:1;">
                                <div style="font-weight:500;">
                                    ${s.title || `来源 ${i + 1}`}
                                    ${kbLabel}
                                </div>
                                <a href="${s.link}" target="_blank" rel="noopener noreferrer" 
                                   style="color:var(--primary); text-decoration:none; font-size:0.8rem; word-break:break-all;">
                                    ${s.link}
                                    <span style="margin-left:0.25rem;">↗</span>
                                </a>
                                <div style="color:var(--text-muted); font-size:0.75rem; margin-top:0.25rem;">
                                    Score: ${s.score.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                // Local document source
                contentHtml += `
                    <div style="margin-top:0.5rem; padding:0.5rem; background:rgba(255,255,255,0.03); border-radius:4px;">
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="font-size:1.2rem;">${icon}</span>
                            <div style="flex:1;">
                                <div style="font-weight:500;">
                                    ${s.source || `来源 ${i + 1}`}
                                    ${kbLabel}
                                </div>
                                <div style="color:var(--text-muted); font-size:0.75rem; margin-top:0.25rem;">
                                    Score: ${s.score.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
        });
        contentHtml += '</div>';
    }

    el.innerHTML = `
        <div class="avatar">${role === 'user' ? 'U' : 'AI'}</div>
        <div class="message-content">${contentHtml}</div>
    `;

    container.appendChild(el);
    container.scrollTop = container.scrollHeight;
}

export function renderFilesToUpload(files, container) {
    Array.from(files).forEach(file => {
        const div = document.createElement('div');
        div.className = 'file-item';
        div.innerHTML = `
            <span>${file.name}</span>
            <span class="badge">${(file.size / 1024).toFixed(1)} KB</span>
        `;
        container.appendChild(div);
    });
}
