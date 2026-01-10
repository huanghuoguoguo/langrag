
export async function fetchKnowledgeBases() {
    const res = await fetch('/api/kb');
    return await res.json();
}

export async function fetchKBDetail(kbId) {
    const res = await fetch(`/api/kb/${kbId}`);
    return await res.json();
}

export async function createKB(data) {
    const res = await fetch('/api/kb', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '创建失败');
    }
    return await res.json();
}

export async function fetchDocuments(kbId) {
    const res = await fetch(`/api/upload/documents/${kbId}`);
    return await res.json();
}

export async function uploadDocuments(formData) {
    const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '上传失败');
    }
    return await res.json();
}

export async function searchKB(searchData) {
    const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(searchData)
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '搜索失败');
    }
    return await res.json();
}

export async function chat(chatData) {
    const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chatData)
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '对话失败');
    }
    return await res.json();
}

export async function fetchEmbedders() {
    const res = await fetch('/api/config/embedders');
    return await res.json();
}

export async function activateEmbedderConfig(name) {
    const res = await fetch('/api/config/embedder/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    });
    if (!res.ok) throw new Error('激活失败');
}

export async function saveEmbedderConfig(data) {
    const res = await fetch('/api/config/embedder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '保存失败');
    }
}

export async function fetchLLMs() {
    const res = await fetch('/api/config/llms');
    return await res.json();
}

export async function saveLLMConfig(data) {
    const res = await fetch('/api/config/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '保存失败');
    }
}

export async function evaluateAnswer(data) {
    const res = await fetch('/api/chat/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || '评估失败');
    }
    return await res.json();
}
