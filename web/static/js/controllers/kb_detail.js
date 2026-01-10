
import * as api from '../api.js';
import * as ui from '../ui.js';
import * as state from '../state.js';
import { setLoading, showToast } from '../utils.js';

export function init() {
    setupFileUpload();
    setupSearchTest();
}

export async function loadKB(kbId) {
    state.setCurrentKbId(kbId);
    try {
        // Parallel fetch
        const [kb, docs] = await Promise.all([
            api.fetchKBDetail(kbId),
            refreshDocuments(kbId) // returns nothing, but we fetch it
        ]);

        ui.renderKBDetail(kb);
        // refreshDocuments already renders docs
    } catch (e) {
        showToast('加载知识库详情失败', 'error');
        console.error(e);
    }
}

export async function refreshDocuments(kbId) {
    if (!kbId) kbId = state.getCurrentKbId();
    if (!kbId) return;

    try {
        const docs = await api.fetchDocuments(kbId);
        ui.renderDocuments(docs, document.getElementById('documents-list'));
        return docs;
    } catch (e) {
        console.error('Load documents error:', e);
    }
}

function setupFileUpload() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const fileListForUpload = document.getElementById('file-list');

    if (!dropzone) return;

    // Drag & Drop
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
        state.addFilesToUpload(files);
        ui.renderFilesToUpload(files, fileListForUpload);
    }

    // Upload Action
    document.getElementById('btn-upload')?.addEventListener('click', async () => {
        const currentKbId = state.getCurrentKbId();
        const filesToUpload = state.getFilesToUpload();

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
            state.clearFilesToUpload();
            fileListForUpload.innerHTML = '';
            refreshDocuments(currentKbId);
        } catch (e) {
            showToast(e.message, 'error');
        } finally {
            setLoading(btn, false);
        }
    });
}

function setupSearchTest() {
    document.getElementById('btn-search')?.addEventListener('click', async () => {
        const currentKbId = state.getCurrentKbId();
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
}
