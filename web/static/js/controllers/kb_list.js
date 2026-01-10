
import * as api from '../api.js';
import * as ui from '../ui.js';
import * as state from '../state.js';
import { setLoading, showToast } from '../utils.js';

let _onKbSelectCallback = null;

export function init(onKbSelect) {
    _onKbSelectCallback = onKbSelect;

    // Initial Load
    refreshKBList();

    // Event Listeners
    setupCreateModal();

    // Back to list button (this technically belongs to detail view but navigates to list)
    // We'll let main.js handle global navigation, but here we handle internal list actions
    document.getElementById('btn-show-create-modal')?.addEventListener('click', () => {
        document.getElementById('create-kb-modal').classList.add('active');
    });
}

export async function refreshKBList() {
    const container = document.getElementById('kb-list-container');
    const chatKbList = document.getElementById('chat-kb-list'); // Also update chat side list

    try {
        const data = await api.fetchKnowledgeBases();
        ui.renderKBList(data, container);
        if (chatKbList) ui.renderKBCheckboxes(data, chatKbList);

        // Re-attach click listeners for KB cards
        container.querySelectorAll('.kb-card').forEach(card => {
            card.addEventListener('click', () => {
                const kbId = card.dataset.id;
                if (_onKbSelectCallback) _onKbSelectCallback(kbId);
            });
        });
    } catch (e) {
        showToast('加载知识库失败', 'error');
        console.error(e);
    }
}

function setupCreateModal() {
    const modal = document.getElementById('create-kb-modal');
    if (!modal) return;

    // Modal Close
    modal.querySelectorAll('.modal-close').forEach(el => {
        el.addEventListener('click', () => modal.classList.remove('active'));
    });
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });

    // VDB Type Change (Show/Hide fields)
    const vdbTypeSelect = document.getElementById('modal-kb-vdb');
    const embedderGroup = document.getElementById('modal-kb-embedder-group');
    const chunkSizeGroup = document.getElementById('modal-kb-chunk-size-group');
    const chunkOverlapGroup = document.getElementById('modal-kb-chunk-overlap-group');

    if (vdbTypeSelect) {
        vdbTypeSelect.addEventListener('change', () => {
            const vdbType = vdbTypeSelect.value;
            if (vdbType === 'web_search') {
                if (embedderGroup) embedderGroup.style.display = 'none';
                if (chunkSizeGroup) chunkSizeGroup.style.display = 'none';
                if (chunkOverlapGroup) chunkOverlapGroup.style.display = 'none';
            } else {
                if (embedderGroup) embedderGroup.style.display = 'block';
                if (chunkSizeGroup) chunkSizeGroup.style.display = 'block';
                if (chunkOverlapGroup) chunkOverlapGroup.style.display = 'block';
            }
        });
    }

    // Confirm Create
    document.getElementById('btn-create-kb-confirm')?.addEventListener('click', async () => {
        const name = document.getElementById('modal-kb-name').value;
        const desc = document.getElementById('modal-kb-desc').value;
        const vdbType = document.getElementById('modal-kb-vdb').value;
        const embedderName = document.getElementById('modal-kb-embedder').value;
        const chunkSize = parseInt(document.getElementById('modal-kb-chunk-size').value);
        const chunkOverlap = parseInt(document.getElementById('modal-kb-chunk-overlap').value);
        const btn = document.getElementById('btn-create-kb-confirm');

        if (!name) return showToast('请输入知识库名称', 'error');

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

            if (vdbType !== 'web_search') {
                payload.embedder_name = embedderName;
                payload.chunk_size = chunkSize;
                payload.chunk_overlap = chunkOverlap;
            }

            await api.createKB(payload);
            showToast('知识库创建成功！');
            modal.classList.remove('active');

            // Reset inputs
            document.getElementById('modal-kb-name').value = '';
            document.getElementById('modal-kb-desc').value = '';

            refreshKBList();
        } catch (e) {
            showToast(e.message, 'error');
        } finally {
            setLoading(btn, false);
        }
    });
}
