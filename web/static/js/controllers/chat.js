
import * as api from '../api.js';
import * as ui from '../ui.js';
import { setLoading, showToast } from '../utils.js';

export function init() {
    setupChat();
}

function setupChat() {
    document.getElementById('btn-send')?.addEventListener('click', sendMessage);
    document.getElementById('chat-input')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
}

async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const btn = document.getElementById('btn-send');

    const query = chatInput.value.trim();
    if (!query) return;

    // Add user message
    ui.appendMessage(chatMessages, 'user', query);
    chatInput.value = '';

    setLoading(btn, true);

    try {
        const history = []; // TODO: Extract history from DOM if needed for context
        // Collect selected KB IDs - currently empty implies auto-routing
        const selectedKBs = [];

        const data = await api.chat({
            kb_ids: selectedKBs,
            query: query,
            history: history
        });

        // Add assistant message and get the element back
        const el = ui.appendMessage(chatMessages, 'assistant', data.answer, data.sources, query);

        // Attach eval handler to the newly created message
        attachEvalHandler(el, query, data);

    } catch (e) {
        ui.appendMessage(chatMessages, 'assistant', `Error: ${e.message}`);
    } finally {
        setLoading(btn, false);
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function attachEvalHandler(messageEl, query, data) {
    const evalBtn = messageEl.querySelector('.btn-eval');
    if (evalBtn) {
        evalBtn.addEventListener('click', async () => {
            // Prepare data
            const contexts = (data.sources || []).map(s => s.content);
            const evalRequest = {
                question: query,
                answer: data.answer,
                contexts: contexts
            };

            const resultContainer = messageEl.querySelector('.eval-result');

            evalBtn.disabled = true;
            evalBtn.innerHTML = '评估中...';
            evalBtn.style.cursor = 'wait';

            try {
                const result = await api.evaluateAnswer(evalRequest);
                ui.renderEvaluationResult(resultContainer, result);
                evalBtn.style.display = 'none'; // Hide button after success
            } catch (e) {
                console.error(e);
                evalBtn.innerHTML = '❌ 评估失败 (点击重试)';
                evalBtn.disabled = false;
                evalBtn.style.cursor = 'pointer';
                showToast(e.message, 'error');
            }
        });
    }
}
