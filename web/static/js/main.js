
import * as kbListController from './controllers/kb_list.js';
import * as kbDetailController from './controllers/kb_detail.js';
import * as chatController from './controllers/chat.js';
import * as modelsController from './controllers/models.js';

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Controllers
    // Pass callback to list controller for when a KB is clicked
    kbListController.init(onKbSelected);

    kbDetailController.init();
    chatController.init();
    modelsController.init();

    setupNavigation();
});

// View Navigation Logic
function onKbSelected(kbId) {
    kbDetailController.loadKB(kbId);
    showView('kb-detail');
}

function showView(targetId) {
    document.querySelectorAll('.section').forEach(s => {
        s.classList.remove('active');
        if (s.id === targetId) s.classList.add('active');
    });

    document.querySelectorAll('.nav-item').forEach(n => {
        n.classList.remove('active');
        if (n.dataset.target === targetId) {
            n.classList.add('active');
            // Ensure detail tab is visible if we switch to it
            if (n.dataset.target === 'kb-detail') n.style.display = 'flex';
        }
    });
}

function setupNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const target = item.dataset.target;
            if (target) {
                // If clicking KB list, refresh it
                if (target === 'kb-list') {
                    kbListController.refreshKBList();
                }
                showView(target);
            }
        });
    });

    // Special case for 'Back to List' button
    document.getElementById('btn-back-to-list')?.addEventListener('click', () => {
        kbListController.refreshKBList();
        showView('kb-list');
    });
}
