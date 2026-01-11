/**
 * KB List Page Component
 */

function kbListPage() {
    return {
        showModal: false,
        form: {
            name: '',
            description: '',
            vdb_type: 'seekdb',
            embedder_name: '',
            chunk_size: 1000,
            chunk_overlap: 100
        },
        creating: false,

        init() {
            Alpine.store('kbs').load();
            Alpine.store('models').load();
        },

        get kbs() {
            return Alpine.store('kbs').list;
        },

        get loading() {
            return Alpine.store('kbs').loading;
        },

        get embedders() {
            return Alpine.store('models').embedders;
        },

        get isWebSearch() {
            return this.form.vdb_type === 'web_search';
        },

        openModal() {
            this.form = {
                name: '',
                description: '',
                vdb_type: 'seekdb',
                embedder_name: '',
                chunk_size: 1000,
                chunk_overlap: 100
            };
            this.showModal = true;
        },

        async createKB() {
            if (!this.form.name) {
                showToast('请输入知识库名称', 'error');
                return;
            }
            if (!this.isWebSearch && !this.form.embedder_name) {
                showToast('请选择 Embedder', 'error');
                return;
            }

            this.creating = true;
            const success = await Alpine.store('kbs').create(this.form);
            this.creating = false;
            if (success) {
                this.showModal = false;
            }
        },

        selectKB(kb) {
            Alpine.store('kbs').loadDetail(kb.id);
            Alpine.store('nav').goto('kb-detail', kb);
        }
    };
}

window.kbListPage = kbListPage;
