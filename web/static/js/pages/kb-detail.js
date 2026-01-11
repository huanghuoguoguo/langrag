/**
 * KB Detail Page Component
 */

function kbDetailPage() {
    return {
        filesToUpload: [],
        uploading: false,
        searchQuery: '',

        get kb() {
            return Alpine.store('kbs').current;
        },

        get documents() {
            return Alpine.store('kbs').documents;
        },

        get searchResults() {
            return Alpine.store('kbs').searchResults;
        },

        get searchLoading() {
            return Alpine.store('kbs').searchLoading;
        },

        get searchMode() {
            return Alpine.store('kbs').searchMode;
        },

        set searchMode(val) {
            Alpine.store('kbs').searchMode = val;
        },

        get useRerank() {
            return Alpine.store('kbs').useRerank;
        },

        set useRerank(val) {
            Alpine.store('kbs').useRerank = val;
        },

        goBack() {
            Alpine.store('kbs').load();
            Alpine.store('nav').goto('kb-list');
        },

        handleFiles(event) {
            const files = event.target.files || event.dataTransfer?.files;
            if (files) {
                this.filesToUpload = [...this.filesToUpload, ...Array.from(files)];
            }
        },

        removeFile(index) {
            this.filesToUpload.splice(index, 1);
        },

        async upload() {
            if (this.filesToUpload.length === 0) {
                showToast('请选择要上传的文件', 'error');
                return;
            }

            this.uploading = true;
            const formData = new FormData();
            formData.append('kb_id', this.kb.id);
            this.filesToUpload.forEach(f => formData.append('files', f));

            try {
                await api.uploadDocuments(formData);
                showToast('文档处理完成', 'success');
                this.filesToUpload = [];
                Alpine.store('kbs').documents = await api.listDocuments(this.kb.id);
            } catch (e) {
                showToast(e.message, 'error');
            } finally {
                this.uploading = false;
            }
        },

        async search() {
            await Alpine.store('kbs').search(this.searchQuery);
        },

        deleteKB() {
            Alpine.store('kbs').deleteKB(this.kb.id);
        }
    };
}

window.kbDetailPage = kbDetailPage;
