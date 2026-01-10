
// State Management
// Simple reactive state store pattern

const state = {
    currentKbId: null,
    filesToUpload: []
};

// Event Bus for state changes (simple pub/sub if needed, 
// strictly speaking regular object export is enough if we import 'state' directly)
// But getters/setters are safer.

export function getCurrentKbId() {
    return state.currentKbId;
}

export function setCurrentKbId(id) {
    state.currentKbId = id;
}

export function getFilesToUpload() {
    return state.filesToUpload;
}

export function setFilesToUpload(files) {
    state.filesToUpload = files;
}

export function addFilesToUpload(newFiles) {
    // Array.from in case it's a FileList
    Array.from(newFiles).forEach(f => state.filesToUpload.push(f));
}

export function clearFilesToUpload() {
    state.filesToUpload = [];
}
