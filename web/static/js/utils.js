export function setLoading(btn, isLoading) {
    const loader = btn.querySelector('.loader');
    if (loader) {
        if (isLoading) loader.classList.add('active');
        else loader.classList.remove('active');
    }
    btn.disabled = isLoading;
}

export function showToast(msg, type = 'success') {
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
