document.addEventListener('DOMContentLoaded', () => {
    const copyBtn = document.getElementById('copyBibBtn');
    const bibtexCode = document.getElementById('bibtex-code');

    if (copyBtn && bibtexCode) {
        copyBtn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(bibtexCode.innerText);
                // Update icon to checkmark temporarily
                copyBtn.innerHTML = '<ion-icon name="checkmark-outline"></ion-icon>';
                setTimeout(() => {
                    copyBtn.innerHTML = '<ion-icon name="copy-outline"></ion-icon>';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
        });
    }
});
