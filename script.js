document.addEventListener('DOMContentLoaded', () => {
    // BibTeX Copy Functionality
    const copyBtn = document.getElementById('copyBibBtn');
    const bibtexCode = document.getElementById('bibtex-code');

    if (copyBtn && bibtexCode) {
        copyBtn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(bibtexCode.innerText);
                // Update icon to checkmark temporarily
                copyBtn.innerHTML = '<ion-icon name="checkmark-outline"></ion-icon>';
                copyBtn.style.color = '#10B981';
                setTimeout(() => {
                    copyBtn.innerHTML = '<ion-icon name="copy-outline"></ion-icon>';
                    copyBtn.style.color = '';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
        });
    }

    // Smooth reveal animation on scroll
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const revealOnScroll = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe sections for scroll animation
    document.querySelectorAll('.section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        revealOnScroll.observe(section);
    });

    // Add active state for buttons on touch devices
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('touchstart', function() {
            this.style.transform = 'translateY(-2px)';
        });
        btn.addEventListener('touchend', function() {
            this.style.transform = '';
        });
    });
});
