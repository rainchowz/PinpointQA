document.addEventListener("DOMContentLoaded", () => {
    const copyBtn = document.getElementById("copyBibBtn");
    const bibtexCode = document.getElementById("bibtex-code");

    if (copyBtn && bibtexCode) {
        copyBtn.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(bibtexCode.innerText);
                copyBtn.innerHTML = '<ion-icon name="checkmark-outline"></ion-icon>';
                copyBtn.style.color = "#34d399";

                setTimeout(() => {
                    copyBtn.innerHTML = '<ion-icon name="copy-outline"></ion-icon>';
                    copyBtn.style.color = "";
                }, 1800);
            } catch (error) {
                console.error("Failed to copy BibTeX:", error);
            }
        });
    }

    const navLinks = Array.from(document.querySelectorAll(".nav-link"));
    const sections = Array.from(document.querySelectorAll("main .section[id]"));

    const setActiveNav = () => {
        let currentId = "";

        sections.forEach((section) => {
            const rect = section.getBoundingClientRect();
            if (rect.top <= 140 && rect.bottom >= 140) {
                currentId = section.id;
            }
        });

        navLinks.forEach((link) => {
            link.classList.toggle("active", link.getAttribute("href") === `#${currentId}`);
        });
    };

    setActiveNav();
    window.addEventListener("scroll", setActiveNav, { passive: true });

    const tbody = document.getElementById("leaderboardBody");
    const sortButtons = Array.from(document.querySelectorAll(".sort-btn"));
    const sortStatus = document.getElementById("sortStatus");

    if (!tbody || sortButtons.length === 0) {
        return;
    }

    const baseRows = Array.from(tbody.querySelectorAll("tr"));
    let currentSort = {
        key: "avgMicro",
        type: "number",
        order: "desc"
    };

    const formatKeyLabel = (key) => {
        return key.replace(/([A-Z])/g, " $1").replace(/^./, (char) => char.toUpperCase());
    };

    const getValue = (row, key, type) => {
        if (key === "model") {
            return row.dataset.model || "";
        }

        const value = row.dataset[key];
        return type === "number" ? Number.parseFloat(value || "0") : (value || "");
    };

    const updateButtons = (activeButton, order) => {
        sortButtons.forEach((button) => {
            const icon = button.querySelector("ion-icon");
            button.classList.remove("active");

            if (!icon) return;

            if (button === activeButton) {
                button.classList.add("active");
                icon.setAttribute("name", order === "desc" ? "arrow-down-outline" : "arrow-up-outline");
            } else {
                icon.setAttribute("name", "swap-vertical-outline");
            }
        });
    };

    const renderRanks = () => {
        const rows = Array.from(tbody.querySelectorAll("tr"));
        rows.forEach((row, index) => {
            const rankCell = row.querySelector(".rank-cell");
            if (rankCell) {
                rankCell.textContent = index + 1;
            }
            row.classList.toggle("top-rank", index < 3);
        });
    };

    const renderRows = () => {
        const sorted = [...baseRows].sort((a, b) => {
            const aValue = getValue(a, currentSort.key, currentSort.type);
            const bValue = getValue(b, currentSort.key, currentSort.type);

            if (currentSort.type === "text") {
                return currentSort.order === "asc"
                    ? String(aValue).localeCompare(String(bValue))
                    : String(bValue).localeCompare(String(aValue));
            }

            return currentSort.order === "asc" ? aValue - bValue : bValue - aValue;
        });

        tbody.innerHTML = "";
        sorted.forEach((row) => tbody.appendChild(row));
        renderRanks();

        if (sortStatus) {
            sortStatus.textContent = `Sorted by ${formatKeyLabel(currentSort.key)} (${currentSort.order === "desc" ? "high to low" : "low to high"})`;
        }
    };

    sortButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const key = button.dataset.sortKey;
            const type = button.dataset.sortType || "text";

            if (!key) return;

            if (currentSort.key === key) {
                currentSort.order = currentSort.order === "desc" ? "asc" : "desc";
            } else {
                currentSort.key = key;
                currentSort.type = type;
                currentSort.order = button.dataset.defaultOrder || (type === "number" ? "desc" : "asc");
            }

            updateButtons(button, currentSort.order);
            renderRows();
        });
    });

    const defaultButton = document.querySelector('.sort-btn[data-sort-key="avgMicro"]');
    if (defaultButton) {
        updateButtons(defaultButton, "desc");
    }
    renderRows();
});
