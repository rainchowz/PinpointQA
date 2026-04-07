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

    if (tbody && sortButtons.length > 0) {
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
    }

    const demoSamples = [
        {
            id: "sample1",
            title: "Sample 1",
            question: "Where is the scissors? Please click on its location.",
            instruction: "Select a frame, click the target location, then save your answer.",
            groundTruthText: "The scissors is to the side of and above the dish drying rack, about 4 cm away. It is also to the side of and below the neon light, about 9 cm away. It is further to the side of and below the kitchen cabinet, about 12 cm away.",
            frames: [
                "assets/human_assist_demo/sample1/1.png",
                "assets/human_assist_demo/sample1/2.png",
                "assets/human_assist_demo/sample1/3.png",
                "assets/human_assist_demo/sample1/4.png"
            ],
            groundTruth: {
                frameIndex: 2,
                x: 0.52728285077951,
                y: 0.4154231625835189
            }
        },
        {
            id: "sample2",
            title: "Sample 2",
            question: "Where is the pen holder? Please click on its location.",
            instruction: "Select a frame, click the target location, then save your answer.",
            groundTruthText: "The pen holder is on the table. It is also under the monitor, about 3 cm away. It is further next to the nintendo switch, about 5 cm away.",
            frames: [
                "assets/human_assist_demo/sample2/1.png",
                "assets/human_assist_demo/sample2/2.png",
                "assets/human_assist_demo/sample2/3.png",
                "assets/human_assist_demo/sample2/4.png"
            ],
            groundTruth: {
                frameIndex: 1,
                x: 0.6275055679287305,
                y: 0.3942238637529496
            }
        },
        {
            id: "sample3",
            title: "Sample 3",
            question: "Where is the mouse? Please click on its location.",
            instruction: "Select a frame, click the target location, then save your answer.",
            groundTruthText: "The mouse is on the table. It is also next to the keyboard, about 3 cm away. It is further next to the headphone, about 5 cm away.",
            frames: [
                "assets/human_assist_demo/sample3/1.png",
                "assets/human_assist_demo/sample3/2.png",
                "assets/human_assist_demo/sample3/3.png",
                "assets/human_assist_demo/sample3/4.png"
            ],
            groundTruth: {
                frameIndex: 2,
                x: 0.6130289532293987,
                y: 0.5169821826280624
            }
        }
    ];

    const DEMO_SCORING_RADIUS = 0.12;

    const demoQuestionText = document.getElementById("demoQuestionText");
    const demoHintPanel = document.getElementById("demoHintPanel");
    const demoHintText = document.getElementById("demoHintText");
    const demoStageCounter = document.getElementById("demoStageCounter");
    const demoStageHint = document.getElementById("demoStageHint");
    const demoMainWrap = document.getElementById("demoMainWrap");
    const demoMainImage = document.getElementById("demoMainImage");
    const demoMainName = document.getElementById("demoMainName");
    const demoMainLock = document.getElementById("demoMainLock");
    const demoClickMarker = document.getElementById("demoClickMarker");
    const demoGtMarker = document.getElementById("demoGtMarker");
    const demoPrevImage = document.getElementById("demoPrevImage");
    const demoPrevName = document.getElementById("demoPrevName");
    const demoNextImage = document.getElementById("demoNextImage");
    const demoNextName = document.getElementById("demoNextName");
    const demoPrevCard = document.getElementById("demoPrevCard");
    const demoNextCard = document.getElementById("demoNextCard");
    const demoStartBtn = document.getElementById("demoStartBtn");
    const demoSaveBtn = document.getElementById("demoSaveBtn");
    const demoHintBtn = document.getElementById("demoHintBtn");
    const demoNextBtn = document.getElementById("demoNextBtn");
    const demoExampleLabel = document.getElementById("demoExampleLabel");
    const demoElapsed = document.getElementById("demoElapsed");
    const demoAccuracy = document.getElementById("demoAccuracy");
    const demoSummary = document.getElementById("demoSummary");

    if (
        demoQuestionText &&
        demoHintPanel &&
        demoHintText &&
        demoStageCounter &&
        demoStageHint &&
        demoMainWrap &&
        demoMainImage &&
        demoMainName &&
        demoMainLock &&
        demoClickMarker &&
        demoGtMarker &&
        demoPrevImage &&
        demoPrevName &&
        demoNextImage &&
        demoNextName &&
        demoPrevCard &&
        demoNextCard &&
        demoStartBtn &&
        demoSaveBtn &&
        demoHintBtn &&
        demoNextBtn &&
        demoExampleLabel &&
        demoElapsed &&
        demoAccuracy &&
        demoSummary
    ) {
        const demoState = {
            sampleIndex: 0,
            frameIndex: 0,
            started: false,
            startTime: null,
            clickPoint: null,
            hintShown: false,
            locationShown: false,
            results: []
        };

        const clampIndex = (value, length) => {
            if (length <= 0) return 0;
            return ((value % length) + length) % length;
        };

        const formatSeconds = (seconds) => `${seconds.toFixed(1)} s`;

        const scoreClick = (sample, clickPoint) => {
            if (!clickPoint) {
                return { accuracy: 0, sameFrame: false, distance: null };
            }

            const gt = sample.groundTruth;
            const sameFrame = Number(clickPoint.frameIndex) === Number(gt.frameIndex);

            if (!sameFrame) {
                return { accuracy: 0, sameFrame: false, distance: null };
            }

            const dx = Number(clickPoint.x) - Number(gt.x);
            const dy = Number(clickPoint.y) - Number(gt.y);
            const distance = Math.sqrt(dx * dx + dy * dy);
            const normalized = Math.min(1, distance / DEMO_SCORING_RADIUS);
            const score = Math.max(0, 1 - normalized * normalized);

            return {
                accuracy: score * 100,
                sameFrame: true,
                distance
            };
        };

        const hideMarker = (marker) => {
            marker.classList.add("hidden");
        };

        const placeMarkerOnImage = (marker, x, y) => {
            const imageWidth = demoMainImage.clientWidth;
            const imageHeight = demoMainImage.clientHeight;
            const offsetLeft = demoMainImage.offsetLeft;
            const offsetTop = demoMainImage.offsetTop;

            if (!imageWidth || !imageHeight) {
                hideMarker(marker);
                return;
            }

            marker.style.left = `${offsetLeft + x * imageWidth}px`;
            marker.style.top = `${offsetTop + y * imageHeight}px`;
            marker.classList.remove("hidden");
        };

        const renderMarkers = () => {
            const sample = demoSamples[demoState.sampleIndex];

            if (demoState.clickPoint && demoState.clickPoint.frameIndex === demoState.frameIndex) {
                placeMarkerOnImage(demoClickMarker, demoState.clickPoint.x, demoState.clickPoint.y);
            } else {
                hideMarker(demoClickMarker);
            }

            if (demoState.locationShown && sample.groundTruth.frameIndex === demoState.frameIndex) {
                placeMarkerOnImage(demoGtMarker, sample.groundTruth.x, sample.groundTruth.y);
            } else {
                hideMarker(demoGtMarker);
            }
        };

        const renderFrame = () => {
            const sample = demoSamples[demoState.sampleIndex];
            const currentFrame = clampIndex(demoState.frameIndex, sample.frames.length);
            const prevFrame = clampIndex(currentFrame - 1, sample.frames.length);
            const nextFrame = clampIndex(currentFrame + 1, sample.frames.length);

            demoMainImage.src = sample.frames[currentFrame];
            demoMainName.textContent = `${currentFrame + 1}.png`;
            demoStageCounter.textContent = `Image ${currentFrame + 1} / ${sample.frames.length}`;

            demoPrevImage.src = sample.frames[prevFrame];
            demoPrevName.textContent = `${prevFrame + 1}.png`;

            demoNextImage.src = sample.frames[nextFrame];
            demoNextName.textContent = `${nextFrame + 1}.png`;
        };

        const renderSummary = () => {
            if (demoState.results.length === 0) {
                demoSummary.textContent = "Start the demo and complete an example to see the running summary.";
                return;
            }

            const avgAccuracy =
                demoState.results.reduce((sum, item) => sum + item.accuracy, 0) / demoState.results.length;
            const avgTime =
                demoState.results.reduce((sum, item) => sum + item.elapsed, 0) / demoState.results.length;

            demoSummary.textContent = `Completed ${demoState.results.length}/${demoSamples.length} examples | Avg accuracy ${avgAccuracy.toFixed(1)}% | Avg time ${formatSeconds(avgTime)}`;
        };

        const renderHintState = () => {
            const sample = demoSamples[demoState.sampleIndex];

            if (demoState.hintShown) {
                demoHintPanel.classList.remove("hidden");
                demoHintText.textContent = sample.groundTruthText;
            } else {
                demoHintPanel.classList.add("hidden");
                demoHintText.textContent = "";
            }

            if (!demoState.started) {
                demoHintBtn.disabled = true;
                demoHintBtn.textContent = "Show Hint";
                return;
            }

            demoHintBtn.disabled = false;

            if (!demoState.hintShown) {
                demoHintBtn.textContent = "Show Hint";
            } else if (!demoState.locationShown) {
                demoHintBtn.textContent = "Show Location";
            } else {
                demoHintBtn.textContent = "Location Shown";
                demoHintBtn.disabled = true;
            }
        };

        const renderSample = () => {
            const sample = demoSamples[demoState.sampleIndex];

            demoQuestionText.textContent = sample.question;
            demoExampleLabel.textContent = `${sample.title} (${demoState.sampleIndex + 1} / ${demoSamples.length})`;
            demoElapsed.textContent = "-";
            demoAccuracy.textContent = "-";
            demoStageHint.textContent = demoState.started
                ? sample.instruction
                : "Click Start to begin this example.";
            demoMainLock.classList.toggle("hidden", demoState.started);
            demoSaveBtn.disabled = !demoState.started || !demoState.clickPoint;

            renderHintState();
            renderFrame();
            renderSummary();
            renderMarkers();
        };

        const startSample = () => {
            demoState.started = true;
            demoState.startTime = performance.now();
            demoState.clickPoint = null;
            demoState.hintShown = false;
            demoState.locationShown = false;

            demoStageHint.textContent = demoSamples[demoState.sampleIndex].instruction;
            demoMainLock.classList.add("hidden");
            demoSaveBtn.disabled = true;
            renderHintState();
            renderMarkers();
        };

        const saveAnswer = () => {
            if (!demoState.started || !demoState.clickPoint || demoState.startTime === null) return;

            const sample = demoSamples[demoState.sampleIndex];
            const elapsed = (performance.now() - demoState.startTime) / 1000;
            const score = scoreClick(sample, demoState.clickPoint);
            const accuracyText = `${score.accuracy.toFixed(1)}%`;

            demoElapsed.textContent = formatSeconds(elapsed);
            demoAccuracy.textContent = accuracyText;

            const existingIndex = demoState.results.findIndex((item) => item.id === sample.id);
            const resultPayload = {
                id: sample.id,
                elapsed,
                accuracy: score.accuracy
            };

            if (existingIndex >= 0) {
                demoState.results[existingIndex] = resultPayload;
            } else {
                demoState.results.push(resultPayload);
            }

            demoStageHint.textContent = score.sameFrame
                ? `Answer recorded. Accuracy ${accuracyText}.`
                : "Answer recorded. Wrong frame selected.";

            demoState.started = false;
            demoState.startTime = null;
            demoSaveBtn.disabled = true;
            renderHintState();
            renderSummary();
        };

        const nextSample = () => {
            demoState.sampleIndex = (demoState.sampleIndex + 1) % demoSamples.length;
            demoState.frameIndex = 0;
            demoState.started = false;
            demoState.startTime = null;
            demoState.clickPoint = null;
            demoState.hintShown = false;
            demoState.locationShown = false;
            renderSample();
        };

        demoMainImage.addEventListener("click", (event) => {
            if (!demoState.started) return;

            const rect = demoMainImage.getBoundingClientRect();
            if (!rect.width || !rect.height) return;

            const x = (event.clientX - rect.left) / rect.width;
            const y = (event.clientY - rect.top) / rect.height;

            demoState.clickPoint = {
                frameIndex: demoState.frameIndex,
                x,
                y
            };

            demoSaveBtn.disabled = false;
            renderMarkers();
        });

        demoMainImage.addEventListener("load", () => {
            requestAnimationFrame(renderMarkers);
        });
        window.addEventListener("resize", renderMarkers);

        demoPrevCard.addEventListener("click", () => {
            if (!demoState.started) return;
            demoState.frameIndex = clampIndex(demoState.frameIndex - 1, demoSamples[demoState.sampleIndex].frames.length);
            renderFrame();
            requestAnimationFrame(renderMarkers);
        });

        demoNextCard.addEventListener("click", () => {
            if (!demoState.started) return;
            demoState.frameIndex = clampIndex(demoState.frameIndex + 1, demoSamples[demoState.sampleIndex].frames.length);
            renderFrame();
            requestAnimationFrame(renderMarkers);
        });

        demoStartBtn.addEventListener("click", startSample);

        demoSaveBtn.addEventListener("click", saveAnswer);

        demoHintBtn.addEventListener("click", () => {
            if (!demoState.started) return;

            if (!demoState.hintShown) {
                demoState.hintShown = true;
                renderHintState();
                return;
            }

            if (!demoState.locationShown) {
                demoState.locationShown = true;
                renderHintState();
                renderMarkers();
            }
        });

        demoNextBtn.addEventListener("click", nextSample);

        renderSample();
    }
});
