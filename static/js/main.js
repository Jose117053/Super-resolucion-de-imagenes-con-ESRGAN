const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const controls = document.getElementById('controls');
const upscaleBtn = document.getElementById('upscale-btn');
const loader = document.getElementById('loader');
const resultsPanel = document.getElementById('results-panel');

const originalImg = document.getElementById('original-img');
const enhancedImg = document.getElementById('enhanced-img');
const downloadBtn = document.getElementById('download-btn');

const comparisonContainer = document.getElementById('comparison-container');
const comparisonOverlay = document.getElementById('comparison-overlay');
const comparisonSlider = document.getElementById('comparison-slider');

let currentFile = null;
let isDragging = false;

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
fileInput.addEventListener('change', handleFileSelect, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        currentFile = files[0];

        resultsPanel.style.display = 'none';
        comparisonContainer.style.display = 'none';
        downloadBtn.style.display = 'none';

        dropZone.querySelector('.upload-text').textContent = currentFile.name;
        dropZone.querySelector('.upload-subtext').textContent = (currentFile.size / 1024 / 1024).toFixed(2) + ' MB';

        controls.style.display = 'block';
    }
}

upscaleBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    upscaleBtn.disabled = true;
    upscaleBtn.textContent = 'Processing...';
    loader.style.display = 'block';
    downloadBtn.style.display = 'none';
    comparisonContainer.style.display = 'none';

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/upscale', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Server error during upscaling');
        }

        const blob = await response.blob();
        const enhancedUrl = URL.createObjectURL(blob);

        const originalUrl = URL.createObjectURL(currentFile);

        enhancedImg.src = enhancedUrl;
        originalImg.src = originalUrl;

        enhancedImg.onload = function () {
            resultsPanel.style.display = 'block';
            comparisonContainer.style.display = 'block';

            function syncDimensions() {
                const rect = enhancedImg.getBoundingClientRect();
                const overlayImg = comparisonOverlay.querySelector('img');
                overlayImg.style.width = rect.width + 'px';
                overlayImg.style.height = rect.height + 'px';
            }

            syncDimensions();
            window.addEventListener('resize', syncDimensions);

            setSliderPosition(50);

            downloadBtn.href = enhancedUrl;
            downloadBtn.download = 'upscaled_' + currentFile.name;
            downloadBtn.style.display = 'inline-block';
        };

    } catch (error) {
        console.error(error);
        alert('Error: ' + error.message);
    } finally {
        upscaleBtn.disabled = false;
        upscaleBtn.textContent = 'Enhance Image 🚀';
        loader.style.display = 'none';
    }
});

function setSliderPosition(percent) {
    percent = Math.max(0, Math.min(100, percent));
    comparisonOverlay.style.width = percent + '%';
    comparisonSlider.style.left = percent + '%';
}

function getSliderPercent(e) {
    const rect = comparisonContainer.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    return ((clientX - rect.left) / rect.width) * 100;
}

comparisonContainer.addEventListener('mousedown', (e) => {
    isDragging = true;
    setSliderPosition(getSliderPercent(e));
});

comparisonContainer.addEventListener('touchstart', (e) => {
    isDragging = true;
    setSliderPosition(getSliderPercent(e));
}, { passive: true });

document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    setSliderPosition(getSliderPercent(e));
});

document.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    setSliderPosition(getSliderPercent(e));
}, { passive: true });

document.addEventListener('mouseup', () => { isDragging = false; });
document.addEventListener('touchend', () => { isDragging = false; });
