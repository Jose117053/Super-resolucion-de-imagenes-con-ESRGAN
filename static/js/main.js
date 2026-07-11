const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const controls = document.getElementById('controls');
const upscaleBtn = document.getElementById('upscale-btn');
const loader = document.getElementById('loader');
const resultsPanel = document.getElementById('results-panel');

const originalImg = document.getElementById('original-img');
const enhancedImg = document.getElementById('enhanced-img');

let currentFile = null;

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

        const reader = new FileReader();
        reader.readAsDataURL(currentFile);
        reader.onloadend = function () {
            originalImg.src = reader.result;
            originalImg.style.display = 'block';

            enhancedImg.style.display = 'none';
            resultsPanel.style.display = 'grid';

            dropZone.querySelector('.upload-text').textContent = currentFile.name;
            dropZone.querySelector('.upload-subtext').textContent = (currentFile.size / 1024 / 1024).toFixed(2) + ' MB';

            controls.style.display = 'block';
        }
    }
}

upscaleBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    upscaleBtn.disabled = true;
    upscaleBtn.textContent = 'Processing...';
    loader.style.display = 'block';
    enhancedImg.style.display = 'none';

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
        const url = URL.createObjectURL(blob);

        enhancedImg.src = url;
        enhancedImg.style.display = 'block';

    } catch (error) {
        console.error(error);
        alert('Error: ' + error.message);
    } finally {
        upscaleBtn.disabled = false;
        upscaleBtn.textContent = 'Enhance Image 🚀';
        loader.style.display = 'none';
    }
});
