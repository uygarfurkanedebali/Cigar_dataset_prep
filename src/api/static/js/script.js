let crops = [];
let currentIndex = 0;

const mainImage = document.getElementById('main-image');
const sourceName = document.getElementById('source-name');
const stats = document.getElementById('stats');
const currentLabel = document.getElementById('current-label');
const loader = document.getElementById('loader');
const viewer = document.getElementById('viewer');

async function fetchCrops() {
    try {
        const response = await fetch('/api/crops');
        crops = await response.json();
        if (crops.length > 0) {
            showImage(0);
            loader.classList.add('hidden');
            viewer.classList.remove('hidden');
        } else {
            loader.innerHTML = '<p>No person detections found.</p>';
        }
    } catch (error) {
        console.error('Failed to fetch crops:', error);
        loader.innerHTML = '<p>Error loading dataset.</p>';
    }
}

function showImage(index) {
    if (index < 0 || index >= crops.length) return;
    
    currentIndex = index;
    const crop = crops[currentIndex];
    
    mainImage.src = crop.url;
    sourceName.textContent = crop.source_image;
    stats.textContent = `${currentIndex + 1} / ${crops.length}`;
    currentLabel.textContent = crop.label || 'Not Labeled';
}

async function labelImage(label) {
    if (crops.length === 0) return;
    
    const crop = crops[currentIndex];
    crop.label = label;
    currentLabel.textContent = label;
    
    // Visual feedback
    currentLabel.style.transform = 'scale(1.2)';
    setTimeout(() => currentLabel.style.transform = 'scale(1)', 200);

    try {
        await fetch(`/api/label?crop_id=${crop.id}&label=${label}`, { method: 'POST' });
    } catch (error) {
        console.error('Failed to label:', error);
    }
}

document.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    
    if (key === 'd') { // Next
        if (currentIndex < crops.length - 1) {
            showImage(currentIndex + 1);
        }
    } else if (key === 'a') { // Prev
        if (currentIndex > 0) {
            showImage(currentIndex - 1);
        }
    } else if (['1', '2', '3', '4'].includes(key)) {
        labelImage(key);
    }
});

// Initialize
fetchCrops();
