document.addEventListener('DOMContentLoaded', function() {
    // Canvas logic
    const canvas = document.getElementById('mnist-canvas');
    const ctx = canvas.getContext('2d');
    let drawing = false;
    let erasing = false;
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000';
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // --- Pointer and Mouse Event Support for Drawing ---
    // Remove old mouse event listeners
    // canvas.addEventListener('mousedown', ...);
    // canvas.addEventListener('mousemove', ...);
    // canvas.addEventListener('mouseup', ...);
    // canvas.addEventListener('mouseout', ...);

    canvas.addEventListener('pointerdown', handlePointerDown);
    canvas.addEventListener('pointermove', handlePointerMove);
    canvas.addEventListener('pointerup', handlePointerUp);
    canvas.addEventListener('pointerout', handlePointerUp); // treat out as up

    function handlePointerDown(e) {
        drawing = true;
        isDrawing = true;
        erasing = (e.button === 2);
        const rect = canvas.getBoundingClientRect();
        ctx.beginPath();
        ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        clearInterval(sendImageInterval);
        sendImageInterval = setInterval(sendImage, 200);
    }

    function handlePointerMove(e) {
        if (!drawing) return;
        const rect = canvas.getBoundingClientRect();
        ctx.strokeStyle = erasing ? '#fff' : '#000';
        ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    }

    function handlePointerUp(e) {
        drawing = false;
        erasing = false;
        ctx.beginPath();
        setTimeout(() => {
            isDrawing = false;
            clearInterval(sendImageInterval);
            sendImageInterval = setInterval(sendImage, 500);
        }, 1000);
    }

    // Prevent scrolling/gesture on touch
    canvas.style.touchAction = 'none';

    document.getElementById('clear-btn').onclick = () => {
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
    };

    // Keyboard shortcut: C to clear, R to reload
    document.addEventListener('keydown', e => {
        if (e.key === 'c' || e.key === 'C') {
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
        }
        if (e.key === 'r' || e.key === 'R') {
            location.reload();
        }
    });

    // WebSocket for real-time prediction
    let sock = new WebSocket('ws://' + window.location.host + '/ws');
    let sendImageInterval;
    let isDrawing = false;
    
    sock.onopen = () => { 
        // Send image less frequently when not drawing
        sendImageInterval = setInterval(sendImage, 500); 
    };

    function sendImage() {
        // Check if canvas is empty (all white)
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        let isEmpty = true;
        
        for (let i = 0; i < data.length; i += 4) {
            if (data[i] !== 255 || data[i + 1] !== 255 || data[i + 2] !== 255) {
                isEmpty = false;
                break;
            }
        }
        
        // Only send if canvas is not empty
        if (!isEmpty) {
            let dataURL = canvas.toDataURL('image/png');
            sock.send(dataURL);
        }
    }

    // Handle WebSocket responses
    sock.onmessage = function(event) {
        try {
            console.log('Received WebSocket data:', event.data);
            const data = JSON.parse(event.data);
            
            if (data.error) {
                console.error('Prediction error:', data.error);
                const predElem = document.getElementById('prediction-value');
                const confElem = document.getElementById('confidence-value');
                if (predElem) predElem.textContent = 'Error';
                if (confElem) confElem.textContent = 'Error';
                document.getElementById('probs').innerHTML = `<div style="color: red;">Error: ${data.error}</div>`;
                return;
            }
            
            const probs = data.probabilities;
            const prediction = data.prediction;
            const confidence = data.confidence;
            
            console.log('Parsed data:', { probs, prediction, confidence });
            
            // Update prediction and confidence displays
            const predElem = document.getElementById('prediction-value');
            const confElem = document.getElementById('confidence-value');
            if (predElem) predElem.textContent = prediction;
            if (confElem) confElem.textContent = (confidence * 100).toFixed(1) + '%';
            
            // Update probability display
            updateProbabilityDisplay(probs, prediction, confidence);
            
            // Update neural network visualization - only animate when prediction changes
            if (nnViewMode && prediction !== lastAnimatedIdx) {
                animateNNPath(prediction);
                document.getElementById('nn-result').textContent = prediction;
                lastAnimatedIdx = prediction;
            }
            
            // Update the old probability display for compatibility
            let maxIdx = 0;
            for (let i = 1; i < 10; i++) {
                if (probs[i] > probs[maxIdx]) maxIdx = i;
            }
            let html = '';
            for (let i = 0; i < 10; i++) {
                let rowClass = 'prob-row' + (i === maxIdx ? ' max' : '');
                html += `<div class='${rowClass}' id='prob-row-${i}'>`;
                html += `<span class='prob-digit'>${i}</span>`;
                html += `<span class='prob-value'>${(probs[i]*100).toFixed(2)}%</span>`;
                html += `</div>`;
            }
            document.getElementById('probs').innerHTML = html;
        } catch (e) {
            console.error('Error parsing WebSocket response:', e);
            const predElem = document.getElementById('prediction-value');
            const confElem = document.getElementById('confidence-value');
            if (predElem) predElem.textContent = 'Error';
            if (confElem) confElem.textContent = 'Error';
            document.getElementById('probs').innerHTML = `<div style="color: red;">Error: ${e.message}</div>`;
        }
    };

    function updateProbabilityDisplay(probs, prediction, confidence) {
        const probsDiv = document.getElementById('probs');
        let html = '';
        
        // Create array of [digit, probability] pairs and sort by probability
        const probPairs = probs.map((prob, digit) => [digit, prob]).sort((a, b) => b[1] - a[1]);
        
        probPairs.forEach(([digit, prob]) => {
            const isMax = digit === prediction;
            const probPercent = (prob * 100).toFixed(1);
            const confidenceClass = getConfidenceClass(prob);
            
            html += `<div class="prob-row ${isMax ? 'max' : ''}">
                <span class="prob-digit">${digit}</span>
                <span class="confidence-indicator ${confidenceClass}"></span>
                <span class="prob-value">${probPercent}%</span>
            </div>`;
        });
        
        probsDiv.innerHTML = html;
    }

    function getConfidenceClass(confidence) {
        if (confidence > 0.8) return 'confidence-high';
        if (confidence > 0.5) return 'confidence-medium';
        return 'confidence-low';
    }

    // --- Stylized Neural Network SVG ---
    function createNNGraph() {
        // l input nodes, 3 hidden layers (l2, l3, l4), 10 output nodes
        let w = 660, h = 420;
        let inputX = 60, l2X = 180, l3X = 320, l4X = 460, outputX = 600;
        let inputN = 10, l2N = 12, l3N = 12, l4N = 12, outputN = 10;
        let inputY = Array.from({length: inputN}, (_, i) => 40 + i * 34);
        let l2Y = Array.from({length: l2N}, (_, i) => 20 + i * 32);
        let l3Y = Array.from({length: l3N}, (_, i) => 20 + i * 32);
        let l4Y = Array.from({length: l4N}, (_, i) => 20 + i * 32);
        let outputY = Array.from({length: outputN}, (_, i) => 40 + i * 34);
        let svg = `<svg id='nn-svg' width='${w}' height='${h}' viewBox='0 0 ${w} ${h}' style='background:none'>`;
        // Lines input -> l2
        for (let i = 0; i < inputN; i++) for (let j = 0; j < l2N; j++) {
            svg += `<line class='nn-line' id='l-i-${i}-l2-${j}' x1='${inputX+12}' y1='${inputY[i]}' x2='${l2X-12}' y2='${l2Y[j]}' stroke='#bbb' stroke-width='2' opacity='0.5' />`;
        }
        // Lines l2 -> l3
        for (let i = 0; i < l2N; i++) for (let j = 0; j < l3N; j++) {
            svg += `<line class='nn-line' id='l-l2-${i}-l3-${j}' x1='${l2X+12}' y1='${l2Y[i]}' x2='${l3X-12}' y2='${l3Y[j]}' stroke='#bbb' stroke-width='2' opacity='0.5' />`;
        }
        // Lines l3 -> l4
        for (let i = 0; i < l3N; i++) for (let j = 0; j < l4N; j++) {
            svg += `<line class='nn-line' id='l-l3-${i}-l4-${j}' x1='${l3X+12}' y1='${l3Y[i]}' x2='${l4X-12}' y2='${l4Y[j]}' stroke='#bbb' stroke-width='2' opacity='0.5' />`;
        }
        // Lines l4 -> output
        for (let i = 0; i < l4N; i++) for (let j = 0; j < outputN; j++) {
            svg += `<line class='nn-line' id='l-l4-${i}-o-${j}' x1='${l4X+12}' y1='${l4Y[i]}' x2='${outputX-16}' y2='${outputY[j]}' stroke='#bbb' stroke-width='2' opacity='0.5' />`;
        }
        // Input nodes
        for (let i = 0; i < inputN; i++) {
            svg += `<circle class='nn-node' id='i-${i}' cx='${inputX}' cy='${inputY[i]}' r='10' fill='#222' stroke='#888' stroke-width='2' />`;
        }
        // l2 nodes
        for (let i = 0; i < l2N; i++) {
            svg += `<circle class='nn-node' id='l2-${i}' cx='${l2X}' cy='${l2Y[i]}' r='12' fill='#222' stroke='#888' stroke-width='2' />`;
        }
        // l3 nodes
        for (let i = 0; i < l3N; i++) {
            svg += `<circle class='nn-node' id='l3-${i}' cx='${l3X}' cy='${l3Y[i]}' r='12' fill='#222' stroke='#888' stroke-width='2' />`;
        }
        // l4 nodes
        for (let i = 0; i < l4N; i++) {
            svg += `<circle class='nn-node' id='l4-${i}' cx='${l4X}' cy='${l4Y[i]}' r='12' fill='#222' stroke='#888' stroke-width='2' />`;
        }
        // Output nodes
        for (let i = 0; i < outputN; i++) {
            svg += `<circle class='nn-node' id='o-${i}' cx='${outputX}' cy='${outputY[i]}' r='14' fill='#222' stroke='#888' stroke-width='2' />`;
            svg += `<text x='${outputX+28}' y='${outputY[i]+6}' font-size='18' fill='#fff' font-family='monospace'>${i}</text>`;
        }
        svg += '</svg>';
        return svg;
    }
    // Insert SVG on page load (before anything else)
    const nnSvgContainer = document.getElementById('nn-svg-container');
    nnSvgContainer.innerHTML = createNNGraph();
    // Add result display above SVG on page load
    if (!document.getElementById('nn-result')) {
        let resultDiv = document.createElement('div');
        resultDiv.id = 'nn-result';
        resultDiv.style = 'font-size: 2em; color: #ff0; text-align: center; margin-bottom: 8px; font-family: monospace;';
        resultDiv.textContent = '';
        nnSvgContainer.parentNode.insertBefore(resultDiv, nnSvgContainer);
    }

    // --- Toggle logic ---
    const toggleBtn = document.getElementById('toggle-view-btn');
    const probsContainer = document.getElementById('probs-container');
    let nnViewMode = false;
    toggleBtn.onclick = () => {
        nnViewMode = !nnViewMode;
        if (nnViewMode) {
            probsContainer.style.display = 'none';
            nnSvgContainer.style.display = '';
            toggleBtn.textContent = 'Show Probability View';
        } else {
            probsContainer.style.display = '';
            nnSvgContainer.style.display = 'none';
            toggleBtn.textContent = 'Show Neural Network View';
            // Clear the neural network result display
            const nnResult = document.getElementById('nn-result');
            if (nnResult) nnResult.textContent = '';
        }
    };

    // Add hotkey: 'v' to toggle view
    document.addEventListener('keydown', e => {
        if (e.key === 'v' || e.key === 'V') {
            toggleBtn.click();
        }
    });

    // --- Neural Network Visualization Utilities ---

    // Layer sizes
    const NN_LAYERS = [10, 12, 12, 12, 10]; // input, l2, l3, l4, output
    const NN_LAYER_IDS = ['i', 'l2', 'l3', 'l4', 'o'];

    // Helper to get node DOM element by layer and index
    function getNode(layer, idx) {
        return document.getElementById(`${layer}-${idx}`);
    }
    // Helper to get line DOM element by layer indices
    function getLine(fromLayer, fromIdx, toLayer, toIdx) {
        if (fromLayer === 'i' && toLayer === 'l2') return document.getElementById(`l-i-${fromIdx}-l2-${toIdx}`);
        if (fromLayer === 'l2' && toLayer === 'l3') return document.getElementById(`l-l2-${fromIdx}-l3-${toIdx}`);
        if (fromLayer === 'l3' && toLayer === 'l4') return document.getElementById(`l-l3-${fromIdx}-l4-${toIdx}`);
        if (fromLayer === 'l4' && toLayer === 'o') return document.getElementById(`l-l4-${fromIdx}-o-${toIdx}`);
        return null;
    }

    // Deterministic pseudo-random shuffle (Fisher-Yates) with seed
    function seededShuffle(arr, seed) {
        let a = arr.slice();
        let random = function() {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };
        for (let i = a.length - 1; i > 0; i--) {
            let j = Math.floor(random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    // Pick N unique, well-distributed nodes from a layer
    function pickNodes(total, n, seed) {
        let arr = Array.from({length: total}, (_, i) => i);
        let shuffled = seededShuffle(arr, seed);
        return shuffled.slice(0, n);
    }

    // --- Color Palette ---
    const COLOR_NODE_INACTIVE = '#2a2a3a';
    const COLOR_NODE_ACTIVE = '#7ec8e3'; // soft blue
    const COLOR_NODE_OUTPUT = '#ffeb3b'; // yellow
    const COLOR_LINE_INACTIVE = '#44445a';
    const COLOR_LINE_ACTIVE = '#7ec8e3';
    const COLOR_LINE_OUTPUT = '#a084e8';
    const OPACITY_INACTIVE = '0.20';
    const OPACITY_ACTIVE = '0.45';
    const OPACITY_OUTPUT = '0.85';

    // --- Main Animation Function ---
    let nnAnimationTimeouts = [];
    function clearNNAnimationTimeouts() {
        nnAnimationTimeouts.forEach(timeout => clearTimeout(timeout));
        nnAnimationTimeouts = [];
    }
    function animateNNPath(outputIdx) {
        clearNNAnimationTimeouts();
        // 1. Dim all nodes and lines
        document.querySelectorAll('.nn-node').forEach(e => e.setAttribute('fill', COLOR_NODE_INACTIVE));
        document.querySelectorAll('.nn-line').forEach(e => {
            e.setAttribute('stroke', COLOR_LINE_INACTIVE);
            e.setAttribute('opacity', OPACITY_INACTIVE);
        });

        // 2. Pick nodes for each layer (deterministic, well-distributed)
        const inputNodes = pickNodes(10, 3, outputIdx * 101 + 1);
        const l2Nodes   = pickNodes(12, 3, outputIdx * 211 + 2);
        const l3Nodes   = pickNodes(12, 3, outputIdx * 307 + 3);
        const l4Nodes   = pickNodes(12, 3, outputIdx * 419 + 4);
        const outNode   = outputIdx;

        // 3. Animate left-to-right, only fire from activated nodes
        let activated = { i: inputNodes, l2: [], l3: [], l4: [], o: [outNode] };

        // Step 1: Highlight input nodes
        nnAnimationTimeouts.push(setTimeout(() => {
            activated.i.forEach(iIdx => {
                let node = getNode('i', iIdx);
                if (node) node.setAttribute('fill', COLOR_NODE_ACTIVE);
            });
        }, 100));
        // Step 2: input -> l2
        nnAnimationTimeouts.push(setTimeout(() => {
            activated.l2 = [];
            l2Nodes.forEach(l2Idx => {
                let fired = false;
                activated.i.forEach(iIdx => {
                    let line = getLine('i', iIdx, 'l2', l2Idx);
                    if (line) {
                        line.setAttribute('stroke', COLOR_LINE_ACTIVE);
                        line.setAttribute('opacity', OPACITY_ACTIVE);
                        fired = true;
                    }
                });
                if (fired) {
                    let node = getNode('l2', l2Idx);
                    if (node) node.setAttribute('fill', COLOR_NODE_ACTIVE);
                    activated.l2.push(l2Idx);
                }
            });
        }, 300));
        // Step 3: l2 -> l3
        nnAnimationTimeouts.push(setTimeout(() => {
            activated.l3 = [];
            l3Nodes.forEach(l3Idx => {
                let fired = false;
                activated.l2.forEach(l2Idx => {
                    let line = getLine('l2', l2Idx, 'l3', l3Idx);
                    if (line) {
                        line.setAttribute('stroke', COLOR_LINE_ACTIVE);
                        line.setAttribute('opacity', OPACITY_ACTIVE);
                        fired = true;
                    }
                });
                if (fired) {
                    let node = getNode('l3', l3Idx);
                    if (node) node.setAttribute('fill', COLOR_NODE_ACTIVE);
                    activated.l3.push(l3Idx);
                }
            });
        }, 500));
        // Step 4: l3 -> l4
        nnAnimationTimeouts.push(setTimeout(() => {
            activated.l4 = [];
            l4Nodes.forEach(l4Idx => {
                let fired = false;
                activated.l3.forEach(l3Idx => {
                    let line = getLine('l3', l3Idx, 'l4', l4Idx);
                    if (line) {
                        line.setAttribute('stroke', COLOR_LINE_ACTIVE);
                        line.setAttribute('opacity', OPACITY_ACTIVE);
                        fired = true;
                    }
                });
                if (fired) {
                    let node = getNode('l4', l4Idx);
                    if (node) node.setAttribute('fill', COLOR_NODE_ACTIVE);
                    activated.l4.push(l4Idx);
                }
            });
        }, 700));
        // Step 5: l4 -> output
        nnAnimationTimeouts.push(setTimeout(() => {
            activated.l4.forEach(l4Idx => {
                let line = getLine('l4', l4Idx, 'o', outNode);
                if (line) {
                    line.setAttribute('stroke', COLOR_LINE_OUTPUT);
                    line.setAttribute('opacity', OPACITY_OUTPUT);
                }
            });
            let node = getNode('o', outNode);
            if (node) node.setAttribute('fill', COLOR_NODE_OUTPUT);
        }, 900));
    }

    // --- Update both views on prediction ---
    const probsElem = document.getElementById('probs');
    let lastAnimatedIdx = null;

    // Removed theme toggle logic and setTheme function
    // Optionally, remember theme
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // setTheme(true); // This line is removed
    }
}); 