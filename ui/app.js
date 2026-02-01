const { ipcRenderer } = require('electron');

// State
let isExpanded = false;
let isPinned = false;
let lastQuestion = null;
let lastAnswer = null;
let isGreenState = false; // Start White

// Elements
const app = document.getElementById('app');
const dbSelector = document.getElementById('dbSelector');
const expandBtn = document.getElementById('expandBtn');
const pinBtn = document.getElementById('pinBtn');
const closeBtn = document.getElementById('closeBtn');

// Minimal view elements
const choiceLetter = document.getElementById('choiceLetter');
const answerText = document.getElementById('answerText');

// Expanded view elements
const questionText = document.getElementById('questionText');
const answerA = document.getElementById('answerA');
const answerB = document.getElementById('answerB');
const answerC = document.getElementById('answerC');
const answerD = document.getElementById('answerD');
const matchScore = document.getElementById('matchScore');

// Listen for mock data from main process (for testing) - MUST BE BEFORE OTHER CODE
ipcRenderer.on('ocr-result', (event, data) => {
    console.log('Received data:', data);
    updateDisplay(data);
});

// WebSocket connection to Python backend
let ws = null;

function connectWebSocket() {
    ws = new WebSocket('ws://localhost:8765');

    ws.onopen = () => {
        console.log('Connected to backend');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'config_update') {
                if (data.active_database && dbSelector) {
                    console.log('Syncing DB state from backend:', data.active_database);
                    let dbVal = data.active_database.toLowerCase();
                    if (dbVal === 'default') dbVal = 'all'; // Fallback
                    dbSelector.value = dbVal;

                    // If it's still blank (invalid value), default to all
                    if (!dbSelector.value) {
                        dbSelector.value = 'all';
                    }
                }

                // Update Icon States
                if (data.auto_click !== undefined && autoClickBtn) {
                    if (data.auto_click) autoClickBtn.classList.add('active');
                    else autoClickBtn.classList.remove('active');
                }

                if (data.auto_scan !== undefined && autoScanBtn) {
                    if (data.auto_scan) autoScanBtn.classList.add('active');
                    else autoScanBtn.classList.remove('active');
                }
            } else {
                updateDisplay(data);
            }
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log('Disconnected from backend, reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };
}

function updateDisplay(data) {
    if (data.matched_choice) {
        // Change Detection
        const currentQuestion = data.question;
        const currentAnswer = data.matched_answer;

        // Toggle Logic:
        // New Question -> Flip Color (White <-> Green)
        // Same Question -> Keep Color
        if (lastQuestion && currentQuestion !== lastQuestion) {
            isGreenState = !isGreenState;
        } else if (!lastQuestion) {
            // First run: Force White (or user preference)
            isGreenState = false;
        }

        answerText.classList.remove('new-match');
        choiceLetter.classList.remove('new-match');
        if (isGreenState) {
            answerText.classList.add('new-match');
            choiceLetter.classList.add('new-match');
        }

        lastQuestion = currentQuestion;
        lastAnswer = currentAnswer;

        choiceLetter.textContent = data.matched_choice;
        answerText.textContent = data.matched_answer || 'No match';
    }

    // Update expanded view
    if (data.question) {
        questionText.textContent = data.question;
    }

    if (data.answers) {
        answerA.textContent = `A: ${data.answers.A || '-'}`;
        answerB.textContent = `B: ${data.answers.B || '-'}`;
        answerC.textContent = `C: ${data.answers.C || '-'}`;
        answerD.textContent = `D: ${data.answers.D || '-'}`;

        // Highlight matched answer
        [answerA, answerB, answerC, answerD].forEach(el => {
            el.classList.remove('matched');
        });

        if (data.matched_choice) {
            const matchedEl = document.getElementById(`answer${data.matched_choice}`);
            if (matchedEl) {
                matchedEl.classList.add('matched');
            }
        }
    }

    if (data.score !== undefined) {
        matchScore.textContent = `Score: ${(data.score * 100).toFixed(1)}%`;
    }

    // Auto-resize window to fit content
    setTimeout(() => {
        const bodyStyle = window.getComputedStyle(document.body);

        let newHeight = 0;
        let newWidth = 0;

        // If expanded, use fixed width but dynamic height
        if (app.classList.contains('expanded')) {
            newHeight = app.scrollHeight + 20; // Add padding
            newWidth = 500;
        } else {
            // For minimal, fit both width and height to content
            const content = document.querySelector('.minimal .content');
            const header = document.querySelector('.header');

            if (content && header) {
                const answerDisplay = document.querySelector('.answer-display');

                // Calculate required width with ample padding
                // FIX: flex:1 causes scrollWidth to match window width (ratchet effect).
                // Measure intrinsic content width instead.
                const measureSpan = document.createElement('span');
                measureSpan.style.visibility = 'hidden';
                measureSpan.style.position = 'absolute'; // Critical: Remove from flow
                measureSpan.style.whiteSpace = 'nowrap';
                measureSpan.style.left = '-9999px';

                // Copy styles accurately
                const computed = window.getComputedStyle(document.querySelector('.answer-display'));
                measureSpan.style.fontFamily = computed.fontFamily;
                measureSpan.style.fontSize = computed.fontSize;
                measureSpan.style.fontWeight = computed.fontWeight;
                measureSpan.style.letterSpacing = computed.letterSpacing;

                // Content: "A | Answer text"
                const currentChoice = document.getElementById('choiceLetter').textContent;
                const currentAnswer = document.getElementById('answerText').textContent;
                measureSpan.textContent = `${currentChoice} | ${currentAnswer}`;

                document.body.appendChild(measureSpan);
                const textWidth = measureSpan.offsetWidth;
                document.body.removeChild(measureSpan);

                // Add padding for header (icons) + container padding + safety
                // Header icons take up ~150px. Text is separate.
                newWidth = Math.max(380, textWidth + 100);

                // DEBUG LOGGING
                // require('fs').appendFileSync('ui_debug_log.txt', `[Renderer] Content: "${currentChoice} | ${currentAnswer}" | TextWidth: ${textWidth} | CalcWidth: ${newWidth}\n`);
                document.body.removeChild(measureSpan);

                // Add padding for header (icons) + container padding + safety
                // Header icons take up ~150px. Text is separate.
                newWidth = Math.max(380, textWidth + 100);

                // Calculate height: Header + Content + padding
                // Add 10px extra to strictly avoid vertical scrollbar usage
                newHeight = header.offsetHeight + content.offsetHeight + 10;
            }
        }

        // Send resize request
        ipcRenderer.send('resize-window', {
            width: Math.ceil(newWidth),
            height: Math.ceil(newHeight)
        });
    }, 50);
}

// Toggle expand/collapse
expandBtn.addEventListener('click', () => {
    isExpanded = !isExpanded;

    if (isExpanded) {
        app.classList.add('expanded');
        expandBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M4 14h6v6M20 10h-6V4M14 10l7-7M3 21l7-7"/>
      </svg>
    `;
        expandBtn.title = 'Collapse';
        // Trigger resize after transition
        setTimeout(() => updateDisplay({}), 200);
    } else {
        app.classList.remove('expanded');
        expandBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
      </svg>
    `;
        expandBtn.title = 'Expand';
    }
});

// Toggle always on top
pinBtn.addEventListener('click', () => {
    ipcRenderer.send('toggle-always-on-top');
});

ipcRenderer.on('always-on-top-changed', (event, isOnTop) => {
    isPinned = isOnTop;
    if (isPinned) {
        pinBtn.classList.add('active');
        pinBtn.title = 'Unpin';
    } else {
        pinBtn.classList.remove('active');
        pinBtn.title = 'Pin on top';
    }
});

// Database switcher
dbSelector.addEventListener('change', () => {
    const selectedDb = dbSelector.value;
    console.log(`Switching database to: ${selectedDb}`);
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'switch_database',
            database: selectedDb
        }));
    } else {
        console.error("WebSocket not connected - cannot switch database");
    }
});

// Tool Icons Logic
const posBtn = document.getElementById('posBtn');
const autoClickBtn = document.getElementById('autoClickBtn');
const autoScanBtn = document.getElementById('autoScanBtn');

if (posBtn) {
    posBtn.addEventListener('click', () => {
        if (ws) ws.send(JSON.stringify({ type: 'launch_config' }));
    });
}

if (autoClickBtn) {
    autoClickBtn.addEventListener('click', () => {
        if (ws) ws.send(JSON.stringify({ type: 'toggle_autoclick' }));
    });
}

if (autoScanBtn) {
    autoScanBtn.addEventListener('click', () => {
        if (ws) ws.send(JSON.stringify({ type: 'toggle_autoscan' }));
    });
}

// Close window
closeBtn.addEventListener('click', () => {
    ipcRenderer.send('close');
});

// Connect to backend on startup
connectWebSocket();

// Request mock data (Disabled for production)
// setTimeout(() => {
//     ipcRenderer.send('request-mock-data');
// }, 3000);
