const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 400,
    height: 100,
    minWidth: 380,
    minHeight: 50,
    maxWidth: 1600,
    maxHeight: 600,
    frame: false,
    transparent: true,
    alwaysOnTop: false,
    resizable: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  mainWindow.loadFile('ui/index.html');

  // AGGRESSIVE FIX: Force max constraints
  mainWindow.setMinimumSize(380, 50);
  mainWindow.setMaximumSize(1600, 600);
  console.log("Main Process v5.6: Window Constraints Applied (Max 1600px)");

  // Enable DevTools for debugging (disabled for production)
  // mainWindow.webContents.openDevTools();

  // Send mock data after UI loads (for testing)
  mainWindow.webContents.on('did-finish-load', () => {
    // console.log('UI loaded, waiting for backend data...');
    // setTimeout(() => sendMockData(), 1000);
  });
}

function sendMockData() {
  // This simulates what the Python backend would send
  const mockData = {
    matched_choice: 'B',
    matched_answer: 'Cat',
    question: 'What animal says meow?',
    answers: {
      A: 'Dog',
      B: 'Cat',
      C: 'Bird',
      D: 'Fish'
    },
    score: 0.95
  };

  console.log('Sending mock data:', mockData);
  mainWindow.webContents.send('ocr-result', mockData);
  console.log('Mock data sent via IPC');
}

// Handle always-on-top toggle
// Handle always-on-top toggle
ipcMain.on('toggle-always-on-top', (event) => {
  const isAlwaysOnTop = mainWindow.isAlwaysOnTop();
  const newState = !isAlwaysOnTop;
  // 'screen-saver' level is needed to be on top of some fullscreen apps
  mainWindow.setAlwaysOnTop(newState, 'screen-saver');
  event.reply('always-on-top-changed', newState);
  console.log(`Always on top set to: ${newState}`);
});

// Handle minimize
ipcMain.on('minimize', () => {
  mainWindow.minimize();
});

// Handle close
ipcMain.on('close', () => {
  mainWindow.close();
});

// Handle mock data request (for testing)
ipcMain.on('request-mock-data', () => {
  console.log('Mock data requested');
  sendMockData();
});

// Handle auto-resize
ipcMain.on('resize-window', (event, { width, height }) => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    const currentSize = mainWindow.getSize();
    // Only resize if significantly different to avoid jitter, but allow height changes
    const widthDiff = Math.abs(currentSize[0] - width);
    const heightDiff = Math.abs(currentSize[1] - height);

    // DEBUG LOGGING
    // require('fs').appendFileSync('ui_debug_log.txt', `[Main] Resize Request: ${width}x${height} | Current: ${currentSize[0]}x${currentSize[1]} | Diff: ${widthDiff}x${heightDiff}\n`);

    if (heightDiff > 2 || widthDiff > 2) {
      const { screen } = require('electron');
      const winBounds = mainWindow.getBounds();
      const display = screen.getDisplayMatching(winBounds);

      let newX = winBounds.x;

      // Check if expanding to the right hits the screen edge
      // If (currentX + newWidth) > (workAreaX + workAreaWidth)
      const rightEdge = winBounds.x + width;
      const screenRightEdge = display.workArea.x + display.workArea.width;

      if (rightEdge > screenRightEdge) {
        // Verify if shifting left fits on screen?
        // Shift left by the overflow amount
        const overflow = rightEdge - screenRightEdge;
        newX = winBounds.x - overflow;

        // Ensure we don't shift off the left edge either
        if (newX < display.workArea.x) {
          newX = display.workArea.x;
        }
      }

      if (newX !== winBounds.x) {
        // Use setBounds to move and resize simultaneously
        mainWindow.setBounds({ x: newX, y: winBounds.y, width: width, height: height });
        // require('fs').appendFileSync('ui_debug_log.txt', `[Main] Applied Smart Shift: Moved X to ${newX}, Size ${width}x${height}\n`);
      } else {
        mainWindow.setSize(width, height);
        // require('fs').appendFileSync('ui_debug_log.txt', `[Main] Applied Size: ${width}x${height}\n`);
      }
    } else {
      // require('fs').appendFileSync('ui_debug_log.txt', `[Main] Skipped Resize (Diff too small)\n`);
    }
  }
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
