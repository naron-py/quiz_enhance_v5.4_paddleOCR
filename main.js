const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 400,
    height: 100,
    minWidth: 150,
    minHeight: 50,
    maxWidth: 800,
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
ipcMain.on('toggle-always-on-top', (event) => {
  const isAlwaysOnTop = mainWindow.isAlwaysOnTop();
  mainWindow.setAlwaysOnTop(!isAlwaysOnTop);
  event.reply('always-on-top-changed', !isAlwaysOnTop);
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
    if (Math.abs(currentSize[1] - height) > 2 || Math.abs(currentSize[0] - width) > 2) {
      mainWindow.setSize(width, height);
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
