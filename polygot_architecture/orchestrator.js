const express = require('express');
const axios = require('axios');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
app.use(express.json());
app.use(cors());

// Health check
app.get('/', (req, res) => {
    res.json({ status: 'Node.js Orchestrator Running!' });
});

// Function to call R service
// Function to call R service
function callRService(ticker) {
    return new Promise((resolve, reject) => {
        // Call R with ticker as command line argument
        const rProcess = spawn('Rscript', [
            '../r-service/stock_visualizer.R',
            ticker
        ]);
        
        let outputData = '';
        let errorData = '';
        
        // Collect output
        rProcess.stdout.on('data', (data) => {
            outputData += data.toString();
        });
        
        rProcess.stderr.on('data', (data) => {
            errorData += data.toString();
            // Only log important R messages
            const msg = data.toString().trim();
            if (msg && !msg.includes('package') && !msg.includes('attached')) {
                console.log('R log:', msg);
            }
        });
        
        rProcess.on('close', (code) => {
            if (code !== 0 && !outputData) {
                console.error('R stderr:', errorData);
                reject(new Error(`R process exited with code ${code}`));
            } else {
                try {
                    // Clean the output - remove any extra whitespace/newlines
                    const cleanOutput = outputData.trim();
                    const result = JSON.parse(cleanOutput);
                    resolve(result);
                } catch (e) {
                    console.error('R output:', outputData);
                    console.error('Parse error:', e.message);
                    reject(new Error('Failed to parse R output: ' + e.message));
                }
            }
        });
        
        setTimeout(() => {
            rProcess.kill();
            reject(new Error('R service timeout (30s)'));
        }, 30000);
    });
}

// Function to call Java service
function callJavaService(ticker, pythonData) {
    return new Promise((resolve, reject) => {
        const javaProcess = spawn('java', [
            '-cp',
            '../java-service;../java-service/gson-2.10.1.jar',
            'StockRiskAnalyzer'
        ]);
        
        let outputData = '';
        let errorData = '';
        
        // Prepare input for Java
        const javaInput = {
            ticker: ticker,
            volatility: pythonData.volatility,
            confidence: pythonData.confidence,
            signal: pythonData.signal
        };
        
        // Send input to Java
        javaProcess.stdin.write(JSON.stringify(javaInput));
        javaProcess.stdin.end();
        
        // Collect output
        javaProcess.stdout.on('data', (data) => {
            outputData += data.toString();
        });
        
        javaProcess.stderr.on('data', (data) => {
            errorData += data.toString();
            const msg = data.toString().trim();
            if (msg.startsWith('Java:')) {
                console.log(msg);
            }
        });
        
        javaProcess.on('close', (code) => {
            if (code !== 0 && !outputData) {
                console.error('Java stderr:', errorData);
                reject(new Error(`Java process exited with code ${code}`));
            } else {
                try {
                    const cleanOutput = outputData.trim();
                    const result = JSON.parse(cleanOutput);
                    resolve(result);
                } catch (e) {
                    console.error('Java output:', outputData);
                    reject(new Error('Failed to parse Java output: ' + e.message));
                }
            }
        });
        
        setTimeout(() => {
            javaProcess.kill();
            reject(new Error('Java service timeout'));
        }, 10000);
    });
}

// Function to call C++ service
function callCppService(ticker, pythonData) {
    return new Promise((resolve, reject) => {
        const cppProcess = spawn('../cpp-service/portfolio_calculator.exe');
        
        let outputData = '';
        let errorData = '';
        
        // Prepare input for C++
        const cppInput = {
            ticker: ticker,
            current_price: pythonData.current_price,
            volatility: pythonData.volatility,
            confidence: pythonData.confidence
        };
        
        // Send input to C++
        cppProcess.stdin.write(JSON.stringify(cppInput));
        cppProcess.stdin.end();
        
        // Collect output
        cppProcess.stdout.on('data', (data) => {
            outputData += data.toString();
        });
        
        cppProcess.stderr.on('data', (data) => {
            errorData += data.toString();
            const msg = data.toString().trim();
            if (msg.startsWith('C++:')) {
                console.log(msg);
            }
        });
        
        cppProcess.on('close', (code) => {
            if (code !== 0 && !outputData) {
                console.error('C++ stderr:', errorData);
                reject(new Error(`C++ process exited with code ${code}`));
            } else {
                try {
                    const cleanOutput = outputData.trim();
                    const result = JSON.parse(cleanOutput);
                    resolve(result);
                } catch (e) {
                    console.error('C++ output:', outputData);
                    reject(new Error('Failed to parse C++ output: ' + e.message));
                }
            }
        });
        
        setTimeout(() => {
            cppProcess.kill();
            reject(new Error('C++ service timeout'));
        }, 5000);
    });
}
// Main endpoint that coordinates all services
app.post('/predict', async (req, res) => {
    const { ticker } = req.body;
    
    console.log(`\n🎯 Node: Received request for ${ticker}`);
    console.log('━'.repeat(50));
    
    try {
        const results = {};
        
        // Step 1: Call Python service
        console.log('📞 Node: Calling Python service...');
        const pythonResponse = await axios.post('http://localhost:5001/analyze', { ticker });
        results.python = pythonResponse.data;
        console.log('✅ Node: Python response received');
        
        // Step 2: Call R service
        console.log('📞 Node: Calling R service...');
        try {
            results.r = await callRService(ticker);
            console.log('✅ Node: R response received');
        } catch (error) {
            console.log('⚠️ Node: R service error:', error.message);
            results.r = { status: 'error', message: error.message };
        }
        
        // Step 3: Call Java service
console.log('📞 Node: Calling Java service...');
try {
    results.java = await callJavaService(ticker, results.python);
    console.log('✅ Node: Java response received');
} catch (error) {
    console.log('⚠️ Node: Java service error:', error.message);
    results.java = { status: 'error', message: error.message };
}
        
        // Step 4: Call C++ service
console.log('📞 Node: Calling C++ service...');
try {
    results.cpp = await callCppService(ticker, results.python);
    console.log('✅ Node: C++ response received');
} catch (error) {
    console.log('⚠️ Node: C++ service error:', error.message);
    results.cpp = { status: 'error', message: error.message };
}

        // Combine results
        const finalResult = {
            ticker: ticker,
            timestamp: new Date().toISOString(),
            prediction: results.python.signal,
            confidence: results.python.confidence,
            services: results
        };
        
        console.log('✅ Node: All services called, sending response');
        console.log('━'.repeat(50));
        
        res.json(finalResult);
        
    } catch (error) {
        console.error('❌ Node: Error:', error.message);
        res.status(500).json({ 
            error: 'Service communication failed',
            details: error.message 
        });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`\n🚀 Node.js Orchestrator running on http://localhost:${PORT}`);
    console.log(`📡 Ready to coordinate multi-language services!\n`);
});
