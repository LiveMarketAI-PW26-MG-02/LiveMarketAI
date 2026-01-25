/**
 * Advanced Stock Analyzer - State Management
 * Centralized state management with observers and persistence
 */

// ========================================
// STATE STRUCTURE
// ========================================

/**
 * Global Application State
 */
const GlobalState = {
  // Current Analysis
  currentSymbol: '',
  lastUpdated: null,
  isAnalyzing: false,
  
  // Stock Data
  stockData: {
    raw: [],
    processed: [],
    metadata: {
      symbol: '',
      company: '',
      lastRefreshed: null,
      dataPoints: 0
    }
  },
  
  // Technical Indicators
  indicators: {
    sma20: [],
    sma50: [],
    sma200: [],
    ema12: [],
    ema26: [],
    rsi: [],
    macd: [],
    bollingerBands: [],
    stochastic: []
  },
  
  // Risk Metrics
  riskMetrics: {
    volatility: 0,
    annualizedVolatility: 0,
    maxDrawdown: 0,
    sharpeRatio: 0,
    beta: 1.0,
    var95: 0, // Value at Risk 95%
    cvar95: 0 // Conditional Value at Risk
  },
  
  // Market Data
  marketData: {
    news: [],
    videos: [],
    sentiment: {
      overall: 'neutral',
      score: 0,
      articles: 0
    }
  },
  
  // Trading Signals
  signals: {
    current: {
      type: 'hold',
      strength: 0,
      message: '',
      timestamp: null
    },
    history: []
  },
  
  // Chart Instances
  charts: {
    price: null,
    volume: null,
    rsi: null,
    macd: null
  },
  
  // UI State
  ui: {
    activeTab: 'overview',
    theme: 'dark',
    chartTimeframe: '3M',
    showVolume: true,
    showIndicators: true
  },
  
  // API State
  api: {
    lastCall: {
      alphaVantage: 0,
      newsAPI: 0,
      youtubeAPI: 0
    },
    callCount: {
      alphaVantage: 0,
      newsAPI: 0,
      youtubeAPI: 0
    },
    rateLimitReached: false,
    errors: []
  },
  
  // User Preferences
  preferences: {
    defaultSymbol: 'IBM',
    autoRefresh: false,
    refreshInterval: 300000, // 5 minutes
    notifications: true,
    soundAlerts: false
  },
  
  // Performance Metrics
  performance: {
    loadTime: 0,
    apiResponseTime: 0,
    chartRenderTime: 0
  }
};

// ========================================
// STATE OBSERVERS
// ========================================

/**
 * Observer pattern for state changes
 */
const StateObservers = {
  observers: new Map(),
  
  /**
   * Subscribe to state changes
   */
  subscribe(key, callback) {
    if (!this.observers.has(key)) {
      this.observers.set(key, []);
    }
    this.observers.get(key).push(callback);
    
    // Return unsubscribe function
    return () => {
      const callbacks = this.observers.get(key);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    };
  },
  
  /**
   * Notify observers of state change
   */
  notify(key, newValue, oldValue) {
    if (this.observers.has(key)) {
      this.observers.get(key).forEach(callback => {
        callback(newValue, oldValue);
      });
    }
  }
};

// ========================================
// STATE MANAGEMENT FUNCTIONS
// ========================================

/**
 * Get state value
 */
function getState(path) {
  const keys = path.split('.');
  let value = GlobalState;
  
  for (const key of keys) {
    if (value && typeof value === 'object' && key in value) {
      value = value[key];
    } else {
      return undefined;
    }
  }
  
  return value;
}

/**
 * Set state value with notification
 */
function setState(path, value) {
  const keys = path.split('.');
  const lastKey = keys.pop();
  let obj = GlobalState;
  
  // Navigate to parent object
  for (const key of keys) {
    if (!(key in obj)) {
      obj[key] = {};
    }
    obj = obj[key];
  }
  
  // Store old value
  const oldValue = obj[lastKey];
  
  // Set new value
  obj[lastKey] = value;
  
  // Notify observers
  StateObservers.notify(path, value, oldValue);
  
  // Save to localStorage if preference data
  if (path.startsWith('preferences.')) {
    savePreferences();
  }
  
  return value;
}

/**
 * Update state value (merge objects)
 */
function updateState(path, updates) {
  const current = getState(path);
  
  if (typeof current === 'object' && !Array.isArray(current)) {
    const merged = { ...current, ...updates };
    setState(path, merged);
    return merged;
  } else {
    setState(path, updates);
    return updates;
  }
}

/**
 * Reset state to initial values
 */
function resetState() {
  const initialState = {
    currentSymbol: '',
    lastUpdated: null,
    isAnalyzing: false,
    stockData: {
      raw: [],
      processed: [],
      metadata: {
        symbol: '',
        company: '',
        lastRefreshed: null,
        dataPoints: 0
      }
    },
    indicators: {
      sma20: [],
      sma50: [],
      sma200: [],
      ema12: [],
      ema26: [],
      rsi: [],
      macd: [],
      bollingerBands: [],
      stochastic: []
    },
    riskMetrics: {
      volatility: 0,
      annualizedVolatility: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      beta: 1.0,
      var95: 0,
      cvar95: 0
    },
    marketData: {
      news: [],
      videos: [],
      sentiment: {
        overall: 'neutral',
        score: 0,
        articles: 0
      }
    },
    signals: {
      current: {
        type: 'hold',
        strength: 0,
        message: '',
        timestamp: null
      },
      history: []
    }
  };
  
  Object.assign(GlobalState, initialState);
  console.log('State reset to initial values');
}

// ========================================
// LOCAL STORAGE PERSISTENCE
// ========================================

/**
 * Save preferences to localStorage
 */
function savePreferences() {
  try {
    const prefs = getState('preferences');
    localStorage.setItem('stockAnalyzerPreferences', JSON.stringify(prefs));
  } catch (error) {
    console.error('Failed to save preferences:', error);
  }
}

/**
 * Load preferences from localStorage
 */
function loadPreferences() {
  try {
    const saved = localStorage.getItem('stockAnalyzerPreferences');
    if (saved) {
      const prefs = JSON.parse(saved);
      setState('preferences', prefs);
      console.log('Preferences loaded from localStorage');
    }
  } catch (error) {
    console.error('Failed to load preferences:', error);
  }
}

/**
 * Save recent searches
 */
function saveRecentSearch(symbol) {
  try {
    let recent = JSON.parse(localStorage.getItem('recentSearches') || '[]');
    
    // Remove duplicate if exists
    recent = recent.filter(s => s !== symbol);
    
    // Add to beginning
    recent.unshift(symbol);
    
    // Keep only last 10
    recent = recent.slice(0, 10);
    
    localStorage.setItem('recentSearches', JSON.stringify(recent));
  } catch (error) {
    console.error('Failed to save recent search:', error);
  }
}

/**
 * Get recent searches
 */
function getRecentSearches() {
  try {
    return JSON.parse(localStorage.getItem('recentSearches') || '[]');
  } catch (error) {
    console.error('Failed to get recent searches:', error);
    return [];
  }
}

/**
 * Clear all stored data
 */
function clearStoredData() {
  try {
    localStorage.removeItem('stockAnalyzerPreferences');
    localStorage.removeItem('recentSearches');
    console.log('Stored data cleared');
  } catch (error) {
    console.error('Failed to clear stored data:', error);
  }
}

// ========================================
// STATE VALIDATION
// ========================================

/**
 * Validate stock data
 */
function validateStockData(data) {
  if (!Array.isArray(data)) return false;
  if (data.length === 0) return false;
  
  // Check required fields
  const requiredFields = ['date', 'open', 'high', 'low', 'close', 'volume'];
  return data.every(item => {
    return requiredFields.every(field => field in item && item[field] !== null);
  });
}

/**
 * Validate indicators
 */
function validateIndicators(indicators) {
  if (!indicators || typeof indicators !== 'object') return false;
  
  // Check if at least one indicator exists
  const hasIndicators = Object.keys(indicators).some(key => {
    return Array.isArray(indicators[key]) && indicators[key].length > 0;
  });
  
  return hasIndicators;
}

// ========================================
// STATE COMPUTATION HELPERS
// ========================================

/**
 * Get current price info
 */
function getCurrentPriceInfo() {
  const data = getState('stockData.raw');
  if (!data || data.length === 0) return null;
  
  const current = data[data.length - 1];
  const previous = data.length > 1 ? data[data.length - 2] : current;
  
  const change = current.close - previous.close;
  const changePercent = (change / previous.close) * 100;
  
  return {
    current: current.close,
    previous: previous.close,
    change: change,
    changePercent: changePercent,
    high: current.high,
    low: current.low,
    volume: current.volume,
    date: current.date
  };
}

/**
 * Get indicator summary
 */
function getIndicatorSummary() {
  const indicators = getState('indicators');
  const summary = {};
  
  Object.keys(indicators).forEach(key => {
    const data = indicators[key];
    if (Array.isArray(data) && data.length > 0) {
      const latest = data[data.length - 1];
      summary[key] = {
        current: latest.value || latest,
        length: data.length
      };
    }
  });
  
  return summary;
}

/**
 * Calculate portfolio statistics
 */
function getPortfolioStats() {
  const stockData = getState('stockData.raw');
  if (!stockData || stockData.length === 0) return null;
  
  const prices = stockData.map(d => d.close);
  const volumes = stockData.map(d => d.volume);
  
  return {
    avgPrice: prices.reduce((a, b) => a + b, 0) / prices.length,
    minPrice: Math.min(...prices),
    maxPrice: Math.max(...prices),
    avgVolume: volumes.reduce((a, b) => a + b, 0) / volumes.length,
    dataPoints: stockData.length,
    dateRange: {
      start: stockData[0].date,
      end: stockData[stockData.length - 1].date
    }
  };
}

// ========================================
// PERFORMANCE TRACKING
// ========================================

/**
 * Start performance timer
 */
function startTimer(name) {
  const key = `timer_${name}`;
  GlobalState.performance[key] = performance.now();
}

/**
 * End performance timer and log
 */
function endTimer(name) {
  const key = `timer_${name}`;
  if (GlobalState.performance[key]) {
    const duration = performance.now() - GlobalState.performance[key];
    console.log(`⏱️ ${name}: ${duration.toFixed(2)}ms`);
    delete GlobalState.performance[key];
    return duration;
  }
  return 0;
}

// ========================================
// DEBUG HELPERS
// ========================================

/**
 * Log current state (debug)
 */
function logState() {
  console.log('📊 Current State:', {
    symbol: GlobalState.currentSymbol,
    dataPoints: GlobalState.stockData.raw.length,
    indicators: Object.keys(GlobalState.indicators).filter(k => 
      GlobalState.indicators[k].length > 0
    ),
    signal: GlobalState.signals.current,
    lastUpdated: GlobalState.lastUpdated
  });
}

/**
 * Export state as JSON
 */
function exportState() {
  const exportData = {
    timestamp: new Date().toISOString(),
    symbol: GlobalState.currentSymbol,
    stockData: GlobalState.stockData,
    indicators: GlobalState.indicators,
    riskMetrics: GlobalState.riskMetrics,
    signals: GlobalState.signals
  };
  
  const json = JSON.stringify(exportData, null, 2);
  return json;
}

/**
 * Download state as file
 */
function downloadState() {
  const json = exportState();
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `stock-analysis-${GlobalState.currentSymbol}-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ========================================
// INITIALIZATION
// ========================================

/**
 * Initialize state management
 */
function initStateManagement() {
  console.log('🔧 Initializing state management...');
  
  // Load preferences
  loadPreferences();
  
  // Set up state observers
  StateObservers.subscribe('currentSymbol', (newSymbol, oldSymbol) => {
    if (newSymbol && newSymbol !== oldSymbol) {
      saveRecentSearch(newSymbol);
      console.log(`Symbol changed: ${oldSymbol} → ${newSymbol}`);
    }
  });
  
  StateObservers.subscribe('signals.current', (newSignal) => {
    console.log(`🎯 New signal: ${newSignal.type.toUpperCase()} - ${newSignal.message}`);
  });
  
  StateObservers.subscribe('ui.theme', (newTheme) => {
    document.body.classList.toggle('theme-light', newTheme === 'light');
    console.log(`🎨 Theme changed to: ${newTheme}`);
  });
  
  console.log('✅ State management initialized');
}

// Initialize on load
if (typeof window !== 'undefined') {
  initStateManagement();
}

// ========================================
// EXPORTS
// ========================================

// Export state management functions
window.StateManager = {
  // Core functions
  getState,
  setState,
  updateState,
  resetState,
  
  // Observers
  subscribe: StateObservers.subscribe.bind(StateObservers),
  
  // Persistence
  savePreferences,
  loadPreferences,
  saveRecentSearch,
  getRecentSearches,
  clearStoredData,
  
  // Validation
  validateStockData,
  validateIndicators,
  
  // Helpers
  getCurrentPriceInfo,
  getIndicatorSummary,
  getPortfolioStats,
  
  // Performance
  startTimer,
  endTimer,
  
  // Debug
  logState,
  exportState,
  downloadState,
  
  // Direct access to state (read-only)
  state: GlobalState
};
