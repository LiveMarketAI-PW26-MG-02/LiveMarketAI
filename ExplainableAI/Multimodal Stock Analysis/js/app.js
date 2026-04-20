/**
 * Advanced Stock Analyzer - Main Application File
 * Bootstraps the application and coordinates all modules
 */

// ========================================
// API CONFIGURATION
// ========================================
const CONFIG = {
  // Replace these with your actual API keys
  ALPHA_VANTAGE_KEY: 'demo', // Get from: https://www.alphavantage.co/support/#api-key
  NEWS_API_KEY: 'YOUR_NEWS_API_KEY', // Get from: https://newsapi.org/register
  YOUTUBE_API_KEY: 'YOUR_YOUTUBE_API_KEY', // Get from: https://console.developers.google.com/
  
  // API Endpoints
  ENDPOINTS: {
    ALPHA_VANTAGE: 'https://www.alphavantage.co/query',
    NEWS_API: 'https://newsapi.org/v2/everything',
    YOUTUBE_API: 'https://www.googleapis.com/youtube/v3/search'
  },
  
  // Default Settings
  DEFAULT_SYMBOL: 'IBM',
  DATA_POINTS: 100,
  NEWS_HOURS: 24,
  VIDEO_HOURS: 1,
  
  // Rate Limiting
  RATE_LIMIT_DELAY: 12000 // 12 seconds between API calls (Alpha Vantage free tier)
};

// ========================================
// APPLICATION STATE
// ========================================
const AppState = {
  currentSymbol: '',
  stockData: [],
  indicators: {},
  news: [],
  videos: [],
  isLoading: false,
  lastAPICall: 0,
  charts: {
    price: null,
    volume: null,
    rsi: null
  }
};

// ========================================
// DOM ELEMENTS
// ========================================
const DOM = {
  // Input Elements
  symbolInput: document.getElementById('symbol-input'),
  analyzeBtn: document.getElementById('analyze-btn'),
  
  // Display Elements
  loading: document.getElementById('loading'),
  errorMsg: document.getElementById('error-msg'),
  errorText: document.getElementById('error-text'),
  dashboard: document.getElementById('dashboard'),
  
  // Price Cards
  currentPrice: document.getElementById('current-price'),
  priceChange: document.getElementById('price-change'),
  volatility: document.getElementById('volatility'),
  maxDrawdown: document.getElementById('max-drawdown'),
  signalType: document.getElementById('signal-type'),
  signalMessage: document.getElementById('signal-message'),
  signalCard: document.getElementById('signal-card'),
  
  // Charts
  priceChart: document.getElementById('price-chart'),
  volumeChart: document.getElementById('volume-chart'),
  rsiChart: document.getElementById('rsi-chart'),
  
  // Indicators
  currentRSI: document.getElementById('current-rsi'),
  rsiStatus: document.getElementById('rsi-status'),
  rsiInterpretation: document.getElementById('rsi-interpretation'),
  macdLine: document.getElementById('macd-line'),
  signalLine: document.getElementById('signal-line'),
  histogram: document.getElementById('histogram'),
  riskVolatility: document.getElementById('risk-volatility'),
  riskDrawdown: document.getElementById('risk-drawdown'),
  
  // Content Containers
  newsContainer: document.getElementById('news-container'),
  videosContainer: document.getElementById('videos-container'),
  
  // Tabs
  tabButtons: document.querySelectorAll('.tab-btn'),
  tabContents: document.querySelectorAll('.tab-content'),
  
  // Quick Stock Buttons
  quickStockBtns: document.querySelectorAll('.stock-quick-btn'),
  
  // Current Time Display
  currentTime: document.getElementById('current-time')
};

// ========================================
// UTILITY FUNCTIONS
// ========================================

/**
 * Show loading state
 */
function showLoading() {
  AppState.isLoading = true;
  DOM.loading.classList.remove('hidden');
  DOM.dashboard.classList.add('hidden');
  DOM.errorMsg.classList.add('hidden');
  DOM.analyzeBtn.disabled = true;
  DOM.analyzeBtn.innerHTML = `
    <i data-lucide="refresh-cw" class="w-5 h-5 animate-spin"></i>
    <span>Analyzing...</span>
  `;
  lucide.createIcons();
}

/**
 * Hide loading state
 */
function hideLoading() {
  AppState.isLoading = false;
  DOM.loading.classList.add('hidden');
  DOM.analyzeBtn.disabled = false;
  DOM.analyzeBtn.innerHTML = `
    <i data-lucide="activity" class="w-5 h-5"></i>
    <span>Analyze</span>
  `;
  lucide.createIcons();
}

/**
 * Show error message
 */
function showError(message) {
  DOM.errorText.textContent = message;
  DOM.errorMsg.classList.remove('hidden');
  setTimeout(() => {
    DOM.errorMsg.classList.add('hidden');
  }, 5000);
}

/**
 * Format time ago
 */
function formatTimeAgo(dateString) {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

/**
 * Update current time display
 */
function updateCurrentTime() {
  const now = new Date();
  const timeString = now.toLocaleTimeString('en-US', { 
    hour: '2-digit', 
    minute: '2-digit',
    hour12: true 
  });
  const dateString = now.toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric',
    year: 'numeric'
  });
  if (DOM.currentTime) {
    DOM.currentTime.textContent = `${dateString} • ${timeString}`;
  }
}

/**
 * Check rate limit
 */
function checkRateLimit() {
  const now = Date.now();
  const timeSinceLastCall = now - AppState.lastAPICall;
  
  if (timeSinceLastCall < CONFIG.RATE_LIMIT_DELAY) {
    const waitTime = Math.ceil((CONFIG.RATE_LIMIT_DELAY - timeSinceLastCall) / 1000);
    showError(`Please wait ${waitTime} seconds before making another request.`);
    return false;
  }
  
  AppState.lastAPICall = now;
  return true;
}

// ========================================
// DATA FETCHING FUNCTIONS
// ========================================

/**
 * Fetch stock data from Alpha Vantage
 */
async function fetchStockData(symbol) {
  try {
    const url = `${CONFIG.ENDPOINTS.ALPHA_VANTAGE}?function=TIME_SERIES_DAILY&symbol=${symbol}&apikey=${CONFIG.ALPHA_VANTAGE_KEY}`;
    const response = await fetch(url);
    const data = await response.json();
    
    // Check for errors
    if (data['Error Message']) {
      throw new Error('Invalid stock symbol');
    }
    
    if (data['Note']) {
      // API limit reached, use mock data
      console.warn('Alpha Vantage API limit reached. Using mock data.');
      return generateMockStockData(symbol);
    }
    
    if (data['Time Series (Daily)']) {
      const timeSeries = data['Time Series (Daily)'];
      const formatted = Object.keys(timeSeries)
        .slice(0, CONFIG.DATA_POINTS)
        .reverse()
        .map(date => ({
          date: date,
          open: parseFloat(timeSeries[date]['1. open']),
          high: parseFloat(timeSeries[date]['2. high']),
          low: parseFloat(timeSeries[date]['3. low']),
          close: parseFloat(timeSeries[date]['4. close']),
          volume: parseInt(timeSeries[date]['5. volume'])
        }));
      
      return formatted;
    }
    
    // Fallback to mock data
    return generateMockStockData(symbol);
    
  } catch (error) {
    console.error('Error fetching stock data:', error);
    showError('Failed to fetch stock data. Using demo data.');
    return generateMockStockData(symbol);
  }
}

/**
 * Fetch news from News API
 */
async function fetchNews(symbol) {
  try {
    // Calculate time filter (last 24 hours)
    const fromDate = new Date();
    fromDate.setHours(fromDate.getHours() - CONFIG.NEWS_HOURS);
    const fromISO = fromDate.toISOString();
    
    const url = `${CONFIG.ENDPOINTS.NEWS_API}?q=${symbol}+stock&from=${fromISO}&sortBy=publishedAt&language=en&apiKey=${CONFIG.NEWS_API_KEY}`;
    const response = await fetch(url);
    const data = await response.json();
    
    if (data.status === 'ok' && data.articles) {
      return data.articles.slice(0, 5);
    }
    
    // Fallback to mock news
    return generateMockNews(symbol);
    
  } catch (error) {
    console.error('Error fetching news:', error);
    return generateMockNews(symbol);
  }
}

/**
 * Fetch YouTube videos (last hour only)
 */
async function fetchYouTubeVideos(symbol) {
  try {
    // Calculate time filter (last 1 hour)
    const oneHourAgo = new Date();
    oneHourAgo.setHours(oneHourAgo.getHours() - CONFIG.VIDEO_HOURS);
    const publishedAfter = oneHourAgo.toISOString();
    
    const url = `${CONFIG.ENDPOINTS.YOUTUBE_API}?part=snippet&q=${symbol}+stock+news&type=video&order=date&publishedAfter=${publishedAfter}&maxResults=6&key=${CONFIG.YOUTUBE_API_KEY}`;
    const response = await fetch(url);
    const data = await response.json();
    
    if (data.items && data.items.length > 0) {
      return data.items;
    }
    
    // Fallback to mock videos
    return generateMockVideos(symbol);
    
  } catch (error) {
    console.error('Error fetching YouTube videos:', error);
    return generateMockVideos(symbol);
  }
}

// ========================================
// ANALYSIS FUNCTIONS
// ========================================

/**
 * Analyze stock and update UI
 */
async function analyzeStock(symbol) {
  if (!symbol || symbol.trim() === '') {
    showError('Please enter a valid stock symbol');
    return;
  }
  
  symbol = symbol.toUpperCase();
  AppState.currentSymbol = symbol;
  
  showLoading();
  
  try {
    // Fetch all data
    const [stockData, news, videos] = await Promise.all([
      fetchStockData(symbol),
      fetchNews(symbol),
      fetchYouTubeVideos(symbol)
    ]);
    
    // Update state
    AppState.stockData = stockData;
    AppState.news = news;
    AppState.videos = videos;
    
    // Calculate indicators
    AppState.indicators = calculateAllIndicators(stockData);
    
    // Update UI
    updateDashboard();
    updateCharts();
    updateNewsSection();
    updateVideosSection();
    
    // Show dashboard
    DOM.dashboard.classList.remove('hidden');
    DOM.dashboard.classList.add('fade-in');
    
  } catch (error) {
    console.error('Error analyzing stock:', error);
    showError('An error occurred while analyzing the stock');
  } finally {
    hideLoading();
  }
}

/**
 * Calculate all technical indicators
 */
function calculateAllIndicators(data) {
  return {
    sma20: calculateSMA(data, 20),
    sma50: calculateSMA(data, 50),
    ema12: calculateEMA(data, 12),
    ema26: calculateEMA(data, 26),
    rsi: calculateRSI(data, 14),
    macd: calculateMACD(data),
    volatility: calculateVolatility(data, 20),
    maxDrawdown: calculateMaxDrawdown(data)
  };
}

// ========================================
// UI UPDATE FUNCTIONS
// ========================================

/**
 * Update dashboard cards
 */
function updateDashboard() {
  const data = AppState.stockData;
  const indicators = AppState.indicators;
  
  if (data.length === 0) return;
  
  // Current price
  const currentPrice = data[data.length - 1].close;
  const previousPrice = data[data.length - 2].close;
  const priceChange = currentPrice - previousPrice;
  const priceChangePercent = ((priceChange / previousPrice) * 100).toFixed(2);
  
  DOM.currentPrice.textContent = `$${currentPrice.toFixed(2)}`;
  
  // Price change with icon
  const isPositive = priceChange >= 0;
  const icon = isPositive ? 'trending-up' : 'trending-down';
  const colorClass = isPositive ? 'text-green-400' : 'text-red-400';
  
  DOM.priceChange.className = `flex items-center gap-2 mt-2 ${colorClass}`;
  DOM.priceChange.innerHTML = `
    <i data-lucide="${icon}" class="w-5 h-5"></i>
    <span>${isPositive ? '+' : ''}${priceChange.toFixed(2)} (${priceChangePercent}%)</span>
  `;
  
  // Volatility
  DOM.volatility.textContent = `${indicators.volatility.toFixed(2)}%`;
  DOM.riskVolatility.textContent = `${indicators.volatility.toFixed(2)}%`;
  
  // Max Drawdown
  DOM.maxDrawdown.textContent = `${indicators.maxDrawdown.toFixed(2)}%`;
  DOM.riskDrawdown.textContent = `-${indicators.maxDrawdown.toFixed(2)}%`;
  
  // Trading Signal
  const signal = generateTradingSignal(indicators);
  updateTradingSignal(signal);
  
  // Update indicator details
  updateIndicatorDetails(indicators);
  
  // Recreate icons
  lucide.createIcons();
}

/**
 * Update trading signal display
 */
function updateTradingSignal(signal) {
  DOM.signalType.textContent = signal.type.toUpperCase();
  DOM.signalMessage.textContent = signal.message;
  
  // Remove old border classes
  DOM.signalCard.classList.remove('border-green-600', 'border-red-600', 'border-yellow-600', 'border-slate-700');
  
  // Add appropriate border color
  if (signal.type === 'buy') {
    DOM.signalCard.classList.add('border-green-600');
    DOM.signalType.className = 'text-2xl font-bold text-green-400';
  } else if (signal.type === 'sell') {
    DOM.signalCard.classList.add('border-red-600');
    DOM.signalType.className = 'text-2xl font-bold text-red-400';
  } else {
    DOM.signalCard.classList.add('border-yellow-600');
    DOM.signalType.className = 'text-2xl font-bold text-yellow-400';
  }
}

/**
 * Update indicator details
 */
function updateIndicatorDetails(indicators) {
  // RSI
  const lastRSI = indicators.rsi[indicators.rsi.length - 1]?.value || 50;
  DOM.currentRSI.textContent = lastRSI.toFixed(2);
  
  if (lastRSI > 70) {
    DOM.rsiStatus.textContent = 'Overbought';
    DOM.rsiInterpretation.textContent = 'Potential reversal down';
  } else if (lastRSI < 30) {
    DOM.rsiStatus.textContent = 'Oversold';
    DOM.rsiInterpretation.textContent = 'Potential reversal up';
  } else {
    DOM.rsiStatus.textContent = 'Neutral';
    DOM.rsiInterpretation.textContent = 'Range-bound market';
  }
  
  // MACD
  const lastMACD = indicators.macd[indicators.macd.length - 1];
  if (lastMACD) {
    DOM.macdLine.textContent = lastMACD.macd.toFixed(2);
    DOM.signalLine.textContent = lastMACD.signal.toFixed(2);
    DOM.histogram.textContent = lastMACD.histogram.toFixed(2);
    DOM.histogram.className = `text-xl font-semibold ${lastMACD.histogram > 0 ? 'text-green-400' : 'text-red-400'}`;
  }
}

/**
 * Update news section
 */
function updateNewsSection() {
  DOM.newsContainer.innerHTML = '';
  
  if (AppState.news.length === 0) {
    DOM.newsContainer.innerHTML = `
      <div class="text-slate-400 text-center py-8">
        No recent news available. Check your News API configuration.
      </div>
    `;
    return;
  }
  
  AppState.news.forEach(article => {
    const newsItem = document.createElement('div');
    newsItem.className = 'news-item';
    newsItem.innerHTML = `
      <a href="${article.url}" target="_blank" rel="noopener noreferrer" class="news-title text-lg">
        ${article.title}
      </a>
      <p class="text-slate-400 text-sm mt-2">${article.description || 'No description available'}</p>
      <div class="news-meta">
        <span>${article.source?.name || 'Unknown Source'}</span>
        <span>${formatTimeAgo(article.publishedAt)}</span>
      </div>
    `;
    DOM.newsContainer.appendChild(newsItem);
  });
}

/**
 * Update videos section
 */
function updateVideosSection() {
  DOM.videosContainer.innerHTML = '';
  
  if (AppState.videos.length === 0) {
    DOM.videosContainer.innerHTML = `
      <div class="text-slate-400 text-center py-8 col-span-2">
        No recent videos available. Check your YouTube API configuration.
      </div>
    `;
    return;
  }
  
  AppState.videos.forEach(video => {
    const videoCard = document.createElement('div');
    videoCard.className = 'video-card';
    
    const thumbnail = video.snippet?.thumbnails?.medium?.url || 'https://via.placeholder.com/320x180?text=Video';
    const videoId = video.id?.videoId || '';
    const videoUrl = videoId ? `https://www.youtube.com/watch?v=${videoId}` : '#';
    
    videoCard.innerHTML = `
      <img src="${thumbnail}" alt="${video.snippet?.title}" class="video-thumbnail">
      <div class="video-content">
        <a href="${videoUrl}" target="_blank" rel="noopener noreferrer" class="video-title">
          ${video.snippet?.title || 'Untitled Video'}
        </a>
        <p class="text-slate-400 text-sm mt-2 line-clamp-2">
          ${video.snippet?.description || 'No description available'}
        </p>
        <div class="text-xs text-slate-500 mt-2">
          ${formatTimeAgo(video.snippet?.publishedAt)}
        </div>
      </div>
    `;
    DOM.videosContainer.appendChild(videoCard);
  });
}

// ========================================
// EVENT HANDLERS
// ========================================

/**
 * Handle analyze button click
 */
function handleAnalyze() {
  const symbol = DOM.symbolInput.value.trim();
  if (checkRateLimit()) {
    analyzeStock(symbol);
  }
}

/**
 * Handle tab switching
 */
function handleTabSwitch(tabName) {
  // Update tab buttons
  DOM.tabButtons.forEach(btn => {
    if (btn.dataset.tab === tabName) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  
  // Update tab content
  DOM.tabContents.forEach(content => {
    if (content.id === `${tabName}-tab`) {
      content.classList.remove('hidden');
      content.classList.add('fade-in');
    } else {
      content.classList.add('hidden');
    }
  });
}

/**
 * Handle quick stock button click
 */
function handleQuickStock(symbol) {
  DOM.symbolInput.value = symbol;
  if (checkRateLimit()) {
    analyzeStock(symbol);
  }
}

// ========================================
// EVENT LISTENERS
// ========================================

// Analyze button
DOM.analyzeBtn.addEventListener('click', handleAnalyze);

// Enter key on input
DOM.symbolInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    handleAnalyze();
  }
});

// Tab buttons
DOM.tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    handleTabSwitch(btn.dataset.tab);
  });
});

// Quick stock buttons
DOM.quickStockBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    handleQuickStock(btn.dataset.symbol);
  });
});

// ========================================
// INITIALIZATION
// ========================================

/**
 * Initialize application
 */
function initApp() {
  console.log('🚀 Advanced Stock Analyzer - Initializing...');
  
  // Set default symbol
  DOM.symbolInput.value = CONFIG.DEFAULT_SYMBOL;
  
  // Update time display
  updateCurrentTime();
  setInterval(updateCurrentTime, 1000);
  
  // Initialize Lucide icons
  lucide.createIcons();
  
  // Load default stock
  setTimeout(() => {
    analyzeStock(CONFIG.DEFAULT_SYMBOL);
  }, 500);
  
  console.log('✅ Application initialized successfully');
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

// Export for use in other modules
window.StockAnalyzer = {
  AppState,
  CONFIG,
  analyzeStock,
  updateCharts
};
