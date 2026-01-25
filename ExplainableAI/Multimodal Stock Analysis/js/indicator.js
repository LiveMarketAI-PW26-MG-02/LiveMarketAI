/**
 * Advanced Stock Analyzer - Technical Indicators
 * Comprehensive technical analysis indicators implementation
 */

// ========================================
// MOVING AVERAGES
// ========================================

/**
 * Simple Moving Average (SMA)
 * @param {Array} data - Stock data array with close prices
 * @param {Number} period - Period for SMA calculation
 * @returns {Array} SMA values
 */
function calculateSMA(data, period) {
  if (!data || data.length < period) {
    return [];
  }
  
  const result = [];
  
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const sum = slice.reduce((acc, item) => acc + item.close, 0);
    const average = sum / period;
    
    result.push({
      date: data[i].date,
      value: parseFloat(average.toFixed(2))
    });
  }
  
  return result;
}

/**
 * Exponential Moving Average (EMA)
 * @param {Array} data - Stock data array with close prices
 * @param {Number} period - Period for EMA calculation
 * @returns {Array} EMA values
 */
function calculateEMA(data, period) {
  if (!data || data.length < period) {
    return [];
  }
  
  const result = [];
  const multiplier = 2 / (period + 1);
  
  // Calculate initial SMA as first EMA value
  let ema = data.slice(0, period).reduce((acc, item) => acc + item.close, 0) / period;
  
  result.push({
    date: data[period - 1].date,
    value: parseFloat(ema.toFixed(2))
  });
  
  // Calculate EMA for remaining data points
  for (let i = period; i < data.length; i++) {
    ema = (data[i].close - ema) * multiplier + ema;
    result.push({
      date: data[i].date,
      value: parseFloat(ema.toFixed(2))
    });
  }
  
  return result;
}

/**
 * Weighted Moving Average (WMA)
 * @param {Array} data - Stock data array
 * @param {Number} period - Period for WMA calculation
 * @returns {Array} WMA values
 */
function calculateWMA(data, period) {
  if (!data || data.length < period) {
    return [];
  }
  
  const result = [];
  const weightSum = (period * (period + 1)) / 2;
  
  for (let i = period - 1; i < data.length; i++) {
    let weightedSum = 0;
    
    for (let j = 0; j < period; j++) {
      const weight = j + 1;
      weightedSum += data[i - period + 1 + j].close * weight;
    }
    
    result.push({
      date: data[i].date,
      value: parseFloat((weightedSum / weightSum).toFixed(2))
    });
  }
  
  return result;
}

// ========================================
// MOMENTUM INDICATORS
// ========================================

/**
 * Relative Strength Index (RSI)
 * @param {Array} data - Stock data array
 * @param {Number} period - Period for RSI calculation (default 14)
 * @returns {Array} RSI values
 */
function calculateRSI(data, period = 14) {
  if (!data || data.length < period + 1) {
    return [];
  }
  
  const result = [];
  const changes = [];
  
  // Calculate price changes
  for (let i = 1; i < data.length; i++) {
    changes.push(data[i].close - data[i - 1].close);
  }
  
  // Calculate RSI for each point
  for (let i = period; i < changes.length; i++) {
    const segment = changes.slice(i - period, i);
    
    const gains = segment.filter(change => change > 0);
    const losses = segment.filter(change => change < 0).map(Math.abs);
    
    const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / period : 0;
    const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / period : 0;
    
    let rsi;
    if (avgLoss === 0) {
      rsi = 100;
    } else {
      const rs = avgGain / avgLoss;
      rsi = 100 - (100 / (1 + rs));
    }
    
    result.push({
      date: data[i + 1].date,
      value: parseFloat(rsi.toFixed(2))
    });
  }
  
  return result;
}

/**
 * Stochastic Oscillator
 * @param {Array} data - Stock data array
 * @param {Number} period - Period for calculation (default 14)
 * @param {Number} smoothK - Smoothing for %K (default 3)
 * @param {Number} smoothD - Smoothing for %D (default 3)
 * @returns {Array} Stochastic values
 */
function calculateStochastic(data, period = 14, smoothK = 3, smoothD = 3) {
  if (!data || data.length < period) {
    return [];
  }
  
  const result = [];
  const rawK = [];
  
  // Calculate raw %K
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const highest = Math.max(...slice.map(d => d.high));
    const lowest = Math.min(...slice.map(d => d.low));
    const current = data[i].close;
    
    const k = ((current - lowest) / (highest - lowest)) * 100;
    rawK.push(k);
  }
  
  // Smooth %K
  const smoothedK = [];
  for (let i = smoothK - 1; i < rawK.length; i++) {
    const slice = rawK.slice(i - smoothK + 1, i + 1);
    const avg = slice.reduce((a, b) => a + b, 0) / smoothK;
    smoothedK.push(avg);
  }
  
  // Calculate %D (SMA of %K)
  for (let i = smoothD - 1; i < smoothedK.length; i++) {
    const slice = smoothedK.slice(i - smoothD + 1, i + 1);
    const d = slice.reduce((a, b) => a + b, 0) / smoothD;
    
    const dataIndex = period - 1 + smoothK - 1 + i;
    
    result.push({
      date: data[dataIndex].date,
      k: parseFloat(smoothedK[i].toFixed(2)),
      d: parseFloat(d.toFixed(2))
    });
  }
  
  return result;
}

/**
 * Rate of Change (ROC)
 * @param {Array} data - Stock data array
 * @param {Number} period - Period for ROC calculation
 * @returns {Array} ROC values
 */
function calculateROC(data, period = 12) {
  if (!data || data.length < period + 1) {
    return [];
  }
  
  const result = [];
  
  for (let i = period; i < data.length; i++) {
    const currentPrice = data[i].close;
    const pastPrice = data[i - period].close;
    const roc = ((currentPrice - pastPrice) / pastPrice) * 100;
    
    result.push({
      date: data[i].date,
      value: parseFloat(roc.toFixed(2))
    });
  }
  
  return result;
}

// ========================================
// TREND INDICATORS
// ========================================

/**
 * Moving Average Convergence Divergence (MACD)
 * @param {Array} data - Stock data array
 * @param {Number} fastPeriod - Fast EMA period (default 12)
 * @param {Number} slowPeriod - Slow EMA period (default 26)
 * @param {Number} signalPeriod - Signal line period (default 9)
 * @returns {Array} MACD values
 */
function calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  if (!data || data.length < slowPeriod) {
    return [];
  }
  
  // Calculate EMAs
  const emaFast = calculateEMA(data, fastPeriod);
  const emaSlow = calculateEMA(data, slowPeriod);
  
  // Calculate MACD line
  const macdLine = [];
  const startIndex = slowPeriod - fastPeriod;
  
  for (let i = 0; i < emaSlow.length; i++) {
    const macd = emaFast[i + startIndex].value - emaSlow[i].value;
    macdLine.push({
      date: emaSlow[i].date,
      macd: parseFloat(macd.toFixed(2))
    });
  }
  
  // Calculate signal line (EMA of MACD)
  const signalLine = calculateEMA(
    macdLine.map(m => ({ close: m.macd, date: m.date })),
    signalPeriod
  );
  
  // Combine MACD line, signal line, and histogram
  const result = [];
  for (let i = 0; i < signalLine.length; i++) {
    const histogram = macdLine[i + signalPeriod - 1].macd - signalLine[i].value;
    
    result.push({
      date: signalLine[i].date,
      macd: macdLine[i + signalPeriod - 1].macd,
      signal: signalLine[i].value,
      histogram: parseFloat(histogram.toFixed(2))
    });
  }
  
  return result;
}

/**
 * Average Directional Index (ADX)
 * @param {Array} data - Stock data array
 * @param {Number} period - Period for ADX calculation (default 14)
 * @returns {Array} ADX values
 */
function calculateADX(data, period = 14) {
  if (!data || data.length < period + 1) {
    return [];
  }
  
  const result = [];
  const trueRanges = [];
  const plusDM = [];
  const minusDM = [];
  
  // Calculate True Range and Directional Movement
  for (let i = 1; i < data.length; i++) {
    const high = data[i].high;
    const low = data[i].low;
    const prevHigh = data[i - 1].high;
    const prevLow = data[i - 1].low;
    const prevClose = data[i - 1].close;
    
    // True Range
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    trueRanges.push(tr);
    
    // Directional Movement
    const upMove = high - prevHigh;
    const downMove = prevLow - low;
    
    plusDM.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDM.push(downMove > upMove && downMove > 0 ? downMove : 0);
  }
  
  // Calculate smoothed values and ADX
  for (let i = period - 1; i < trueRanges.length; i++) {
    const avgTR = trueRanges.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
    const avgPlusDM = plusDM.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
    const avgMinusDM = minusDM.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
    
    const plusDI = (avgPlusDM / avgTR) * 100;
    const minusDI = (avgMinusDM / avgTR) * 100;
    
    const dx = Math.abs(plusDI - minusDI) / (plusDI + minusDI) * 100;
    
    result.push({
      date: data[i + 1].date,
      adx: parseFloat(dx.toFixed(2)),
      plusDI: parseFloat(plusDI.toFixed(2)),
      minusDI: parseFloat(minusDI.toFixed(2))
    });
  }
  
  return result;
}

// ========================================
// VOLATILITY INDICATORS
// ========================================

/**
 * Bollinger Bands
 * @param {Array} data - Stock data array
 * @param {Number} period - Period for calculation (default 20)
 * @param {Number} stdDev - Standard deviations (default 2)
 * @returns {Array} Bollinger Bands values
 */
function calculateBollingerBands(data, period = 20, stdDev = 2) {
  if (!data || data.length < period) {
    return [];
  }
  
  const result = [];
  const sma = calculateSMA(data, period);
  
  for (let i = 0; i < sma.length; i++) {
    const dataIndex = i + period - 1;
    const slice = data.slice(i, dataIndex + 1);
    const mean = sma[i].value;
    
    // Calculate standard deviation
    const squaredDiffs = slice.map(item => Math.pow(item.close - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period;
    const standardDeviation = Math.sqrt(variance);
    
    result.push({
      date: data[dataIndex].date,
      middle: mean,
      upper: parseFloat((mean + stdDev * standardDeviation).toFixed(2)),
      lower: parseFloat((mean - stdDev * standardDeviation).toFixed(2)),
      bandwidth: parseFloat((2 * stdDev * standardDeviation).toFixed(2))
    });
  }
  
  return result;
}

/**
 * Average True Range (ATR)
 * @param {Array} data - Stock data array
 * @param {Number} period - Period for ATR calculation (default 14)
 * @returns {Array} ATR values
 */
function calculateATR(data, period = 14) {
  if (!data || data.length < period + 1) {
    return [];
  }
  
  const result = [];
  const trueRanges = [];
  
  // Calculate True Range for each period
  for (let i = 1; i < data.length; i++) {
    const high = data[i].high;
    const low = data[i].low;
    const prevClose = data[i - 1].close;
    
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    
    trueRanges.push(tr);
  }
  
  // Calculate ATR
  for (let i = period - 1; i < trueRanges.length; i++) {
    const slice = trueRanges.slice(i - period + 1, i + 1);
    const atr = slice.reduce((a, b) => a + b, 0) / period;
    
    result.push({
      date: data[i + 1].date,
      value: parseFloat(atr.toFixed(2))
    });
  }
  
  return result;
}

/**
 * Keltner Channels
 * @param {Array} data - Stock data array
 * @param {Number} period - Period for calculation (default 20)
 * @param {Number} multiplier - ATR multiplier (default 2)
 * @returns {Array} Keltner Channel values
 */
function calculateKeltnerChannels(data, period = 20, multiplier = 2) {
  if (!data || data.length < period) {
    return [];
  }
  
  const ema = calculateEMA(data, period);
  const atr = calculateATR(data, period);
  
  const result = [];
  const startIndex = period;
  
  for (let i = 0; i < atr.length && i < ema.length - startIndex + 1; i++) {
    const middle = ema[i + startIndex - 1].value;
    const atrValue = atr[i].value;
    
    result.push({
      date: atr[i].date,
      middle: middle,
      upper: parseFloat((middle + multiplier * atrValue).toFixed(2)),
      lower: parseFloat((middle - multiplier * atrValue).toFixed(2))
    });
  }
  
  return result;
}

// ========================================
// VOLUME INDICATORS
// ========================================

/**
 * On-Balance Volume (OBV)
 * @param {Array} data - Stock data array
 * @returns {Array} OBV values
 */
function calculateOBV(data) {
  if (!data || data.length < 2) {
    return [];
  }
  
  const result = [];
  let obv = 0;
  
  result.push({
    date: data[0].date,
    value: 0
  });
  
  for (let i = 1; i < data.length; i++) {
    if (data[i].close > data[i - 1].close) {
      obv += data[i].volume;
    } else if (data[i].close < data[i - 1].close) {
      obv -= data[i].volume;
    }
    
    result.push({
      date: data[i].date,
      value: obv
    });
  }
  
  return result;
}

/**
 * Volume-Weighted Average Price (VWAP)
 * @param {Array} data - Stock data array
 * @returns {Array} VWAP values
 */
function calculateVWAP(data) {
  if (!data || data.length === 0) {
    return [];
  }
  
  const result = [];
  let cumulativeTPV = 0; // Typical Price * Volume
  let cumulativeVolume = 0;
  
  for (let i = 0; i < data.length; i++) {
    const typicalPrice = (data[i].high + data[i].low + data[i].close) / 3;
    const tpv = typicalPrice * data[i].volume;
    
    cumulativeTPV += tpv;
    cumulativeVolume += data[i].volume;
    
    const vwap = cumulativeTPV / cumulativeVolume;
    
    result.push({
      date: data[i].date,
      value: parseFloat(vwap.toFixed(2))
    });
  }
  
  return result;
}

// ========================================
// HELPER FUNCTIONS
// ========================================

/**
 * Calculate all indicators at once
 * @param {Array} data - Stock data array
 * @returns {Object} All calculated indicators
 */
function calculateAllIndicators(data) {
  if (!data || data.length === 0) {
    return {};
  }
  
  return {
    sma20: calculateSMA(data, 20),
    sma50: calculateSMA(data, 50),
    sma200: calculateSMA(data, 200),
    ema12: calculateEMA(data, 12),
    ema26: calculateEMA(data, 26),
    ema50: calculateEMA(data, 50),
    rsi: calculateRSI(data, 14),
    macd: calculateMACD(data, 12, 26, 9),
    bollingerBands: calculateBollingerBands(data, 20, 2),
    atr: calculateATR(data, 14),
    stochastic: calculateStochastic(data, 14, 3, 3),
    adx: calculateADX(data, 14),
    obv: calculateOBV(data),
    vwap: calculateVWAP(data),
    roc: calculateROC(data, 12),
    keltner: calculateKeltnerChannels(data, 20, 2)
  };
}

/**
 * Get indicator interpretation
 * @param {String} indicator - Indicator name
 * @param {Number} value - Indicator value
 * @returns {Object} Interpretation
 */
function interpretIndicator(indicator, value) {
  const interpretations = {
    rsi: (val) => {
      if (val > 70) return { signal: 'sell', message: 'Overbought', strength: 'strong' };
      if (val < 30) return { signal: 'buy', message: 'Oversold', strength: 'strong' };
      if (val > 60) return { signal: 'sell', message: 'Approaching overbought', strength: 'weak' };
      if (val < 40) return { signal: 'buy', message: 'Approaching oversold', strength: 'weak' };
      return { signal: 'hold', message: 'Neutral zone', strength: 'neutral' };
    },
    macd: (val) => {
      if (val.histogram > 0) return { signal: 'buy', message: 'Bullish momentum', strength: 'moderate' };
      if (val.histogram < 0) return { signal: 'sell', message: 'Bearish momentum', strength: 'moderate' };
      return { signal: 'hold', message: 'Neutral momentum', strength: 'neutral' };
    },
    adx: (val) => {
      if (val.adx > 25) return { signal: 'trending', message: 'Strong trend', strength: 'strong' };
      if (val.adx < 20) return { signal: 'ranging', message: 'Weak trend', strength: 'weak' };
      return { signal: 'moderate', message: 'Moderate trend', strength: 'moderate' };
    }
  };
  
  return interpretations[indicator] ? interpretations[indicator](value) : { signal: 'unknown', message: 'No interpretation available' };
}

// ========================================
// EXPORTS
// ========================================

// Make functions available globally
if (typeof window !== 'undefined') {
  window.TechnicalIndicators = {
    // Moving Averages
    calculateSMA,
    calculateEMA,
    calculateWMA,
    
    // Momentum
    calculateRSI,
    calculateStochastic,
    calculateROC,
    
    // Trend
    calculateMACD,
    calculateADX,
    
    // Volatility
    calculateBollingerBands,
    calculateATR,
    calculateKeltnerChannels,
    
    // Volume
    calculateOBV,
    calculateVWAP,
    
    // Helpers
    calculateAllIndicators,
    interpretIndicator
  };
}
