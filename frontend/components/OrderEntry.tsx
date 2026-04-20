
import React, { useState } from 'react';
import { StockData } from '../types';

interface OrderEntryProps {
  selectedStock?: StockData;
}

const OrderEntry: React.FC<OrderEntryProps> = ({ selectedStock }) => {
  const [orderType, setOrderType] = useState<'BUY' | 'SELL'>('BUY');
  const [executionMode, setExecutionMode] = useState<'MARKET' | 'LIMIT' | 'SL' | 'SLM'>('MARKET');
  const [qty, setQty] = useState('1');
  const [price, setPrice] = useState(selectedStock?.price.toString() || '0');
  const [swipeProgress, setSwipeProgress] = useState(0);

  // Sync price when stock changes
  React.useEffect(() => {
    if (selectedStock) setPrice(selectedStock.price.toString());
  }, [selectedStock]);

  const handleSwipe = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setSwipeProgress(val);
    if (val === 100) {
      alert(`Order Placed: ${orderType} ${qty} shares of ${selectedStock?.symbol} @ ${executionMode === 'MARKET' ? 'Market' : price}`);
      setTimeout(() => setSwipeProgress(0), 1000);
    }
  };

  const handleQtyPreset = (multiplier: number) => {
    // Just a mock behavior for quantity presets
    setQty((Math.floor(100 * multiplier)).toString());
  };

  return (
    <div className="p-6 bg-white">
      <h3 className="text-[12px] font-black text-[#222] uppercase tracking-widest mb-4">Execution Panel</h3>
      
      <div className="flex bg-[#F5F5F5] rounded-xl p-1 mb-6">
        <button 
          onClick={() => setOrderType('BUY')}
          className={`flex-1 py-2.5 text-[12px] font-bold rounded-lg transition-all ${orderType === 'BUY' ? 'bg-white text-[#0A8A3A] shadow-sm' : 'text-[#666]'}`}
        >
          BUY
        </button>
        <button 
          onClick={() => setOrderType('SELL')}
          className={`flex-1 py-2.5 text-[12px] font-bold rounded-lg transition-all ${orderType === 'SELL' ? 'bg-white text-[#C0342B] shadow-sm' : 'text-[#666]'}`}
        >
          SELL
        </button>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-6">
        {['MARKET', 'LIMIT', 'SL', 'SLM'].map((m) => (
          <button 
            key={m}
            onClick={() => setExecutionMode(m as any)}
            className={`py-2 text-[9px] font-black rounded-lg border transition-all ${executionMode === m ? 'border-[#222] bg-[#222] text-white shadow-md' : 'border-[#F0F0F0] text-[#999] hover:border-gray-400'}`}
          >
            {m}
          </button>
        ))}
      </div>

      <div className="space-y-4 mb-6">
        <div>
          <label className="text-[10px] font-black text-[#999] mb-1.5 block uppercase tracking-wide">Quantity</label>
          <div className="relative">
            <input 
              type="number" 
              value={qty}
              onChange={(e) => setQty(e.target.value)}
              className="w-full bg-white border border-[#F0F0F0] rounded-xl px-4 py-3 text-sm font-bold focus:outline-none focus:ring-2 focus:ring-black/5 focus:border-black transition-all"
            />
            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-[10px] font-bold text-[#CCC]">SHARES</span>
          </div>
        </div>
        <div>
          <label className="text-[10px] font-black text-[#999] mb-1.5 block uppercase tracking-wide">Price</label>
          <div className="relative">
            <input 
              type="number" 
              disabled={executionMode === 'MARKET'}
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              className="w-full bg-white border border-[#F0F0F0] rounded-xl px-4 py-3 text-sm font-bold focus:outline-none focus:ring-2 focus:ring-black/5 focus:border-black transition-all disabled:opacity-40 disabled:bg-gray-50"
            />
            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-[10px] font-bold text-[#CCC]">USD</span>
          </div>
        </div>
      </div>

      <div className="flex gap-2 mb-8">
        {[0.25, 0.5, 0.75, 1].map(p => (
          <button 
            key={p} 
            onClick={() => handleQtyPreset(p)}
            className="flex-1 py-1.5 text-[10px] font-bold border border-[#F0F0F0] rounded-lg hover:border-[#222] hover:bg-gray-50 text-[#666] transition-all"
          >
            {p * 100}%
          </button>
        ))}
      </div>

      <div className="flex justify-between items-center mb-6 px-1">
        <span className="text-[11px] font-bold text-[#999]">Est. Margin Req.</span>
        <span className="text-sm font-black text-[#222]">${(parseFloat(qty || '0') * parseFloat(price || '0') * 0.1).toFixed(2)}</span>
      </div>

      <div className="relative h-14 rounded-2xl bg-[#F5F5F5] overflow-hidden flex items-center p-1 cursor-pointer">
        <div 
          className={`absolute left-1 top-1 bottom-1 rounded-xl transition-all duration-100 ${orderType === 'BUY' ? 'bg-[#0A8A3A]/20' : 'bg-[#C0342B]/20'}`}
          style={{ width: `calc(${swipeProgress}% - 8px)` }}
        />
        <input 
          type="range"
          min="0"
          max="100"
          value={swipeProgress}
          onChange={handleSwipe}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-30"
        />
        <div className="relative z-10 w-full flex items-center gap-3">
          <div 
            className={`w-12 h-12 rounded-xl flex items-center justify-center text-white shadow-lg transition-transform ${orderType === 'BUY' ? 'bg-[#0A8A3A]' : 'bg-[#C0342B]'}`} 
            style={{ transform: `translateX(${Math.min(swipeProgress * 3, 260)}px)` }}
          >
            <span className="font-bold text-lg">→</span>
          </div>
          {swipeProgress < 40 && (
            <span className="text-[12px] font-black text-[#222] uppercase tracking-widest ml-4">
              Swipe to {orderType}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default OrderEntry;
