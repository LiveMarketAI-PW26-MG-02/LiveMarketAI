
import React, { useState, useMemo } from 'react';
import { ViewState, StockData, Timeframe } from './types';
import { WATCHLIST, MARKET_OVERVIEW } from './constants';
import { HomeIcon, ChartIcon, WatchlistIcon, PortfolioIcon, ChevronDown, ChevronUp } from './components/Icons';
import Sparkline from './components/Sparkline';
import MainChart from './components/MainChart';
import AdvancedPanel from './components/AdvancedPanel';
import OrderEntry from './components/OrderEntry';

interface DashboardProps {
  setView: (view: ViewState) => void;
  setSelectedStock: (stock: StockData | null) => void;
}

const Dashboard: React.FC<DashboardProps> = ({ setView, setSelectedStock }) => {
  const [activeRange, setActiveRange] = useState<Timeframe>('1M');

  return (
    <div className="flex-1 overflow-y-auto p-8 space-y-8 bg-white hide-scrollbar animate-in fade-in duration-500">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-[10px] font-black text-[#999] uppercase tracking-widest mb-1">Total Portfolio Value</h1>
          <div className="flex items-baseline gap-4">
            <span className="text-4xl font-black text-[#222] tracking-tighter">$242,500.00</span>
            <span className="text-[#0A8A3A] text-lg font-bold tracking-tight">+ $4,240.22 (1.2%)</span>
          </div>
        </div>
        <div className="flex gap-3">
          <button onClick={() => setView('PORTFOLIO')} className="px-6 py-2.5 bg-[#222] text-white rounded-xl text-[11px] font-black uppercase tracking-widest shadow-xl hover:bg-black transition-all">Open Vault</button>
          <button className="px-6 py-2.5 border border-[#F0F0F0] text-[#222] rounded-xl text-[11px] font-black uppercase tracking-widest hover:bg-gray-50 transition-all">Export Report</button>
        </div>
      </header>

      <section className="grid grid-cols-3 gap-6">
        {MARKET_OVERVIEW.map((item) => (
          <div key={item.name} className="bg-white p-6 rounded-2xl border border-[#F0F0F0] card-shadow">
            <div className="flex justify-between mb-2">
              <span className="text-[10px] font-black text-[#999] uppercase tracking-widest">{item.name}</span>
              <span className={`text-[10px] font-black ${item.percent >= 0 ? 'text-[#0A8A3A]' : 'text-[#C0342B]'}`}>
                {item.percent > 0 ? '↑' : '↓'} {item.percent}%
              </span>
            </div>
            <div className="text-2xl font-black text-[#222] tracking-tighter">{item.price.toLocaleString()}</div>
          </div>
        ))}
      </section>

      <section className="bg-[#FAFAFA] rounded-2xl p-6 border border-[#F0F0F0]">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-[14px] font-black text-[#222] uppercase tracking-widest">Global Exposure</h2>
          <div className="flex gap-2">
            {['1D', '1M', '3M', '6M', 'YTD', '1Y', '5Y'].map((t) => (
              <button 
                key={t} 
                onClick={() => setActiveRange(t as Timeframe)}
                className={`px-3 py-1 text-[10px] font-black transition-colors ${activeRange === t ? 'text-[#222] border-b-2 border-[#222]' : 'text-[#999] hover:text-[#666]'}`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <div className="h-64 w-full relative min-h-[256px]">
          <MainChart timeframe={activeRange} />
        </div>
      </section>

      <section>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-[14px] font-black text-[#222] uppercase tracking-widest">Active Benchmarks</h2>
          <span className="px-3 py-1 bg-[#FAFAFA] border border-[#F0F0F0] rounded-lg text-[9px] font-black text-[#999] uppercase tracking-widest">Real-time Feed</span>
        </div>
        <div className="grid grid-cols-2 gap-4">
          {WATCHLIST.slice(0, 4).map(stock => (
            <div key={stock.symbol} className="p-4 bg-white border border-[#F0F0F0] rounded-xl card-shadow flex justify-between items-center">
              <div>
                <p className="text-xs font-black">{stock.symbol}</p>
                <p className="text-[10px] text-[#999]">{stock.name}</p>
              </div>
              <div className="text-right">
                <p className="text-xs font-bold">${stock.price}</p>
                <p className={`text-[10px] font-black ${stock.change >= 0 ? 'text-[#0A8A3A]' : 'text-[#C0342B]'}`}>{stock.changePercent}%</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

const VaultStocks: React.FC<{ onSelect: (s: StockData) => void }> = ({ onSelect }) => (
  <div className="flex-1 overflow-y-auto p-8 bg-white hide-scrollbar animate-in fade-in duration-500">
    <div className="mb-8 border-b border-[#F0F0F0] pb-6">
      <h1 className="text-3xl font-black text-[#222] tracking-tighter mb-1">Asset Vault</h1>
      <p className="text-[10px] text-[#999] font-black uppercase tracking-[0.1em]">Select an instrument to initialize the Intelligence Suite</p>
    </div>

    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {WATCHLIST.map(stock => (
        <button 
          key={stock.symbol}
          onClick={() => onSelect(stock)}
          className="p-6 bg-white border border-[#F0F0F0] rounded-2xl text-left hover:border-[#222] hover:shadow-xl transition-all group relative overflow-hidden"
        >
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-xl font-black text-[#222] tracking-tighter">{stock.symbol}</h2>
              <p className="text-[10px] text-[#999] font-bold uppercase tracking-tight truncate w-32">{stock.name}</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-black text-[#222]">${stock.price.toFixed(2)}</p>
              <p className={`text-[10px] font-black ${stock.change >= 0 ? 'text-[#0A8A3A]' : 'text-[#C0342B]'}`}>
                {stock.change >= 0 ? '↑' : '↓'} {stock.changePercent}%
              </p>
            </div>
          </div>
          <div className="h-12 w-full opacity-40 group-hover:opacity-100 transition-opacity">
            <Sparkline data={stock.sparkline} color={stock.change >= 0 ? '#0A8A3A' : '#C0342B'} />
          </div>
          <div className="mt-4 pt-4 border-t border-[#F9F9F9] flex justify-between items-center">
            <span className="text-[9px] font-black text-[#999] uppercase tracking-widest">Select Instrument</span>
            <div className="w-6 h-6 rounded-full bg-gray-50 flex items-center justify-center group-hover:bg-[#222] transition-colors">
              <span className="text-[#222] group-hover:text-white text-xs">→</span>
            </div>
          </div>
        </button>
      ))}
    </div>
  </div>
);

const App: React.FC = () => {
  const [view, setView] = useState<ViewState>('HOME');
  const [selectedStock, setSelectedStock] = useState<StockData | null>(null);
  const [activeTimeframe, setActiveTimeframe] = useState<Timeframe>('1D');

  const handleStockSelection = (stock: StockData) => {
    setSelectedStock(stock);
    setView('STOCK_DETAIL');
  };

  const NavItem = ({ id, icon: Icon, label, disabled = false }: { id: ViewState, icon?: any, label: string, disabled?: boolean }) => {
    const isActive = view === id;
    return (
      <button 
        disabled={disabled}
        onClick={() => setView(id)}
        className={`w-full flex items-center gap-3 px-4 py-2 rounded-lg transition-all text-[11px] font-bold relative ${
          isActive 
            ? 'bg-white shadow-sm text-[#222] ring-1 ring-black/5' 
            : disabled 
              ? 'text-[#CCC] cursor-not-allowed opacity-60'
              : 'text-[#666] hover:bg-gray-100 hover:text-[#222]'
        }`}
      >
        {Icon && <Icon className="w-3.5 h-3.5" />}
        <span className="truncate tracking-tight">{label}</span>
      </button>
    );
  };

  const IntelligenceWorkspace = (tabId: string, title: string, subtitle: string) => {
    if (!selectedStock) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center bg-white p-12 text-center animate-in fade-in duration-500">
           <div className="w-20 h-20 bg-[#FAFAFA] rounded-3xl flex items-center justify-center mb-6 shadow-sm border border-[#F0F0F0]">
             <PortfolioIcon className="text-gray-300 w-8 h-8" />
           </div>
           <h2 className="text-xl font-black text-[#222] tracking-tighter mb-2">Quant Context Missing</h2>
           <p className="text-[11px] text-[#999] font-black uppercase tracking-[0.1em] mb-8 max-w-xs">You must select an instrument from the Asset Vault to initialize this intelligence module.</p>
           <button 
             onClick={() => setView('PORTFOLIO')}
             className="px-8 py-3 bg-[#222] text-white rounded-xl text-[11px] font-black uppercase tracking-widest shadow-xl hover:bg-black transition-all"
           >
             Open Vault
           </button>
        </div>
      );
    }

    return (
      <div className="flex-1 flex flex-col overflow-hidden bg-white animate-in slide-in-from-right-4 duration-500">
        <div className="p-8 border-b border-[#F0F0F0] flex justify-between items-center">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h1 className="text-3xl font-black text-[#222] tracking-tighter">{title}</h1>
              <div className="px-2 py-0.5 bg-[#222] rounded-md text-[9px] text-white font-black uppercase tracking-widest">{selectedStock.symbol}</div>
            </div>
            <p className="text-[10px] text-[#999] font-black uppercase tracking-[0.1em]">Institutional Analytics: {subtitle}</p>
          </div>
          <div className="flex gap-4">
            <div className="text-right">
              <p className="text-[9px] font-black text-[#999] uppercase tracking-widest">Protocol Status</p>
              <p className="text-xs font-black text-[#0A8A3A]">VERIFIED</p>
            </div>
            <div className="h-8 w-[1px] bg-gray-100" />
            <div className="text-right">
              <p className="text-[9px] font-black text-[#999] uppercase tracking-widest">Decision Confidence</p>
              <p className="text-xs font-black text-[#222]">92.4%</p>
            </div>
          </div>
        </div>
        <div className="flex-1 overflow-hidden">
          <AdvancedPanel defaultTab={tabId} />
        </div>
      </div>
    );
  };

  const isIntelLocked = !selectedStock;

  return (
    <div className="h-screen w-screen flex bg-[#F6F8FA]">
      <aside className="w-64 border-r border-[#F0F0F0] bg-[#FAFAFA] flex flex-col p-4 z-20 overflow-y-auto hide-scrollbar">
        <div className="flex items-center gap-3 px-4 py-6 mb-4">
          <div className="w-9 h-9 bg-[#222] rounded-xl flex items-center justify-center shadow-xl">
            <div className="w-4 h-4 border-2 border-white rounded-sm rotate-45" />
          </div>
          <span className="font-black text-xl tracking-tighter">LiveMarketAI</span>
        </div>
        
        <div className="space-y-6 flex-1">
          <section>
            <h3 className="text-[9px] font-black text-[#CCC] uppercase tracking-widest px-4 mb-2">Core Operations</h3>
            <nav className="space-y-0.5">
              <NavItem id="HOME" icon={HomeIcon} label="Dashboard" />
              <NavItem id="PORTFOLIO" icon={PortfolioIcon} label="Vault (Selection)" />
              <NavItem id="TRADES" icon={WatchlistIcon} label="Activity" />
            </nav>
          </section>

          <section>
            <h3 className="text-[9px] font-black text-[#CCC] uppercase tracking-widest px-4 mb-2">Decision Intel</h3>
            <nav className="space-y-0.5">
              <NavItem id="DECISION_TIMELINE" label="Decision Timeline" disabled={isIntelLocked} />
              <NavItem id="PORTFOLIO_INTEL" label="Portfolio Intelligence" disabled={isIntelLocked} />
              <NavItem id="STRATEGY_LIFE" label="Strategy Lifecycle" disabled={isIntelLocked} />
              <NavItem id="OBJECTIVE_CTRL" label="Objective Control" disabled={isIntelLocked} />
              <NavItem id="ALERT_INTEL" label="Alert Intelligence" disabled={isIntelLocked} />
              <NavItem id="DECISION_COMPARATOR" label="Decision Comparator" disabled={isIntelLocked} />
            </nav>
          </section>

          <section>
            <h3 className="text-[9px] font-black text-[#CCC] uppercase tracking-widest px-4 mb-2">Operational Suite</h3>
            <nav className="space-y-0.5">
              <NavItem id="SYSTEM_TRUST" label="System Trust Panel" disabled={isIntelLocked} />
              <NavItem id="DATA_INTEGRITY" label="Data Integrity Monitor" disabled={isIntelLocked} />
              <NavItem id="MARKET_INTERCONNECT" label="Market Interconnect" disabled={isIntelLocked} />
              <NavItem id="HUMAN_OVERSIGHT" label="Human Oversight" disabled={isIntelLocked} />
              <NavItem id="PRODUCTION_READINESS" label="Production Readiness" disabled={isIntelLocked} />
            </nav>
          </section>

          <section>
            <h3 className="text-[9px] font-black text-[#CCC] uppercase tracking-widest px-4 mb-2">Quant Labs</h3>
            <nav className="space-y-0.5">
              <NavItem id="CONFIDENCE" label="Confidence Engine" disabled={isIntelLocked} />
              <NavItem id="REGIME" label="Regime Intelligence" disabled={isIntelLocked} />
              <NavItem id="SCENARIO" label="Scenario Studio" disabled={isIntelLocked} />
              <NavItem id="CAUSALITY" label="Causality Lab" disabled={isIntelLocked} />
              <NavItem id="FORWARD_RISK" label="Forward Risk Lab" disabled={isIntelLocked} />
            </nav>
          </section>
        </div>

        {selectedStock && (
          <section className="mt-8 pt-6 border-t border-gray-200/60 pb-4">
            <div className="flex justify-between items-center px-4 mb-3">
              <h3 className="text-[9px] font-black text-[#CCC] uppercase tracking-widest">Active Focus</h3>
              <span className="w-1.5 h-1.5 rounded-full bg-[#0A8A3A] animate-pulse" />
            </div>
            <button 
              onClick={() => setView('STOCK_DETAIL')}
              className="w-full flex items-center justify-between px-4 py-3 bg-white shadow-sm ring-1 ring-[#F0F0F0] rounded-xl transition-all"
            >
              <p className="text-[11px] font-black text-[#222] tracking-tighter">{selectedStock.symbol}</p>
              <p className={`text-[10px] font-black ${selectedStock.change >= 0 ? 'text-[#0A8A3A]' : 'text-[#C0342B]'}`}>
                 {selectedStock.changePercent}%
              </p>
            </button>
          </section>
        )}
      </aside>

      <main className="flex-1 flex flex-col overflow-hidden relative">
        {view === 'HOME' && <Dashboard setView={setView} setSelectedStock={setSelectedStock} />}
        
        {view === 'PORTFOLIO' && <VaultStocks onSelect={handleStockSelection} />}

        {view === 'STOCK_DETAIL' && selectedStock && (
          <div className="flex-1 flex overflow-hidden animate-in fade-in duration-300">
            <div className="flex-1 overflow-y-auto hide-scrollbar bg-white flex flex-col">
              <div className="p-6 border-b border-[#F0F0F0] flex justify-between items-center sticky top-0 bg-white z-10 glass-panel">
                <div className="flex items-center gap-6">
                  <div>
                    <div className="flex items-center gap-2">
                      <h1 className="text-2xl font-black text-[#222] tracking-tighter">{selectedStock.symbol}</h1>
                      <span className="px-2 py-0.5 bg-[#F5F5F5] rounded text-[9px] font-black text-[#666] uppercase tracking-tight">Focus Node</span>
                    </div>
                    <p className="text-xs text-[#999] font-medium">{selectedStock.name}</p>
                  </div>
                  <div className="h-10 w-[1px] bg-[#F0F0F0]" />
                  <div>
                    <div className="text-2xl font-black text-[#222] tracking-tighter">${selectedStock.price}</div>
                    <div className={`text-xs font-bold ${selectedStock.change >= 0 ? 'text-[#0A8A3A]' : 'text-[#C0342B]'}`}>
                      {selectedStock.change >= 0 ? '+' : ''}{selectedStock.change} ({selectedStock.changePercent}%)
                    </div>
                  </div>
                </div>

                <div className="flex gap-2">
                  {['1D', '1M', '3M', '6M', 'YTD', '1Y', '5Y'].map(tf => (
                    <button 
                      key={tf} 
                      onClick={() => setActiveTimeframe(tf as Timeframe)}
                      className={`px-4 py-1.5 text-[11px] font-black uppercase rounded-xl border transition-all ${activeTimeframe === tf ? 'bg-[#222] text-white border-[#222] shadow-lg' : 'bg-white text-[#999] border-[#F0F0F0] hover:border-[#999]'}`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>

              <div className="p-6 flex-1 flex flex-col min-h-[450px]">
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center gap-4">
                    <span className="text-[10px] font-black text-[#999] uppercase tracking-widest">Instrument Terminal</span>
                    <div className="h-4 w-[1px] bg-gray-200" />
                    <span className="text-[9px] font-black text-[#0A8A3A] uppercase tracking-tighter">Mode: Institutional</span>
                  </div>
                </div>
                <div className="flex-1 min-h-[300px] relative">
                  <MainChart timeframe={activeTimeframe} />
                </div>
              </div>

              <div className="p-6 border-t border-[#F0F0F0] bg-[#FAFAFA]">
                <div className="grid grid-cols-4 gap-6">
                  {[{ l: 'SMA_10', v: '186.20' }, { l: 'SMA_50', v: '182.45' }, { l: 'EMA_12', v: '187.05' }, { l: 'EMA_26', v: '184.90' }, { l: 'RSI_14', v: '64.20' }, { l: 'MACD', v: '1.25' }, { l: 'Signal', v: '0.85' }, { l: 'BB_Position', v: '0.78' }].map(i => (
                    <div key={i.l} className="bg-white p-5 rounded-2xl border border-[#F0F0F0] card-shadow">
                      <p className="text-[9px] font-black text-[#999] uppercase mb-1 tracking-widest">{i.l}</p>
                      <p className="text-sm font-black text-[#222]">{i.v}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="w-[380px] border-l border-[#F0F0F0] bg-white overflow-y-auto hide-scrollbar">
              <OrderEntry selectedStock={selectedStock} />
              <div className="border-t border-[#F0F0F0]">
                <AdvancedPanel />
              </div>
            </div>
          </div>
        )}
        
        {/* Decision Intel Modules */}
        {view === 'DECISION_TIMELINE' && IntelligenceWorkspace('DECISION_TIMELINE', 'Decision Timeline', 'Lifecycle and state awareness')}
        {view === 'PORTFOLIO_INTEL' && IntelligenceWorkspace('PORTFOLIO_INTEL', 'Portfolio Intelligence', 'Exposure and risk decomposition')}
        {view === 'STRATEGY_LIFE' && IntelligenceWorkspace('STRATEGY_LIFE', 'Strategy Lifecycle', 'Signal decay management')}
        {view === 'OBJECTIVE_CTRL' && IntelligenceWorkspace('OBJECTIVE_CTRL', 'Objective Control', 'Intent encoding and optimization')}
        {view === 'ALERT_INTEL' && IntelligenceWorkspace('ALERT_INTEL', 'Alert Intelligence', 'Operational alerts')}
        {view === 'DECISION_COMPARATOR' && IntelligenceWorkspace('DECISION_COMPARATOR', 'Decision Comparator', 'Comparative logic')}

        {/* Operational Suite Modules */}
        {view === 'SYSTEM_TRUST' && IntelligenceWorkspace('SYSTEM_TRUST', 'System Trust Panel', 'Latency monitoring')}
        {view === 'DATA_INTEGRITY' && IntelligenceWorkspace('DATA_INTEGRITY', 'Data Integrity Monitor', 'Source verification')}
        {view === 'MARKET_INTERCONNECT' && IntelligenceWorkspace('MARKET_INTERCONNECT', 'Market Interconnect', 'Cross-asset coupling')}
        {view === 'HUMAN_OVERSIGHT' && IntelligenceWorkspace('HUMAN_OVERSIGHT', 'Human Oversight', 'Accountability tracking')}
        {view === 'PRODUCTION_READINESS' && IntelligenceWorkspace('PRODUCTION_READINESS', 'Production Readiness', 'Operational health')}

        {/* Quant Labs Modules */}
        {view === 'CONFIDENCE' && IntelligenceWorkspace('CONFIDENCE', 'Confidence Engine', 'Probabilistic quantification')}
        {view === 'REGIME' && IntelligenceWorkspace('REGIME', 'Regime Intelligence', 'Dynamic classification')}
        {view === 'SCENARIO' && IntelligenceWorkspace('SCENARIO', 'Scenario Studio', 'Shock modeling')}
        {view === 'CAUSALITY' && IntelligenceWorkspace('CAUSALITY', 'Causality Lab', 'Causal attribution')}
        {view === 'FORWARD_RISK' && IntelligenceWorkspace('FORWARD_RISK', 'Forward Risk Lab', 'Institutional risk infrastructure')}

        {view === 'TRADES' && (
           <div className="flex-1 flex flex-col items-center justify-center bg-white text-center p-8 animate-in zoom-in-95 duration-500">
             <div className="w-16 h-16 bg-gray-50 rounded-2xl flex items-center justify-center mb-4">
                <WatchlistIcon className="text-gray-300 w-8 h-8" />
             </div>
             <p className="text-[11px] font-black text-[#222] uppercase tracking-widest mb-1">Transaction Ledger</p>
             <p className="text-[10px] text-[#999] font-medium">Historical trade logs and execution reports.</p>
           </div>
        )}
      </main>
    </div>
  );
};

export default App;
