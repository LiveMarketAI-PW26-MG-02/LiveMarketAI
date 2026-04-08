import React, { useState } from 'react';
import Dashboard from './pages/Dashboard';
import InstrumentDetail from './pages/InstrumentDetail';
import './App.css';

export default function App() {
  const [selectedSymbol, setSelectedSymbol] = useState(null);

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <span className="brand-sigil">◈</span>
            <span className="brand-name">AAAI</span>
            <span className="brand-sub">Multimodal Equity Discovery</span>
          </div>
          {selectedSymbol && (
            <button className="btn-back" onClick={() => setSelectedSymbol(null)}>
              ← Instrument Browser
            </button>
          )}
        </div>
      </header>
      <main className="app-main">
        {selectedSymbol ? (
          <InstrumentDetail symbol={selectedSymbol} onBack={() => setSelectedSymbol(null)} />
        ) : (
          <Dashboard onSelectSymbol={setSelectedSymbol} />
        )}
      </main>
    </div>
  );
}
