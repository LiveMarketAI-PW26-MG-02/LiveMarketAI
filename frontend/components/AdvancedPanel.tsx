
import React, { useState, useMemo, useEffect } from 'react';
import { ANALYTICS_GROUPS, METRIC_DESCRIPTIONS } from '../constants';

interface AdvancedPanelProps {
  defaultTab?: string;
}

const AdvancedPanel: React.FC<AdvancedPanelProps> = ({ defaultTab }) => {
  const [activeTab, setActiveTab] = useState<string>(defaultTab || ANALYTICS_GROUPS[0].id);
  const [hoveredMetric, setHoveredMetric] = useState<string | null>(null);

  // Sync state if defaultTab prop changes from parent
  useEffect(() => {
    if (defaultTab) {
      setActiveTab(defaultTab);
    }
  }, [defaultTab]);

  const activeGroup = useMemo(() => 
    ANALYTICS_GROUPS.find(g => g.id === activeTab) || ANALYTICS_GROUPS[0]
  , [activeTab]);

  return (
    <div className="flex flex-col h-full bg-white relative">
      <div className="px-8 py-4 border-b border-[#F0F0F0]">
        <div className="flex justify-between items-center mb-3 px-1">
          <h2 className="text-[10px] font-black text-[#666] uppercase tracking-widest">Quant Intelligence Hub</h2>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-[#0A8A3A] animate-pulse"></div>
            <span className="text-[9px] text-[#999] font-black uppercase tracking-tighter">Real-Time Core</span>
          </div>
        </div>
        
        {/* Professional Scrollable Tab Bar */}
        <div className="flex items-center gap-8 overflow-x-auto hide-scrollbar pb-1 -mx-4 px-4 mask-fade-right">
          {ANALYTICS_GROUPS.map((group) => (
            <button
              key={group.id}
              onClick={() => setActiveTab(group.id)}
              className={`whitespace-nowrap pb-2 text-[11px] font-black uppercase tracking-tight transition-all relative ${
                activeTab === group.id 
                ? 'text-[#222]' 
                : 'text-[#999] hover:text-[#666]'
              }`}
            >
              {group.title}
              {activeTab === group.id && (
                <div className="absolute bottom-0 left-0 right-0 h-[2.5px] bg-[#222] animate-in fade-in slide-in-from-bottom-1" />
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-8 space-y-6 hide-scrollbar">
        {/* Intelligence Grid */}
        <div className="bg-white rounded-2xl border border-[#F0F0F0] card-shadow p-8 animate-in fade-in slide-in-from-top-1 duration-300">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-y-10 gap-x-8">
            {Object.entries(activeGroup.metrics).map(([key, value]) => (
              <div 
                key={key} 
                className="relative group cursor-help"
                onMouseEnter={() => setHoveredMetric(key)}
                onMouseLeave={() => setHoveredMetric(null)}
              >
                <div className="flex items-center gap-1.5 mb-1.5">
                  <p className="text-[9px] text-[#999] font-black uppercase truncate tracking-widest leading-none" title={key}>
                    {key.replace(/_/g, ' ')}
                  </p>
                  <div className="w-2.5 h-2.5 rounded-full border border-gray-100 flex items-center justify-center text-[7px] text-gray-300 font-bold group-hover:border-[#222] group-hover:text-[#222] transition-colors">
                    i
                  </div>
                </div>
                <p className="text-[18px] font-black text-[#222] tracking-tighter leading-none">{value}</p>
                
                {hoveredMetric === key && (
                  <div className="absolute z-50 bottom-full left-0 mb-3 w-64 p-5 bg-[#222] text-white rounded-2xl shadow-2xl text-[11px] leading-relaxed font-medium pointer-events-none animate-in fade-in zoom-in-95 duration-200">
                    <p className="font-black text-[#0A8A3A] mb-2 uppercase tracking-widest border-b border-white/10 pb-1.5">Alpha utility</p>
                    {METRIC_DESCRIPTIONS[key] || `High-resolution quantitative indicator for the ${activeGroup.title.toLowerCase()} suite.`}
                    <div className="absolute top-full left-4 w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] border-t-[#222]"></div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="p-8 bg-[#FAFAFA] border border-[#F0F0F0] rounded-2xl border-dashed">
          <div className="flex items-center gap-2 mb-3">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#222" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            <h4 className="text-[11px] font-black text-[#222] uppercase tracking-widest">Protocol Documentation</h4>
          </div>
          <p className="text-[11px] text-[#666] leading-relaxed font-medium max-w-3xl">
            Domain: <b>{activeGroup.title}</b>. These inputs are aggregated from 12 separate Bayesian sub-models. The confidence intervals are recalculated at the tick-level. For institutional execution, cross-reference this tab with the <b>Execution Intelligence</b> module to ensure liquidity depth parity.
          </p>
        </div>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        .mask-fade-right {
          mask-image: linear-gradient(to right, black 85%, transparent 100%);
        }
      `}} />
    </div>
  );
};

export default AdvancedPanel;
