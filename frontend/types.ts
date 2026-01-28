
export type Timeframe = '1D' | '1M' | '3M' | '6M' | 'YTD' | '1Y' | '5Y';

export interface StockData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: string;
  bid: number;
  ask: number;
  open: number;
  high: number;
  low: number;
  prevClose: number;
  sparkline: number[];
}

export interface MetricGroup {
  id: string;
  title: string;
  metrics: { [key: string]: string | number };
}

export type ViewState = 
  | 'HOME' 
  | 'STOCK_DETAIL' 
  | 'ANALYTICS' 
  | 'PORTFOLIO' 
  | 'TRADES'
  | 'CONFIDENCE'
  | 'CAUSALITY'
  | 'EXPLAINABILITY'
  | 'ADAPTIVE'
  | 'REGIME'
  | 'FORWARD_RISK'
  | 'EXECUTION'
  | 'ORCHESTRATOR'
  | 'EVOLUTION'
  // Institutional Infrastructure
  | 'PORTFOLIO_INTEL'
  | 'SCENARIO'
  | 'STRATEGY_LIFE'
  | 'DATA_INTEGRITY'
  | 'CAPITAL_GOV'
  | 'DECISION_LEDGER'
  | 'OBJECTIVE_CTRL'
  | 'AGENT_HUB'
  | 'RESEARCH_MEM'
  // Operational & Decision Intelligence
  | 'DECISION_TIMELINE'
  | 'ALERT_INTEL'
  | 'DECISION_COMPARATOR'
  | 'CONTEXTUAL_MEMORY'
  | 'MODEL_CONTROL'
  | 'SYSTEM_TRUST'
  | 'MARKET_INTERCONNECT'
  | 'HUMAN_OVERSIGHT'
  | 'PRODUCTION_READINESS';
