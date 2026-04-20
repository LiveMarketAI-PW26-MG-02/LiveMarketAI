import { StockData, MetricGroup } from './types';

export const WATCHLIST: StockData[] = [
  { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 894.20, change: 15.45, changePercent: 1.76, volume: '44.8M', bid: 894.15, ask: 894.25, open: 880.00, high: 900.50, low: 878.00, prevClose: 878.75, sparkline: [880, 885, 890, 888, 895, 894.20] },
  { symbol: 'MSFT', name: 'Microsoft Corp.', price: 425.22, change: 3.10, changePercent: 0.73, volume: '22.1M', bid: 425.18, ask: 425.25, open: 422.10, high: 426.50, low: 421.80, prevClose: 422.12, sparkline: [422, 423, 425, 424, 426, 425.22] },
  { symbol: 'AAPL', name: 'Apple Inc.', price: 189.45, change: 2.34, changePercent: 1.25, volume: '52.4M', bid: 189.40, ask: 189.50, open: 187.10, high: 190.20, low: 186.50, prevClose: 187.11, sparkline: [187, 188, 187.5, 189, 188.5, 189.45] },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 178.15, change: -1.22, changePercent: -0.68, volume: '35.2M', bid: 178.10, ask: 178.20, open: 179.30, high: 180.10, low: 177.50, prevClose: 179.37, sparkline: [179, 180, 178.5, 179, 177.8, 178.15] },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 154.20, change: 0.85, changePercent: 0.55, volume: '28.4M', bid: 154.15, ask: 154.25, open: 153.10, high: 155.20, low: 152.80, prevClose: 153.35, sparkline: [153, 154, 153.5, 154.5, 153.8, 154.20] },
  { symbol: 'META', name: 'Meta Platforms Inc.', price: 485.30, change: 5.45, changePercent: 1.14, volume: '18.9M', bid: 485.20, ask: 485.40, open: 480.10, high: 488.20, low: 479.50, prevClose: 479.85, sparkline: [480, 482, 485, 483, 487, 485.30] },
  { symbol: 'AVGO', name: 'Broadcom Inc.', price: 1245.50, change: 12.30, changePercent: 1.00, volume: '4.2M', bid: 1245.10, ask: 1245.90, open: 1233.10, high: 1250.20, low: 1230.50, prevClose: 1233.20, sparkline: [1233, 1238, 1242, 1240, 1248, 1245.50] },
  { symbol: 'TSLA', name: 'Tesla Inc.', price: 175.22, change: -4.12, changePercent: -2.30, volume: '102M', bid: 175.20, ask: 175.25, open: 179.30, high: 180.10, low: 174.50, prevClose: 179.34, sparkline: [179, 178, 176, 177, 175, 175.22] },
  { symbol: 'LLY', name: 'Eli Lilly and Co.', price: 765.40, change: 8.20, changePercent: 1.08, volume: '3.8M', bid: 765.20, ask: 765.60, open: 757.10, high: 768.20, low: 755.50, prevClose: 757.20, sparkline: [757, 760, 763, 762, 766, 765.40] },
  { symbol: 'JPM', name: 'JPMorgan Chase & Co.', price: 198.12, change: 1.45, changePercent: 0.74, volume: '12.4M', bid: 198.05, ask: 198.15, open: 196.10, high: 199.20, low: 195.50, prevClose: 196.67, sparkline: [196, 197, 198, 197.5, 198.5, 198.12] },
];

export const MARKET_OVERVIEW = [
  { name: 'S&P 500', price: 5222.68, change: 25.40, percent: 0.49 },
  { name: 'NASDAQ', price: 16175.45, change: -12.30, percent: -0.08 },
  { name: 'DOW JONES', price: 39131.53, change: 156.10, percent: 0.40 },
];

export const METRIC_DESCRIPTIONS: { [key: string]: string } = {
  Sharpe: "Sharpe Ratio: Measures risk-adjusted return relative to a risk-free rate.",
  VaR: "Value at Risk: Estimating the maximum potential loss over a timeframe.",
  CVaR: "Conditional Value at Risk: Expected loss given that the loss exceeds the VaR threshold.",
  MCR: "Marginal Contribution to Risk: Amount of total risk contributed by an individual position.",
  Drift: "Feature Drift: Deviation of model inputs from training distribution.",
  Latency: "Data Freshness: Milliseconds since last L1/L2 update sync.",
  Feasibility: "Execution Feasibility: Probability of filling desired size at current depth.",
  Outcome_Bias: "Post-Trade Analysis: Statistical measure of selection vs luck in signal outcome.",
  Alpha_HalfLife: "Measure of the time it takes for a predictive signal to lose 50% of its information edge.",
};

export const ANALYTICS_GROUPS: MetricGroup[] = [
  {
    id: 'PORTFOLIO_INTEL',
    title: 'Portfolio Intelligence',
    metrics: {
      Exposure_Aggregate: '$24.2M',
      Sector_Concentration: 'Tech (42%)',
      Factor_Beta: '1.14',
      Marginal_Risk_Contrib: '0.04',
      Capital_Efficiency: '0.88',
      Tail_Risk_Agg: '-4.2%'
    }
  },
  {
    id: 'SCENARIO',
    title: 'Scenario Studio',
    metrics: {
      Rate_Shock_100bp: '-2.4%',
      Fed_Pivot_Impact: '+5.2%',
      CPI_Miss_Exp: '-1.8%',
      Counterfactual_Vol: '24.2',
      Regime_Stress: 'Pass',
      Tail_Event_Prob: '0.04'
    }
  },
  {
    id: 'STRATEGY_LIFE',
    title: 'Strategy Lifecycle',
    metrics: {
      Health_Score: '0.92',
      Signal_Decay: 'Linear',
      Alpha_HalfLife: '4.2h',
      Overfit_P_Value: '0.002',
      BT_Live_Divergence: 'Minimal',
      Deprecation_Flag: 'Green'
    }
  },
  {
    id: 'DATA_INTEGRITY',
    title: 'Data Integrity Monitor',
    metrics: {
      Freshness_Ms: '12ms',
      Lineage_Path: 'Direct -> L1',
      Feature_Drift: '0.02',
      Anomalous_Input: '0',
      Trust_Score: '0.99',
      Sync_Stability: 'High'
    }
  },
  {
    id: 'OBJECTIVE_CTRL',
    title: 'Objective Control',
    metrics: {
      Risk_Appetite: 'Growth',
      Time_Horizon: 'Medium',
      Constraint_Set: 'Standard',
      Target_Yield: '8.4%',
      Drawdown_Cap: '-10%',
      ESG_Alignment: '0.84'
    }
  },
  {
    id: 'DECISION_TIMELINE',
    title: 'Decision Timeline',
    metrics: {
      Current_State: 'Active',
      Lifecycle_Stage: 'Execution',
      Expiry_T_Minus: '14m 22s',
      Regime_Validity: 'Bull_Phase_1',
      Approval_Time: '09:42:01',
      Observed_Outcome: 'Pending'
    }
  },
  {
    id: 'ALERT_INTEL',
    title: 'Alert Intelligence',
    metrics: {
      Confidence_Alert: 'Enabled',
      Regime_Drift: 'Low',
      Feasibility_Warning: 'None',
      Priority_Status: 'P1-Critical',
      Degradation_Flag: 'Clear',
      Signal_Noise_Ratio: '42.2'
    }
  },
  {
    id: 'DECISION_COMPARATOR',
    title: 'Decision Comparator',
    metrics: {
      Selected_Alpha: '0.12',
      Alt_B_Alpha: '0.08',
      Choice_Justify: 'Risk-Adj',
      Attribution_Delta: '0.04',
      Opportunity_Cost: '-$1,240',
      Logic_Path: 'Bayesian'
    }
  },
  {
    id: 'CONTEXTUAL_MEMORY',
    title: 'Contextual Memory',
    metrics: {
      Historical_Notes: 'Regime_Sync',
      Past_Outcome_ID: '7F-44',
      Learned_Bias: '-0.02',
      Context_Weight: '0.15',
      Annotation_Count: '12',
      Memory_Persistence: 'High'
    }
  },
  {
    id: 'MODEL_CONTROL',
    title: 'Model Control',
    metrics: {
      Active_Version: 'v5.4.1',
      Last_Rollback: 'N/A',
      Perf_Lift: '+2.4%',
      Stability_Index: '0.98',
      Decision_Link: 'MRS-Agg',
      Drift_Threshold: '0.05'
    }
  },
  {
    id: 'SYSTEM_TRUST',
    title: 'System Trust Panel',
    metrics: {
      L1_Latency: '8ms',
      Source_Sync: '99.9%',
      Trust_Badge: 'Institutional',
      Data_Staleness: 'Minimal',
      Vendor_Health: 'Nominal',
      Degraded_Mode: 'Inactive'
    }
  },
  {
    id: 'MARKET_INTERCONNECT',
    title: 'Market Interconnect',
    metrics: {
      Lead_Lag_Ratio: '1.24',
      Contagion_Idx: '0.14',
      Spillover_Risk: 'Low',
      Cross_Asset_C: '0.82',
      Coupling_Trend: 'Static',
      Macro_Sync: '0.64'
    }
  },
  {
    id: 'HUMAN_OVERSIGHT',
    title: 'Human Oversight',
    metrics: {
      Override_Status: 'Inactive',
      Last_Manual_ID: 'U-001',
      Reason_Capture: 'Enabled',
      Human_Accuracy: '88%',
      Model_Delta: '0.02',
      Accountability: 'High'
    }
  },
  {
    id: 'PRODUCTION_READINESS',
    title: 'Production Readiness',
    metrics: {
      System_Uptime: '99.99%',
      Ready_Score: '0.96',
      Kill_Switch: 'SAFE',
      Data_Pipeline: 'Healthy',
      Exec_Readiness: 'High',
      Model_Auth: 'Verified'
    }
  },
  {
    id: 'CONFIDENCE',
    title: 'Confidence Engine',
    metrics: {
      Signal_Confidence: '92%',
      Model_Entropy: '0.08',
      Certainty_Ratio: '0.84',
      Probability_Band: '±1.2%',
      Noise_Floor: '-42dB',
      Validation_Score: '0.96'
    }
  }
];