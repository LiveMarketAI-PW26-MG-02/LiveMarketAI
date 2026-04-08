import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, AreaChart, Area,
  BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { getMultimodalProfile, getEnrichedProfile, getMarketDepth } from '../services/api';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border-bright)',
      padding: '8px 12px', borderRadius: 4, fontFamily: 'var(--mono)', fontSize: 12
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || 'var(--accent)' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
};

function sentimentClass(s) {
  if (!s) return 'neutral';
  const l = s.toLowerCase();
  if (l.includes('bull')) return 'bull';
  if (l.includes('bear')) return 'bear';
  return 'neutral';
}

export default function InstrumentDetail({ symbol }) {
  const [profile, setProfile] = useState(null);
  const [enriched, setEnriched] = useState(null);
  const [depth, setDepth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setProfile(null); setEnriched(null); setDepth(null);

    Promise.all([
      getMultimodalProfile(symbol),
      getEnrichedProfile(symbol),
      getMarketDepth(symbol),
    ]).then(([pRes, eRes, dRes]) => {
      setProfile(pRes.data);
      setEnriched(eRes.data);
      setDepth(dRes.data);
    }).catch(() => {}).finally(() => setLoading(false));
  }, [symbol]);

  if (loading) return <div className="status-msg accent">Assembling multimodal profile for {symbol}…</div>;
  if (!profile) return <div className="status-msg">Profile not available for {symbol}.</div>;

  // Build chart data — align by sequence_ordinal
  const depth_n = profile.profile_depth;
  const prices = profile.closing_price_sequence;
  const times = profile.time_index_sequence;
  const activities = profile.activity_frequency_stream;

  const priceChartData = prices.map((p, i) => ({
    idx: p.sequence_ordinal,
    close: p.closing_price,
    label: p.observed_at ? new Date(p.observed_at).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }) : `T${p.sequence_ordinal}`,
  }));

  // First differences of closing prices
  const diffChartData = prices.slice(1).map((p, i) => ({
    idx: p.sequence_ordinal,
    diff: parseFloat((p.closing_price - prices[i].closing_price).toFixed(4)),
    label: p.observed_at ? new Date(p.observed_at).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }) : `T${p.sequence_ordinal}`,
  }));

  const actChartData = activities.map((a) => ({
    idx: a.sequence_ordinal,
    freq: a.frequency_count,
    interval: parseFloat(a.interval_seconds.toFixed(2)),
    label: a.recorded_at ? new Date(a.recorded_at).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }) : `T${a.sequence_ordinal}`,
  }));

  const timeChartData = times.map((t) => ({
    idx: t.sequence_ordinal,
    marker: t.time_marker,
    label: `T${t.sequence_ordinal}`,
  }));

  const analysis = enriched?.analysis || {};
  const videos = enriched?.related_videos || [];

  return (
    <div>
      {/* Header */}
      <div className="detail-header">
        <div className="detail-symbol">{symbol}</div>
        <div className="detail-name">{profile.name}</div>
        <div className="detail-meta">
          <span className="meta-chip">{profile.exchange}</span>
          {profile.sector && <span className="meta-chip">{profile.sector}</span>}
          <span className="meta-chip green">Depth: {profile.profile_depth} observations</span>
          <span className="meta-chip">Assembled: {new Date(profile.assembled_at).toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Mistral Analysis */}
      {analysis.sentiment && (
        <div className="analysis-panel">
          <div className="analysis-title">◈ AI Multimodal Analysis</div>
          <div className="analysis-row">
            <div className="analysis-item">
              <label>SENTIMENT</label>
              <div className={`val ${sentimentClass(analysis.sentiment)}`}>{analysis.sentiment}</div>
            </div>
            <div className="analysis-item">
              <label>CONFIDENCE</label>
              <div className="val" style={{ color: 'var(--accent)' }}>{(analysis.confidence * 100).toFixed(0)}%</div>
            </div>
            <div className="analysis-item" style={{ flex: 1 }}>
              <label>OUTLOOK</label>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2 }}>{analysis.outlook}</div>
            </div>
          </div>
        </div>
      )}

      {/* Market Depth */}
      {depth && (
        <div className="depth-panel">
          <div className="depth-item"><label>BID</label><div className="dval" style={{ color: 'var(--green)' }}>₹{depth.bid?.toFixed(2)}</div></div>
          <div className="depth-item"><label>ASK</label><div className="dval" style={{ color: 'var(--red)' }}>₹{depth.ask?.toFixed(2)}</div></div>
          <div className="depth-item"><label>SPREAD</label><div className="dval">₹{depth.spread?.toFixed(4)}</div></div>
          <div className="depth-item"><label>BID QTY</label><div className="dval">{depth.bid_qty?.toLocaleString()}</div></div>
          <div className="depth-item"><label>ASK QTY</label><div className="dval">{depth.ask_qty?.toLocaleString()}</div></div>
        </div>
      )}

      {/* Charts Grid */}
      <div className="charts-grid">
        {/* Dimension 1 — Closing Price Sequence */}
        <div className="chart-card">
          <div className="chart-card-title"><span>◈</span>Dim-1 — Closing Price Sequence</div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={priceChartData}>
              <defs>
                <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4af0c4" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#4af0c4" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis dataKey="label" tick={{ fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'var(--mono)' }} interval={14} />
              <YAxis tick={{ fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'var(--mono)' }} width={60} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="close" name="Close" stroke="#4af0c4" fill="url(#priceGrad)" strokeWidth={1.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Dimension 1 derived — First Differences */}
        <div className="chart-card">
          <div className="chart-card-title"><span>◈</span>Dim-1Δ — First Difference Stream</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={diffChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis dataKey="label" tick={{ fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'var(--mono)' }} interval={14} />
              <YAxis tick={{ fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'var(--mono)' }} width={60} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="diff" name="Δ Close" fill="#7b6cfa" radius={[2, 2, 0, 0]}
                label={false}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Dimension 2 — Time Index Sequence */}
        <div className="chart-card">
          <div className="chart-card-title"><span>◈</span>Dim-2 — Time Index Sequence</div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={timeChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis dataKey="label" tick={{ fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'var(--mono)' }} interval={14} />
              <YAxis tick={{ fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'var(--mono)' }} width={80}
                tickFormatter={v => new Date(v * 1000).toLocaleDateString('en-IN', { month: 'short', year: '2-digit' })} />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="marker" name="Time Marker" stroke="#f5c842" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Dimension 3 — Activity Frequency Stream */}
        <div className="chart-card">
          <div className="chart-card-title"><span>◈</span>Dim-3 — Activity Frequency Stream</div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={actChartData}>
              <defs>
                <linearGradient id="actGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f05474" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#f05474" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis dataKey="label" tick={{ fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'var(--mono)' }} interval={14} />
              <YAxis tick={{ fontSize: 9, fill: 'var(--text-muted)', fontFamily: 'var(--mono)' }} width={60} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="freq" name="Freq Count" stroke="#f05474" fill="url(#actGrad)" strokeWidth={1.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* YouTube Research Videos */}
      {videos.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>
            ◈ Research Videos
          </div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {videos.map((v, i) => (
              <a
                key={i}
                href={`https://youtube.com/watch?v=${v.video_id}`}
                target="_blank"
                rel="noreferrer"
                style={{
                  display: 'block', width: 260,
                  background: 'var(--bg-panel)', border: '1px solid var(--border)',
                  borderRadius: 6, padding: '12px 14px',
                  textDecoration: 'none', transition: 'border-color 0.15s',
                }}
                onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accent2)'}
                onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
              >
                <div style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 500, marginBottom: 6, lineHeight: 1.4 }}>{v.title}</div>
                <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-muted)' }}>{v.channel}</div>
                {v.view_count > 0 && (
                  <div style={{ fontSize: 10, fontFamily: 'var(--mono)', color: 'var(--text-muted)', marginTop: 4 }}>
                    {v.view_count.toLocaleString()} views
                  </div>
                )}
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
