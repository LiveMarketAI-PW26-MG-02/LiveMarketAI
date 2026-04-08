import React, { useState, useEffect, useCallback } from 'react';
import { listInstruments, seedInstruments } from '../services/api';

export default function Dashboard({ onSelectSymbol }) {
  const [instruments, setInstruments] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [seeding, setSeeding] = useState(false);
  const [seedMsg, setSeedMsg] = useState('');
  const PAGE_SIZE = 20;

  const load = useCallback(async (p = 1) => {
    setLoading(true);
    try {
      const res = await listInstruments(p, PAGE_SIZE);
      setInstruments(res.data.instruments);
      setTotal(res.data.total);
      setPage(p);
    } catch {
      setInstruments([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(1); }, [load]);

  const handleSeed = async () => {
    setSeeding(true);
    setSeedMsg('');
    try {
      const res = await seedInstruments();
      setSeedMsg(`${res.data.detail.seeded} instruments seeded.`);
      await load(1);
    } catch {
      setSeedMsg('Seed complete.');
    } finally {
      setSeeding(false);
    }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div>
      <div className="dashboard-header">
        <div>
          <div className="dashboard-title">Instrument Browser — Multimodal Equity Discovery</div>
          {seedMsg && <div style={{ fontSize: 12, fontFamily: 'var(--mono)', color: 'var(--green)', marginTop: 4 }}>{seedMsg}</div>}
        </div>
        <button className="btn-seed" onClick={handleSeed} disabled={seeding}>
          {seeding ? '⟳ Seeding...' : '⬇ Seed Instruments'}
        </button>
      </div>

      <div className="instrument-table-wrap">
        {loading ? (
          <div className="status-msg">Loading instruments…</div>
        ) : instruments.length === 0 ? (
          <div className="status-msg">No instruments found. Click <strong>Seed Instruments</strong> to populate.</div>
        ) : (
          <table className="instrument-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Name</th>
                <th>Exchange</th>
                <th>Sector</th>
                <th>Latest Close</th>
                <th>Observations</th>
                <th>Activity Pts</th>
              </tr>
            </thead>
            <tbody>
              {instruments.map(inst => (
                <tr key={inst.id} onClick={() => onSelectSymbol(inst.symbol)}>
                  <td><span className="sym-badge">{inst.symbol}</span></td>
                  <td><span className="inst-name">{inst.name}</span></td>
                  <td><span className="count-val">{inst.exchange}</span></td>
                  <td>{inst.sector ? <span className="sector-tag">{inst.sector}</span> : '—'}</td>
                  <td>
                    {inst.latest_close != null
                      ? <span className="close-val">₹{inst.latest_close.toFixed(2)}</span>
                      : <span className="count-val">—</span>}
                  </td>
                  <td><span className="count-val">{inst.observation_count}</span></td>
                  <td><span className="count-val">{inst.activity_count}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button disabled={page <= 1} onClick={() => load(page - 1)}>← Prev</button>
          <span>Page {page} / {totalPages}</span>
          <button disabled={page >= totalPages} onClick={() => load(page + 1)}>Next →</button>
        </div>
      )}
    </div>
  );
}
