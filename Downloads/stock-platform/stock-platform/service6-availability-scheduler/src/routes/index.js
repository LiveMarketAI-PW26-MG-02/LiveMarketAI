const { Router } = require('express');
const r = Router();
const { ExchangeHours, StockAvailability } = require('../models/index');

r.get('/health', (_, res) => res.json({ status:'ok', service:'availability-scheduler', port: process.env.PORT }));

// ── Exchange Hours ────────────────────────────────────────────────────────────
r.post('/exchanges', async (req, res) => {
  try { res.status(201).json({ success:true, data: await ExchangeHours.create(req.body) }); }
  catch(e){ e.code==='ER_DUP_ENTRY' ? res.status(409).json({success:false,message:'Exchange already exists'}) : res.status(500).json({success:false,message:e.message}); }
});

r.get('/exchanges', async (_, res) => {
  try { res.json({ success:true, data: await ExchangeHours.findAll() }); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/exchanges/status', async (_, res) => {
  try { res.json({ success:true, data: await ExchangeHours.checkAllExchanges() }); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/exchanges/:id', async (req, res) => {
  try {
    const ex = await ExchangeHours.findById(req.params.id);
    ex ? res.json({success:true,data:ex}) : res.status(404).json({success:false,message:'Not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/exchanges/:name/is-open', async (req, res) => {
  try { res.json({ success:true, data: await ExchangeHours.isOpen(req.params.name) }); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.put('/exchanges/:id', async (req, res) => {
  try { res.json({ success:true, data: await ExchangeHours.update(req.params.id, req.body) }); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.delete('/exchanges/:id', async (req, res) => {
  try {
    const ok = await ExchangeHours.delete(req.params.id);
    ok ? res.json({success:true,message:'Deleted'}) : res.status(404).json({success:false,message:'Not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// ── Stock Availability ────────────────────────────────────────────────────────
r.post('/availability', async (req, res) => {
  try { res.status(201).json({ success:true, data: await StockAvailability.create(req.body) }); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/availability', async (req, res) => {
  try {
    const result = await StockAvailability.findAll({ status: req.query.status, page: parseInt(req.query.page||1), limit: parseInt(req.query.limit||20) });
    res.json({ success:true, ...result });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/availability/stock/:stock_id', async (req, res) => {
  try { res.json({ success:true, data: await StockAvailability.findByStockId(req.params.stock_id) }); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/availability/stock/:stock_id/check', async (req, res) => {
  try { res.json({ success:true, data: await StockAvailability.isStockAvailable(req.params.stock_id) }); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/availability/:id', async (req, res) => {
  try {
    const a = await StockAvailability.findById(req.params.id);
    a ? res.json({success:true,data:a}) : res.status(404).json({success:false,message:'Not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.put('/availability/:id', async (req, res) => {
  try { res.json({ success:true, data: await StockAvailability.update(req.params.id, req.body) }); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.delete('/availability/:id', async (req, res) => {
  try {
    const ok = await StockAvailability.delete(req.params.id);
    ok ? res.json({success:true,message:'Deleted'}) : res.status(404).json({success:false,message:'Not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

module.exports = r;
