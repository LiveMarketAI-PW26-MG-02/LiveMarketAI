const { Router } = require('express');
const r = Router();
const PriceCache = require('../models/index');

r.get('/health', (_, res) => res.json({ status:'ok', service:'price-cache', port: process.env.PORT }));

// Warm up cache from DB
r.post('/cache/warmup', async (_, res) => {
  try {
    const result = await PriceCache.warmUp();
    res.json({ success:true, message:`Cache warmed for ${result.warmed} stocks`, data:result });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Get all cached prices
r.get('/cache', async (_, res) => {
  try {
    const data = await PriceCache.getAll();
    res.json({ success:true, count: data.length, data });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Get one stock price from cache
r.get('/cache/:stock_id', async (req, res) => {
  try {
    const data = await PriceCache.get(req.params.stock_id);
    data ? res.json({success:true,data}) : res.status(404).json({success:false,message:'Not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Set / update price in cache
r.put('/cache/:stock_id', async (req, res) => {
  try {
    const data = await PriceCache.set(req.params.stock_id, req.body);
    res.json({ success:true, message:'Cache updated', data });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Invalidate one stock's cache
r.delete('/cache/:stock_id', async (req, res) => {
  try {
    await PriceCache.invalidate(req.params.stock_id);
    res.json({ success:true, message:'Cache invalidated' });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Check TTL for a stock
r.get('/cache/:stock_id/ttl', async (req, res) => {
  try {
    const ttl = await PriceCache.getTTL(req.params.stock_id);
    res.json({ success:true, stock_id: req.params.stock_id, ttl_seconds: ttl });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

module.exports = r;
