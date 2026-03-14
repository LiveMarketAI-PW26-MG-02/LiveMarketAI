const { Router } = require('express');
const r = Router();
const { StockPrice, PriceRule } = require('../models/index');

r.get('/health', (_, res) => res.json({ status:'ok', service:'price-adjustment-engine', port: process.env.PORT }));

// Stock Prices
r.post('/prices', async (req, res) => {
  try {
    const p = await StockPrice.upsert(req.body.stock_id, req.body.base_price, req.body.currency);
    res.status(201).json({ success:true, data:p });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/prices', async (req, res) => {
  try {
    const result = await StockPrice.findAll({ page:parseInt(req.query.page||1), limit:parseInt(req.query.limit||20) });
    res.json({ success:true, ...result });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/prices/:stock_id', async (req, res) => {
  try {
    const p = await StockPrice.findByStockId(req.params.stock_id);
    p ? res.json({success:true,data:p}) : res.status(404).json({success:false,message:'Not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Apply rules to one stock
r.post('/prices/:stock_id/apply-rules', async (req, res) => {
  try {
    const p = await StockPrice.applyRules(req.params.stock_id);
    p ? res.json({success:true,message:'Rules applied',data:p}) : res.status(404).json({success:false,message:'Stock price not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Apply rules to ALL stocks
r.post('/prices/apply-rules/all', async (req, res) => {
  try {
    const result = await StockPrice.applyRulesToAll();
    res.json({ success:true, message:`Rules applied to ${result.updated} stocks`, data:result });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Rules CRUD
r.post('/rules', async (req, res) => {
  try { res.status(201).json({success:true,data:await PriceRule.create(req.body)}); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/rules', async (req, res) => {
  try { res.json({success:true,data:await PriceRule.findAll({status:req.query.status})}); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/rules/active', async (_, res) => {
  try { res.json({success:true,data:await PriceRule.findActive()}); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/rules/:id', async (req, res) => {
  try {
    const rule = await PriceRule.findById(req.params.id);
    rule ? res.json({success:true,data:rule}) : res.status(404).json({success:false,message:'Rule not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.put('/rules/:id', async (req, res) => {
  try { res.json({success:true,data:await PriceRule.update(req.params.id,req.body)}); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.delete('/rules/:id', async (req, res) => {
  try {
    const ok = await PriceRule.delete(req.params.id);
    ok ? res.json({success:true,message:'Rule deleted'}) : res.status(404).json({success:false,message:'Not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

module.exports = r;
