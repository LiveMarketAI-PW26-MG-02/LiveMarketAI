const { Router } = require('express');
const r = Router();
const { Stock, Category } = require('../models/index');

// Health
r.get('/health', (_, res) => res.json({ status: 'ok', service: 'stock-catalog', port: process.env.PORT }));
r.get('/catalog/stats', async (_, res) => { try { res.json({ success:true, data: await Stock.getStats() }); } catch(e){ res.status(500).json({success:false,message:e.message}); }});

// Stocks
r.post('/stocks', async (req, res) => {
  try {
    const ex = await Stock.findBySymbol(req.body.stock_symbol);
    if (ex) return res.status(409).json({ success:false, message:`Symbol ${req.body.stock_symbol} already exists` });
    const s = await Stock.create(req.body);
    res.status(201).json({ success:true, data:s });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/stocks', async (req, res) => {
  try {
    const { page=1,limit=20,status,category_id,sector,exchange,country,search,sort_by,sort_order } = req.query;
    const result = await Stock.findAll({ page:parseInt(page), limit:Math.min(parseInt(limit),100), status, category_id:category_id?parseInt(category_id):undefined, sector, exchange, country, search, sort_by, sort_order });
    res.json({ success:true, ...result });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/stocks/symbol/:symbol', async (req, res) => {
  try {
    const s = await Stock.findBySymbol(req.params.symbol);
    if (!s) return res.status(404).json({success:false,message:'Stock not found'});
    res.json({success:true,data:s});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.get('/stocks/:id', async (req, res) => {
  try {
    const s = await Stock.findById(req.params.id);
    if (!s) return res.status(404).json({success:false,message:'Stock not found'});
    res.json({success:true,data:s});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.put('/stocks/:id', async (req, res) => {
  try {
    const s = await Stock.update(req.params.id, req.body);
    if (!s) return res.status(404).json({success:false,message:'Stock not found'});
    res.json({success:true,data:s});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

r.delete('/stocks/:id', async (req, res) => {
  try {
    const ok = await Stock.delete(req.params.id);
    if (!ok) return res.status(404).json({success:false,message:'Stock not found'});
    res.json({success:true,message:'Stock delisted'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Categories
r.post('/categories', async (req, res) => {
  try { res.status(201).json({success:true,data:await Category.create(req.body)}); }
  catch(e){ e.code==='ER_DUP_ENTRY'?res.status(409).json({success:false,message:'Category exists'}):res.status(500).json({success:false,message:e.message}); }
});
r.get('/categories', async (_, res) => { try { res.json({success:true,data:await Category.findAll()}); } catch(e){ res.status(500).json({success:false,message:e.message}); }});
r.get('/categories/:id', async (req, res) => {
  try { const c=await Category.findById(req.params.id); c?res.json({success:true,data:c}):res.status(404).json({success:false,message:'Not found'}); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});
r.put('/categories/:id', async (req, res) => {
  try { res.json({success:true,data:await Category.update(req.params.id,req.body)}); }
  catch(e){ res.status(500).json({success:false,message:e.message}); }
});
r.delete('/categories/:id', async (req, res) => {
  try { const ok=await Category.delete(req.params.id); ok?res.json({success:true,message:'Deleted'}):res.status(404).json({success:false,message:'Not found'}); }
  catch(e){ e.code==='ER_ROW_IS_REFERENCED_2'?res.status(409).json({success:false,message:'Has stocks'}):res.status(500).json({success:false,message:e.message}); }
});

module.exports = r;
