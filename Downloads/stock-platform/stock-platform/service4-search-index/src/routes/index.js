const { Router } = require('express');
const r = Router();
const SearchIndex = require('../models/index');

r.get('/health', (_, res) => res.json({ status:'ok', service:'search-index', port: process.env.PORT }));

// Bulk index all stocks from MySQL into Elasticsearch
r.post('/search/reindex', async (_, res) => {
  try {
    const result = await SearchIndex.bulkIndex();
    res.json({ success:true, message:`Indexed ${result.indexed} stocks`, data: result });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Search stocks
r.get('/search', async (req, res) => {
  try {
    const { q, sector, industry, exchange, country, category, status, page=1, limit=20 } = req.query;
    const result = await SearchIndex.search({
      q, sector, industry, exchange, country, category, status,
      page: parseInt(page), limit: Math.min(parseInt(limit), 100),
    });
    res.json({ success:true, ...result });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Index a single stock (called after create/update in catalog service)
r.post('/search/index', async (req, res) => {
  try {
    await SearchIndex.indexStock(req.body);
    res.json({ success:true, message:'Stock indexed successfully' });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Remove a stock from index
r.delete('/search/:stock_id', async (req, res) => {
  try {
    await SearchIndex.deleteStock(req.params.stock_id);
    res.json({ success:true, message:'Stock removed from index' });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

module.exports = r;
