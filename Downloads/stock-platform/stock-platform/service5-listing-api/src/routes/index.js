const { Router } = require('express');
const r = Router();
const StockListing = require('../models/index');

r.get('/health', (_, res) => res.json({ status:'ok', service:'stock-listing-api', port: process.env.PORT }));

// Main stock listing — paginated
r.get('/listings', async (req, res) => {
  try {
    const { page=1, limit=20, sector, exchange, country, category_id, status, sort_by, sort_order } = req.query;
    const result = await StockListing.list({
      page:        parseInt(page),
      limit:       Math.min(parseInt(limit), 100),
      sector, exchange, country,
      category_id: category_id ? parseInt(category_id) : undefined,
      status:      status || 'active',
      sort_by, sort_order,
    });
    res.json({ success:true, ...result });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Single stock full detail
r.get('/listings/:stock_id', async (req, res) => {
  try {
    const data = await StockListing.getOne(req.params.stock_id);
    data ? res.json({success:true,data}) : res.status(404).json({success:false,message:'Stock not found'});
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

// Filter options for frontend dropdowns
r.get('/listings/meta/filters', async (_, res) => {
  try {
    res.json({ success:true, data: await StockListing.getFilters() });
  } catch(e){ res.status(500).json({success:false,message:e.message}); }
});

module.exports = r;
