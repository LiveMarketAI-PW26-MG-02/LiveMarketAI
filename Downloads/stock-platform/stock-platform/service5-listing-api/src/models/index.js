const { pool } = require('../config/db');
const redis    = require('../config/redis');

class StockListing {

  // Main listing — joins stocks + prices, enriches with Redis cache
  static async list({ page=1, limit=20, sector, exchange, country, category_id, status='active', sort_by='stock_name', sort_order='ASC' }={}) {
    const offset = (page-1)*limit;
    const conds  = ['s.status=?'];
    const params = [status];

    if (sector)      { conds.push('s.sector=?');      params.push(sector); }
    if (exchange)    { conds.push('s.exchange=?');    params.push(exchange); }
    if (country)     { conds.push('s.country=?');     params.push(country); }
    if (category_id) { conds.push('s.category_id=?'); params.push(category_id); }

    const where   = `WHERE ${conds.join(' AND ')}`;
    const allowed = ['stock_name','stock_symbol','market_cap','s.created_at'];
    const safe    = allowed.includes(sort_by) ? sort_by : 'stock_name';
    const order   = sort_order.toUpperCase()==='DESC' ? 'DESC' : 'ASC';

    // Fetch from MySQL — catalog + base price
    const [rows] = await pool.execute(
      `SELECT
         s.stock_id, s.stock_symbol, s.stock_name,
         s.sector,   s.exchange,     s.country,
         s.market_cap, s.currency, s.status,
         c.category_name AS category,
         sp.base_price, sp.adjusted_price
       FROM stocks s
       LEFT JOIN stock_categories c  ON s.category_id  = c.category_id
       LEFT JOIN stock_prices      sp ON s.stock_id     = sp.stock_id
       ${where}
       ORDER BY ${safe} ${order}
       LIMIT ? OFFSET ?`,
      [...params, limit, offset]
    );

    const [[{ total }]] = await pool.execute(
      `SELECT COUNT(*) AS total FROM stocks s ${where}`, params
    );

    // Enrich each row with live price from Redis cache
    const data = await Promise.all(rows.map(async (row) => {
      let current_price = row.adjusted_price || row.base_price;
      try {
        const cached = await redis.get(`stock:price:${row.stock_id}`);
        if (cached) {
          const c = JSON.parse(cached);
          current_price = c.current_price;
        }
      } catch(_) { /* cache miss is fine */ }

      // Return exactly the API Response fields from your spec
      return {
        stock_id:      row.stock_id,
        stock_symbol:  row.stock_symbol,
        stock_name:    row.stock_name,
        sector:        row.sector,
        category:      row.category,
        current_price: parseFloat(current_price || 0),
        market_cap:    row.market_cap ? parseFloat(row.market_cap) : null,
        exchange:      row.exchange,
      };
    }));

    return {
      data,
      pagination: {
        total,
        page,
        limit,
        total_pages: Math.ceil(total / limit),
        has_next:    page * limit < total,
        has_prev:    page > 1,
      }
    };
  }

  // Single stock detail — full data
  static async getOne(stock_id) {
    const [rows] = await pool.execute(
      `SELECT s.*, c.category_name AS category, sp.base_price, sp.adjusted_price, sp.last_updated
       FROM stocks s
       LEFT JOIN stock_categories c  ON s.category_id = c.category_id
       LEFT JOIN stock_prices      sp ON s.stock_id    = sp.stock_id
       WHERE s.stock_id = ?`, [stock_id]
    );
    if (!rows.length) return null;
    const row = rows[0];

    let current_price = row.adjusted_price || row.base_price;
    let cache_data = {};
    try {
      const cached = await redis.get(`stock:price:${stock_id}`);
      if (cached) { cache_data = JSON.parse(cached); current_price = cache_data.current_price; }
    } catch(_) {}

    return {
      stock_id:      row.stock_id,
      stock_symbol:  row.stock_symbol,
      stock_name:    row.stock_name,
      sector:        row.sector,
      industry:      row.industry,
      exchange:      row.exchange,
      currency:      row.currency,
      country:       row.country,
      isin:          row.isin,
      category:      row.category,
      description:   row.description,
      market_cap:    row.market_cap ? parseFloat(row.market_cap) : null,
      ipo_date:      row.ipo_date,
      status:        row.status,
      tags:          row.tags ? (typeof row.tags==='string'?JSON.parse(row.tags):row.tags) : [],
      current_price: parseFloat(current_price || 0),
      day_high:      cache_data.day_high  || null,
      day_low:       cache_data.day_low   || null,
      volume:        cache_data.volume    || null,
      last_trade_time: cache_data.last_trade_time || null,
    };
  }

  // Filter options — unique values for dropdowns
  static async getFilters() {
    const [sectors]   = await pool.execute(`SELECT DISTINCT sector   FROM stocks WHERE sector   IS NOT NULL AND status='active' ORDER BY sector`);
    const [exchanges] = await pool.execute(`SELECT DISTINCT exchange FROM stocks WHERE exchange IS NOT NULL AND status='active' ORDER BY exchange`);
    const [countries] = await pool.execute(`SELECT DISTINCT country  FROM stocks WHERE country  IS NOT NULL AND status='active' ORDER BY country`);
    const [cats]      = await pool.execute(`SELECT category_id, category_name FROM stock_categories ORDER BY category_name`);
    return {
      sectors:    sectors.map(r=>r.sector),
      exchanges:  exchanges.map(r=>r.exchange),
      countries:  countries.map(r=>r.country),
      categories: cats,
    };
  }
}

module.exports = StockListing;
