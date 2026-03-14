const redis = require('../config/redis');
const { pool } = require('../config/db');

const CACHE_TTL = 60; // seconds
const KEY = (stock_id) => `stock:price:${stock_id}`;

class PriceCache {

  // Set price in Redis cache
  static async set(stock_id, data) {
    const payload = {
      stock_id:        stock_id,
      current_price:   data.current_price,
      volume:          data.volume          || 0,
      day_high:        data.day_high        || data.current_price,
      day_low:         data.day_low         || data.current_price,
      last_trade_time: data.last_trade_time || new Date().toISOString(),
      cache_updated_at: new Date().toISOString(),
    };
    await redis.setex(KEY(stock_id), CACHE_TTL, JSON.stringify(payload));
    return payload;
  }

  // Get from cache — fall back to DB if miss
  static async get(stock_id) {
    const cached = await redis.get(KEY(stock_id));
    if (cached) {
      const data = JSON.parse(cached);
      data._source = 'cache';
      return data;
    }
    // Cache miss — load from stock_prices table
    const [rows] = await pool.execute(
      `SELECT sp.stock_id, sp.adjusted_price AS current_price, sp.last_updated AS last_trade_time,
              s.stock_symbol, s.stock_name
       FROM stock_prices sp JOIN stocks s ON sp.stock_id=s.stock_id
       WHERE sp.stock_id=?`, [stock_id]);
    if (!rows.length) return null;
    const row = rows[0];
    const data = await this.set(stock_id, {
      current_price:   parseFloat(row.current_price),
      volume:          0,
      day_high:        parseFloat(row.current_price),
      day_low:         parseFloat(row.current_price),
      last_trade_time: row.last_trade_time,
    });
    data.stock_symbol = row.stock_symbol;
    data.stock_name   = row.stock_name;
    data._source = 'db';
    return data;
  }

  // Get all cached prices
  static async getAll() {
    const keys = await redis.keys('stock:price:*');
    if (!keys.length) return [];
    const values = await redis.mget(...keys);
    return values.filter(Boolean).map(v => JSON.parse(v));
  }

  // Delete from cache
  static async invalidate(stock_id) {
    await redis.del(KEY(stock_id));
    return true;
  }

  // Bulk load all stock prices into cache from DB
  static async warmUp() {
    const [rows] = await pool.execute(
      `SELECT sp.stock_id, sp.adjusted_price AS current_price, sp.last_updated AS last_trade_time
       FROM stock_prices sp JOIN stocks s ON sp.stock_id=s.stock_id WHERE s.status='active'`);
    for (const row of rows) {
      await this.set(row.stock_id, {
        current_price:   parseFloat(row.current_price),
        last_trade_time: row.last_trade_time,
      });
    }
    return { warmed: rows.length };
  }

  // TTL check
  static async getTTL(stock_id) {
    return redis.ttl(KEY(stock_id));
  }
}

module.exports = PriceCache;
