const { pool } = require('../config/db');

class StockPrice {
  static async upsert(stock_id, base_price, currency = 'USD') {
    await pool.execute(
      `INSERT INTO stock_prices (stock_id, base_price, adjusted_price, currency)
       VALUES (?,?,?,?)
       ON DUPLICATE KEY UPDATE base_price=VALUES(base_price), currency=VALUES(currency)`,
      [stock_id, base_price, base_price, currency]
    );
    return this.findByStockId(stock_id);
  }

  static async findByStockId(stock_id) {
    const [rows] = await pool.execute(
      `SELECT sp.*, s.stock_symbol, s.stock_name
       FROM stock_prices sp
       JOIN stocks s ON sp.stock_id = s.stock_id
       WHERE sp.stock_id = ?`, [stock_id]);
    return rows[0] || null;
  }

  static async findAll({ page=1, limit=20 }={}) {
    const offset = (page-1)*limit;
    const [rows] = await pool.execute(
      `SELECT sp.*, s.stock_symbol, s.stock_name
       FROM stock_prices sp JOIN stocks s ON sp.stock_id=s.stock_id
       ORDER BY sp.last_updated DESC LIMIT ? OFFSET ?`, [limit, offset]);
    const [[{total}]] = await pool.execute(`SELECT COUNT(*) AS total FROM stock_prices`);
    return { data: rows, pagination: { total, page, limit, total_pages: Math.ceil(total/limit) }};
  }

  // Apply all active rules to a stock's base price
  static async applyRules(stock_id) {
    const price = await this.findByStockId(stock_id);
    if (!price) return null;

    const now = new Date();
    const [rules] = await pool.execute(
      `SELECT * FROM price_adjustment_rules
       WHERE status='active' AND valid_from <= ? AND valid_to >= ?
       ORDER BY rule_id ASC`, [now, now]);

    let adjusted = parseFloat(price.base_price);

    for (const rule of rules) {
      const val = parseFloat(rule.adjustment_value);
      if (rule.adjustment_type === 'discount') {
        adjusted = adjusted - (adjusted * val / 100);
      } else if (rule.adjustment_type === 'multiplier') {
        adjusted = adjusted * val;
      } else if (rule.adjustment_type === 'override') {
        adjusted = val;
      }
    }

    adjusted = Math.max(0, parseFloat(adjusted.toFixed(4)));
    await pool.execute(`UPDATE stock_prices SET adjusted_price=? WHERE stock_id=?`, [adjusted, stock_id]);
    return this.findByStockId(stock_id);
  }

  // Apply rules to ALL stocks
  static async applyRulesToAll() {
    const [stocks] = await pool.execute(`SELECT stock_id FROM stocks WHERE status='active'`);
    for (const s of stocks) await this.applyRules(s.stock_id);
    return { updated: stocks.length };
  }
}

class PriceRule {
  static async create(data) {
    const [r] = await pool.execute(
      `INSERT INTO price_adjustment_rules (rule_name,adjustment_type,adjustment_value,valid_from,valid_to,status)
       VALUES (?,?,?,?,?,?)`,
      [data.rule_name, data.adjustment_type, data.adjustment_value, data.valid_from, data.valid_to, data.status||'active']);
    return this.findById(r.insertId);
  }

  static async findById(id) {
    const [rows] = await pool.execute(`SELECT * FROM price_adjustment_rules WHERE rule_id=?`, [id]);
    return rows[0]||null;
  }

  static async findAll({ status }={}) {
    const where = status ? `WHERE status=?` : '';
    const params = status ? [status] : [];
    const [rows] = await pool.execute(`SELECT * FROM price_adjustment_rules ${where} ORDER BY valid_from DESC`, params);
    return rows;
  }

  static async findActive() {
    const now = new Date();
    const [rows] = await pool.execute(
      `SELECT * FROM price_adjustment_rules WHERE status='active' AND valid_from<=? AND valid_to>=?`, [now, now]);
    return rows;
  }

  static async update(id, data) {
    const fields = ['rule_name','adjustment_type','adjustment_value','valid_from','valid_to','status'];
    const sets=[],params=[];
    for (const k of fields) { if(data[k]!==undefined){sets.push(`${k}=?`);params.push(data[k]);} }
    if (!sets.length) return this.findById(id);
    params.push(id);
    await pool.execute(`UPDATE price_adjustment_rules SET ${sets.join(',')} WHERE rule_id=?`, params);
    return this.findById(id);
  }

  static async delete(id) {
    const [r] = await pool.execute(`DELETE FROM price_adjustment_rules WHERE rule_id=?`, [id]);
    return r.affectedRows > 0;
  }
}

module.exports = { StockPrice, PriceRule };
