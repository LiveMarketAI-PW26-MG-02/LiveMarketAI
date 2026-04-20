const { pool } = require('../config/db');

class Order {

  // ── Step 10: Pre-execution verification ──────────────────────────────────
  // Before placing order — verify stock exists and has a valid price
  static async verify(stock_id) {
    const [stocks] = await pool.execute(
      `SELECT s.stock_id, s.stock_symbol, s.stock_name, s.status,
              sp.adjusted_price AS closing_price
       FROM stocks s
       LEFT JOIN stock_prices sp ON s.stock_id = sp.stock_id
       WHERE s.stock_id = ?`,
      [stock_id]
    );
    if (!stocks.length)              return { valid: false, reason: 'Stock not found' };
    if (stocks[0].status !== 'active') return { valid: false, reason: `Stock is ${stocks[0].status}` };
    if (!stocks[0].closing_price)    return { valid: false, reason: 'No price available for this stock' };
    return { valid: true, stock: stocks[0] };
  }

  // ── Step 1: Create order ──────────────────────────────────────────────────
  // Saves closing_price + quantity + position in one record
  static async create({ investor_id, stock_id, quantity }) {
    // Verify stock before placing
    const check = await this.verify(stock_id);
    if (!check.valid) throw new Error(check.reason);

    // Get next position number for this investor
    const [[{ last_position }]] = await pool.execute(
      `SELECT COALESCE(MAX(position), 0) AS last_position
       FROM orders WHERE investor_id = ?`,
      [investor_id]
    );
    const position     = last_position + 1;
    const closing_price = check.stock.closing_price;

    const [result] = await pool.execute(
      `INSERT INTO orders (investor_id, stock_id, closing_price, quantity, position, status)
       VALUES (?, ?, ?, ?, ?, 'confirmed')`,
      [investor_id, stock_id, closing_price, quantity, position]
    );
    return this.findById(result.insertId);
  }

  // ── Step 3: Get all orders for investor ───────────────────────────────────
  static async findAllByInvestor(investor_id) {
    const [rows] = await pool.execute(
      `SELECT o.order_id, o.investor_id, o.stock_id,
              s.stock_symbol, s.stock_name,
              o.closing_price, o.quantity, o.position,
              o.status, o.created_at
       FROM orders o
       JOIN stocks s ON o.stock_id = s.stock_id
       WHERE o.investor_id = ?
       ORDER BY o.position ASC`,
      [investor_id]
    );
    return rows;
  }

  // ── Find single order by ID ───────────────────────────────────────────────
  static async findById(order_id) {
    const [rows] = await pool.execute(
      `SELECT o.*, s.stock_symbol, s.stock_name
       FROM orders o
       JOIN stocks s ON o.stock_id = s.stock_id
       WHERE o.order_id = ?`,
      [order_id]
    );
    return rows[0] || null;
  }

  // ── Step 4: Total capital spent by investor ───────────────────────────────
  // Sum of (closing_price × quantity) across all confirmed orders
  static async getTotalCapital(investor_id) {
    const [[result]] = await pool.execute(
      `SELECT
         COUNT(*)                                   AS total_orders,
         COALESCE(SUM(closing_price * quantity), 0) AS total_capital,
         COALESCE(SUM(quantity), 0)                 AS total_units
       FROM orders
       WHERE investor_id = ? AND status = 'confirmed'`,
      [investor_id]
    );
    return {
      investor_id:   parseInt(investor_id),
      total_orders:  parseInt(result.total_orders),
      total_units:   parseInt(result.total_units),
      total_capital: parseFloat(result.total_capital),
    };
  }

  // ── Step 5 & 8: Total units held per stock ────────────────────────────────
  // Combines quantity of all confirmed orders for same stock
  static async getHoldingsPerStock(investor_id) {
    const [rows] = await pool.execute(
      `SELECT
         o.stock_id,
         s.stock_symbol,
         s.stock_name,
         SUM(o.quantity)                       AS total_quantity,
         SUM(o.closing_price * o.quantity)     AS total_invested,
         AVG(o.closing_price)                  AS avg_price
       FROM orders o
       JOIN stocks s ON o.stock_id = s.stock_id
       WHERE o.investor_id = ? AND o.status = 'confirmed'
       GROUP BY o.stock_id, s.stock_symbol, s.stock_name
       ORDER BY total_quantity DESC`,
      [investor_id]
    );
    return rows.map(r => ({
      ...r,
      total_quantity:  parseInt(r.total_quantity),
      total_invested:  parseFloat(r.total_invested),
      avg_price:       parseFloat(r.avg_price),
    }));
  }

  // ── Step 6: Orders sorted by position ─────────────────────────────────────
  static async getOrdersSortedByPosition(investor_id) {
    const [rows] = await pool.execute(
      `SELECT o.order_id, o.stock_id, s.stock_symbol, s.stock_name,
              o.closing_price, o.quantity, o.position, o.status, o.created_at
       FROM orders o
       JOIN stocks s ON o.stock_id = s.stock_id
       WHERE o.investor_id = ?
       ORDER BY o.position ASC`,
      [investor_id]
    );
    return rows;
  }

  // ── Step 7: Paginated order history ───────────────────────────────────────
  static async getPaginated(investor_id, { page = 1, limit = 10 } = {}) {
    const offset = (page - 1) * limit;
    const [rows] = await pool.execute(
      `SELECT o.order_id, o.stock_id, s.stock_symbol, s.stock_name,
              o.closing_price, o.quantity, o.position, o.status, o.created_at
       FROM orders o
       JOIN stocks s ON o.stock_id = s.stock_id
       WHERE o.investor_id = ?
       ORDER BY o.position ASC
       LIMIT ? OFFSET ?`,
      [investor_id, limit, offset]
    );
    const [[{ total }]] = await pool.execute(
      `SELECT COUNT(*) AS total FROM orders WHERE investor_id = ?`,
      [investor_id]
    );
    return {
      data: rows,
      pagination: {
        total,
        page,
        limit,
        total_pages: Math.ceil(total / limit),
        has_next:    page * limit < total,
        has_prev:    page > 1,
      },
    };
  }

  // ── Step 11: Order activity — count of orders per stock ───────────────────
  static async getActivityPerStock(investor_id) {
    const [rows] = await pool.execute(
      `SELECT
         o.stock_id,
         s.stock_symbol,
         s.stock_name,
         COUNT(o.order_id) AS order_count
       FROM orders o
       JOIN stocks s ON o.stock_id = s.stock_id
       WHERE o.investor_id = ? AND o.status = 'confirmed'
       GROUP BY o.stock_id, s.stock_symbol, s.stock_name
       ORDER BY order_count DESC`,
      [investor_id]
    );
    return rows.map(r => ({ ...r, order_count: parseInt(r.order_count) }));
  }

  // ── Step 12: Summary — quantity + order count per stock ───────────────────
  static async getSummary(investor_id) {
    const [rows] = await pool.execute(
      `SELECT
         o.stock_id,
         s.stock_symbol,
         s.stock_name,
         SUM(o.quantity)    AS total_quantity,
         COUNT(o.order_id)  AS order_count
       FROM orders o
       JOIN stocks s ON o.stock_id = s.stock_id
       WHERE o.investor_id = ? AND o.status = 'confirmed'
       GROUP BY o.stock_id, s.stock_symbol, s.stock_name
       ORDER BY total_quantity DESC`,
      [investor_id]
    );
    return rows.map(r => ({
      ...r,
      total_quantity: parseInt(r.total_quantity),
      order_count:    parseInt(r.order_count),
    }));
  }

  // ── Step 13: Full portfolio — price + quantity + order count ──────────────
  static async getPortfolio(investor_id) {
    const [rows] = await pool.execute(
      `SELECT
         o.stock_id,
         s.stock_symbol,
         s.stock_name,
         s.sector,
         s.exchange,
         AVG(o.closing_price)              AS avg_closing_price,
         SUM(o.quantity)                   AS total_quantity,
         COUNT(o.order_id)                 AS order_count,
         SUM(o.closing_price * o.quantity) AS total_invested
       FROM orders o
       JOIN stocks s ON o.stock_id = s.stock_id
       WHERE o.investor_id = ? AND o.status = 'confirmed'
       GROUP BY o.stock_id, s.stock_symbol, s.stock_name, s.sector, s.exchange
       ORDER BY total_invested DESC`,
      [investor_id]
    );
    return rows.map(r => ({
      ...r,
      avg_closing_price: parseFloat(r.avg_closing_price),
      total_quantity:    parseInt(r.total_quantity),
      order_count:       parseInt(r.order_count),
      total_invested:    parseFloat(r.total_invested),
    }));
  }
}

module.exports = Order;
