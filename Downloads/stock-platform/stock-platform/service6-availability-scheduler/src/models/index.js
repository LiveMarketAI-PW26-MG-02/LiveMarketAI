const { pool } = require('../config/db');
const moment = require('moment-timezone');

class ExchangeHours {
  static async create(data) {
    const [r] = await pool.execute(
      `INSERT INTO exchange_hours (exchange_name, open_time, close_time, timezone)
       VALUES (?,?,?,?)`,
      [data.exchange_name, data.open_time, data.close_time, data.timezone || 'UTC']
    );
    return this.findById(r.insertId);
  }

  static async findById(id) {
    const [rows] = await pool.execute(`SELECT * FROM exchange_hours WHERE exchange_id=?`, [id]);
    return rows[0] || null;
  }

  static async findAll() {
    const [rows] = await pool.execute(`SELECT * FROM exchange_hours ORDER BY exchange_name`);
    return rows;
  }

  static async update(id, data) {
    const fields = ['exchange_name','open_time','close_time','timezone'];
    const sets=[],params=[];
    for (const k of fields) { if(data[k]!==undefined){sets.push(`${k}=?`);params.push(data[k]);} }
    if (!sets.length) return this.findById(id);
    params.push(id);
    await pool.execute(`UPDATE exchange_hours SET ${sets.join(',')} WHERE exchange_id=?`, params);
    return this.findById(id);
  }

  static async delete(id) {
    const [r] = await pool.execute(`DELETE FROM exchange_hours WHERE exchange_id=?`, [id]);
    return r.affectedRows > 0;
  }

  // Check if a specific exchange is currently open
  static async isOpen(exchange_name) {
    const [rows] = await pool.execute(
      `SELECT * FROM exchange_hours WHERE exchange_name=?`, [exchange_name]
    );
    if (!rows.length) return { is_open: false, reason: 'Exchange not found' };

    const ex = rows[0];
    const now = moment().tz(ex.timezone);
    const day = now.day(); // 0=Sun, 6=Sat

    // Weekends closed
    if (day === 0 || day === 6) {
      return { is_open: false, exchange_name, reason: 'Weekend', current_time: now.format('HH:mm:ss'), timezone: ex.timezone };
    }

    const currentTime = now.format('HH:mm:ss');
    const isOpen = currentTime >= ex.open_time && currentTime <= ex.close_time;

    return {
      is_open:      isOpen,
      exchange_name,
      current_time: currentTime,
      open_time:    ex.open_time,
      close_time:   ex.close_time,
      timezone:     ex.timezone,
      reason:       isOpen ? 'Market is open' : 'Outside trading hours',
    };
  }

  // Check all exchanges
  static async checkAllExchanges() {
    const [rows] = await pool.execute(`SELECT * FROM exchange_hours`);
    return Promise.all(rows.map(ex => this.isOpen(ex.exchange_name)));
  }
}

class StockAvailability {
  static async create(data) {
    const [r] = await pool.execute(
      `INSERT INTO stock_availability (stock_id, available_from, available_until, event_name, status)
       VALUES (?,?,?,?,?)`,
      [data.stock_id, data.available_from, data.available_until, data.event_name||null, data.status||'active']
    );
    return this.findById(r.insertId);
  }

  static async findById(id) {
    const [rows] = await pool.execute(
      `SELECT sa.*, s.stock_symbol, s.stock_name
       FROM stock_availability sa JOIN stocks s ON sa.stock_id=s.stock_id
       WHERE sa.availability_id=?`, [id]
    );
    return rows[0] || null;
  }

  static async findByStockId(stock_id) {
    const [rows] = await pool.execute(
      `SELECT * FROM stock_availability WHERE stock_id=? ORDER BY available_from DESC`, [stock_id]
    );
    return rows;
  }

  static async findAll({ status, page=1, limit=20 }={}) {
    const offset = (page-1)*limit;
    const where  = status ? 'WHERE sa.status=?' : '';
    const params = status ? [status, limit, offset] : [limit, offset];
    const [rows] = await pool.execute(
      `SELECT sa.*, s.stock_symbol, s.stock_name
       FROM stock_availability sa JOIN stocks s ON sa.stock_id=s.stock_id
       ${where} ORDER BY sa.available_from DESC LIMIT ? OFFSET ?`, params
    );
    const [[{total}]] = await pool.execute(
      `SELECT COUNT(*) AS total FROM stock_availability sa ${where}`,
      status ? [status] : []
    );
    return { data: rows, pagination: { total, page, limit, total_pages: Math.ceil(total/limit) }};
  }

  static async update(id, data) {
    const fields = ['available_from','available_until','event_name','status'];
    const sets=[],params=[];
    for (const k of fields) { if(data[k]!==undefined){sets.push(`${k}=?`);params.push(data[k]);} }
    if (!sets.length) return this.findById(id);
    params.push(id);
    await pool.execute(`UPDATE stock_availability SET ${sets.join(',')} WHERE availability_id=?`, params);
    return this.findById(id);
  }

  static async delete(id) {
    const [r] = await pool.execute(`DELETE FROM stock_availability WHERE availability_id=?`, [id]);
    return r.affectedRows > 0;
  }

  // Check if a stock is currently available based on schedules
  static async isStockAvailable(stock_id) {
    const now = new Date();
    const [rows] = await pool.execute(
      `SELECT sa.*, s.stock_symbol, s.stock_name, s.exchange, s.status AS stock_status
       FROM stock_availability sa
       JOIN stocks s ON sa.stock_id = s.stock_id
       WHERE sa.stock_id=? AND sa.status='active'
         AND sa.available_from <= ? AND sa.available_until >= ?
       ORDER BY sa.available_from DESC LIMIT 1`,
      [stock_id, now, now]
    );

    if (!rows.length) {
      // No special schedule — check exchange hours
      const [stockRows] = await pool.execute(
        `SELECT stock_id, stock_symbol, stock_name, exchange, status FROM stocks WHERE stock_id=?`, [stock_id]
      );
      if (!stockRows.length) return { is_available: false, reason: 'Stock not found' };
      const stock = stockRows[0];
      if (stock.status !== 'active') return { is_available: false, reason: `Stock is ${stock.status}` };
      const exchangeStatus = await ExchangeHours.isOpen(stock.exchange);
      return { is_available: exchangeStatus.is_open, stock_symbol: stock.stock_symbol, ...exchangeStatus };
    }

    const schedule = rows[0];
    return {
      is_available:    schedule.status === 'active',
      stock_id,
      stock_symbol:    schedule.stock_symbol,
      stock_name:      schedule.stock_name,
      event_name:      schedule.event_name,
      available_from:  schedule.available_from,
      available_until: schedule.available_until,
      reason:          'Special schedule active',
    };
  }

  // Auto-expire past schedules
  static async expirePastSchedules() {
    const [r] = await pool.execute(
      `UPDATE stock_availability SET status='paused'
       WHERE status='active' AND available_until < NOW()`
    );
    return { expired: r.affectedRows };
  }
}

module.exports = { ExchangeHours, StockAvailability };
