const { pool } = require('../config/db');

class Stock {
  static async create(data) {
    const sql = `INSERT INTO stocks (stock_symbol,stock_name,category_id,description,sector,industry,exchange,currency,isin,market_cap,ipo_date,country,tags,status)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)`;
    const [r] = await pool.execute(sql, [
      data.stock_symbol.toUpperCase(), data.stock_name, data.category_id||null,
      data.description||null, data.sector||null, data.industry||null,
      data.exchange||null, data.currency||'USD', data.isin||null,
      data.market_cap||null, data.ipo_date||null, data.country||null,
      data.tags ? JSON.stringify(data.tags) : null, data.status||'active'
    ]);
    return this.findById(r.insertId);
  }

  static async findById(id) {
    const [rows] = await pool.execute(
      `SELECT s.*, c.category_name FROM stocks s LEFT JOIN stock_categories c ON s.category_id=c.category_id WHERE s.stock_id=?`, [id]);
    return rows[0] ? this._parse(rows[0]) : null;
  }

  static async findBySymbol(symbol) {
    const [rows] = await pool.execute(
      `SELECT s.*, c.category_name FROM stocks s LEFT JOIN stock_categories c ON s.category_id=c.category_id WHERE s.stock_symbol=?`, [symbol.toUpperCase()]);
    return rows[0] ? this._parse(rows[0]) : null;
  }

  static async findAll({ page=1, limit=20, status, category_id, sector, exchange, country, search, sort_by='created_at', sort_order='DESC' }={}) {
    const offset = (page-1)*limit;
    const conds = [], params = [];
    if (status)      { conds.push('s.status=?');      params.push(status); }
    if (category_id) { conds.push('s.category_id=?'); params.push(category_id); }
    if (sector)      { conds.push('s.sector=?');      params.push(sector); }
    if (exchange)    { conds.push('s.exchange=?');    params.push(exchange); }
    if (country)     { conds.push('s.country=?');     params.push(country); }
    if (search) {
      conds.push('(s.stock_name LIKE ? OR s.stock_symbol LIKE ? OR s.sector LIKE ?)');
      const q=`%${search}%`; params.push(q,q,q);
    }
    const where = conds.length ? `WHERE ${conds.join(' AND ')}` : '';
    const safe = ['stock_name','stock_symbol','market_cap','ipo_date','created_at'].includes(sort_by) ? sort_by : 'created_at';
    const order = sort_order.toUpperCase()==='ASC'?'ASC':'DESC';
    const [rows] = await pool.execute(
      `SELECT s.*, c.category_name FROM stocks s LEFT JOIN stock_categories c ON s.category_id=c.category_id ${where} ORDER BY s.${safe} ${order} LIMIT ? OFFSET ?`,
      [...params, limit, offset]);
    const [[{total}]] = await pool.execute(`SELECT COUNT(*) AS total FROM stocks s ${where}`, params);
    return { data: rows.map(this._parse.bind(this)), pagination: { total, page, limit, total_pages: Math.ceil(total/limit), has_next: page*limit<total, has_prev: page>1 }};
  }

  static async update(id, data) {
    const fields = ['stock_symbol','stock_name','category_id','description','sector','industry','exchange','currency','isin','market_cap','ipo_date','country','tags','status'];
    const sets=[],params=[];
    for (const k of fields) {
      if (data[k]===undefined) continue;
      sets.push(`${k}=?`);
      params.push(k==='tags'?JSON.stringify(data[k]):k==='stock_symbol'?data[k].toUpperCase():data[k]);
    }
    if (!sets.length) return this.findById(id);
    params.push(id);
    await pool.execute(`UPDATE stocks SET ${sets.join(',')} WHERE stock_id=?`, params);
    return this.findById(id);
  }

  static async delete(id) {
    const [r] = await pool.execute(`UPDATE stocks SET status='delisted' WHERE stock_id=?`, [id]);
    return r.affectedRows > 0;
  }

  static async getStats() {
    const [[t]] = await pool.execute(`SELECT COUNT(*) AS total, SUM(status='active') AS active, SUM(status='suspended') AS suspended, SUM(status='delisted') AS delisted FROM stocks`);
    const [bySector] = await pool.execute(`SELECT sector, COUNT(*) AS count FROM stocks WHERE sector IS NOT NULL GROUP BY sector`);
    const [byExchange] = await pool.execute(`SELECT exchange, COUNT(*) AS count FROM stocks WHERE exchange IS NOT NULL GROUP BY exchange`);
    return { totals: t, by_sector: bySector, by_exchange: byExchange };
  }

  static _parse(r) {
    return { ...r, tags: r.tags ? (typeof r.tags==='string'?JSON.parse(r.tags):r.tags) : [], market_cap: r.market_cap ? parseFloat(r.market_cap) : null };
  }
}

class Category {
  static async create(data) {
    const [r] = await pool.execute(`INSERT INTO stock_categories (category_name,description) VALUES (?,?)`, [data.category_name, data.description||null]);
    return this.findById(r.insertId);
  }
  static async findById(id) {
    const [rows] = await pool.execute(`SELECT * FROM stock_categories WHERE category_id=?`, [id]);
    return rows[0]||null;
  }
  static async findAll() {
    const [rows] = await pool.execute(`SELECT * FROM stock_categories ORDER BY category_name`);
    return rows;
  }
  static async update(id, data) {
    const sets=[],params=[];
    if (data.category_name!==undefined){sets.push('category_name=?');params.push(data.category_name);}
    if (data.description!==undefined){sets.push('description=?');params.push(data.description);}
    if (!sets.length) return this.findById(id);
    params.push(id);
    await pool.execute(`UPDATE stock_categories SET ${sets.join(',')} WHERE category_id=?`, params);
    return this.findById(id);
  }
  static async delete(id) {
    const [r] = await pool.execute(`DELETE FROM stock_categories WHERE category_id=?`, [id]);
    return r.affectedRows>0;
  }
}

module.exports = { Stock, Category };
