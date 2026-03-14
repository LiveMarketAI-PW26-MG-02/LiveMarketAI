const { client, INDEX } = require('../config/elastic');
const { pool } = require('../config/db');

class SearchIndex {

  // Index a single stock document
  static async indexStock(stock) {
    await client.index({
      index: INDEX,
      id:    String(stock.stock_id),
      document: {
        stock_id:     String(stock.stock_id),
        stock_symbol: stock.stock_symbol,
        stock_name:   stock.stock_name,
        category:     stock.category_name || null,
        sector:       stock.sector        || null,
        industry:     stock.industry      || null,
        tags:         Array.isArray(stock.tags) ? stock.tags : (stock.tags ? JSON.parse(stock.tags) : []),
        description:  stock.description   || null,
        market_cap:   stock.market_cap    ? parseFloat(stock.market_cap) : null,
        country:      stock.country       || null,
        exchange:     stock.exchange      || null,
        status:       stock.status        || 'active',
      }
    });
    await client.indices.refresh({ index: INDEX });
    return true;
  }

  // Bulk index all stocks from MySQL
  static async bulkIndex() {
    const [stocks] = await pool.execute(
      `SELECT s.*, c.category_name FROM stocks s LEFT JOIN stock_categories c ON s.category_id=c.category_id`);

    if (!stocks.length) return { indexed: 0 };

    const operations = stocks.flatMap(s => [
      { index: { _index: INDEX, _id: String(s.stock_id) } },
      {
        stock_id:     String(s.stock_id),
        stock_symbol: s.stock_symbol,
        stock_name:   s.stock_name,
        category:     s.category_name || null,
        sector:       s.sector        || null,
        industry:     s.industry      || null,
        tags:         s.tags ? (typeof s.tags==='string' ? JSON.parse(s.tags) : s.tags) : [],
        description:  s.description   || null,
        market_cap:   s.market_cap    ? parseFloat(s.market_cap) : null,
        country:      s.country       || null,
        exchange:     s.exchange      || null,
        status:       s.status,
      }
    ]);

    const result = await client.bulk({ operations });
    await client.indices.refresh({ index: INDEX });
    return { indexed: stocks.length, errors: result.errors };
  }

  // Full-text search
  static async search({ q, sector, industry, exchange, country, category, status='active', page=1, limit=20 }) {
    const from = (page - 1) * limit;
    const must   = [];
    const filter = [];

    // Full text across name, symbol, description
    if (q) {
      must.push({
        multi_match: {
          query:  q,
          fields: ['stock_name^3', 'stock_symbol^3', 'description', 'tags^2'],
          fuzziness: 'AUTO',
        }
      });
    }

    if (status)   filter.push({ term: { status } });
    if (sector)   filter.push({ term: { sector } });
    if (industry) filter.push({ term: { industry } });
    if (exchange) filter.push({ term: { exchange } });
    if (country)  filter.push({ term: { country } });
    if (category) filter.push({ term: { category } });

    const { hits } = await client.search({
      index: INDEX,
      from,
      size:  limit,
      query: {
        bool: {
          must:   must.length   ? must   : [{ match_all: {} }],
          filter: filter.length ? filter : [],
        }
      },
      sort: q ? ['_score'] : [{ stock_name: 'asc' }],
    });

    return {
      data: hits.hits.map(h => ({ ...h._source, _score: h._score })),
      pagination: {
        total:       hits.total.value,
        page,
        limit,
        total_pages: Math.ceil(hits.total.value / limit),
      }
    };
  }

  // Delete from index
  static async deleteStock(stock_id) {
    await client.delete({ index: INDEX, id: String(stock_id) });
    return true;
  }
}

module.exports = SearchIndex;
