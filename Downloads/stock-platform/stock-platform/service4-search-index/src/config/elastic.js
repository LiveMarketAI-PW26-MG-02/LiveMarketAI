const { Client } = require('@elastic/elasticsearch');

const client = new Client({
  node: process.env.ES_HOST || 'http://localhost:9200',
});

const INDEX = 'stock_search_index';

const testConnection = async () => {
  try {
    await client.ping();
    console.log('✅ Elasticsearch connected');
  } catch(e) {
    console.error('❌ Elasticsearch failed:', e.message);
    process.exit(1);
  }
};

// Create index with correct mappings matching your spec
const createIndex = async () => {
  const exists = await client.indices.exists({ index: INDEX });
  if (exists) return console.log('ℹ️  Index already exists');

  await client.indices.create({
    index: INDEX,
    mappings: {
      properties: {
        stock_id:     { type: 'keyword' },
        stock_symbol: { type: 'text', fields: { keyword: { type: 'keyword' } } },
        stock_name:   { type: 'text' },
        category:     { type: 'keyword' },
        sector:       { type: 'keyword' },
        industry:     { type: 'keyword' },
        tags:         { type: 'keyword' },
        description:  { type: 'text' },
        market_cap:   { type: 'double' },
        country:      { type: 'keyword' },
        exchange:     { type: 'keyword' },
        status:       { type: 'keyword' },
      }
    }
  });
  console.log('✅ Elasticsearch index created');
};

module.exports = { client, INDEX, testConnection, createIndex };
