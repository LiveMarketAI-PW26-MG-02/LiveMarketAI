require('dotenv').config();
const express = require('express');
const helmet = require('helmet');
const morgan = require('morgan');
const cors = require('cors');
const { testConnection } = require('./src/config/db');
const { testConnection: esTest, createIndex } = require('./src/config/elastic');
const routes = require('./src/routes/index');
const SearchIndex = require('./src/models/index');

const app = express();
const PORT = process.env.PORT || 3004;

app.use(helmet()); app.use(cors()); app.use(express.json()); app.use(morgan('dev'));
app.use('/api/v1', routes);
app.use((_, res) => res.status(404).json({ success:false, message:'Not found' }));

const start = async () => {
  await testConnection();
  await esTest();
  await createIndex();
  // Bulk index all existing stocks on startup
  const result = await SearchIndex.bulkIndex();
  console.log(`🔍 Indexed ${result.indexed} stocks into Elasticsearch`);
  app.listen(PORT, () => console.log(`🚀 Service 4 — Search Index running on :${PORT}`));
};
start();
