require('dotenv').config();
const express = require('express');
const helmet = require('helmet');
const morgan = require('morgan');
const cors = require('cors');
const { testConnection } = require('./src/config/db');
const routes = require('./src/routes/index');
const PriceCache = require('./src/models/index');

const app = express();
const PORT = process.env.PORT || 3003;

app.use(helmet()); app.use(cors()); app.use(express.json()); app.use(morgan('dev'));
app.use('/api/v1', routes);
app.use((_, res) => res.status(404).json({ success:false, message:'Not found' }));

const start = async () => {
  await testConnection();
  // Warm up cache on startup
  const result = await PriceCache.warmUp();
  console.log(`🔥 Cache warmed for ${result.warmed} stocks`);
  app.listen(PORT, () => console.log(`🚀 Service 3 — Price Cache running on :${PORT}`));
};
start();
