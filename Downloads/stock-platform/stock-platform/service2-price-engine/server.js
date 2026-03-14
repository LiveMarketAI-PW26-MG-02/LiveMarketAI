require('dotenv').config();
const express = require('express');
const helmet = require('helmet');
const morgan = require('morgan');
const cors = require('cors');
const cron = require('node-cron');
const { testConnection } = require('./src/config/db');
const routes = require('./src/routes/index');
const { StockPrice } = require('./src/models/index');

const app = express();
const PORT = process.env.PORT || 3002;

app.use(helmet()); app.use(cors()); app.use(express.json()); app.use(morgan('dev'));
app.use('/api/v1', routes);
app.use((_, res) => res.status(404).json({ success:false, message:'Not found' }));

const start = async () => {
  await testConnection();

  // Run price adjustment every minute
  cron.schedule('* * * * *', async () => {
    console.log('⚙️  Running price adjustment engine...');
    const result = await StockPrice.applyRulesToAll();
    console.log(`✅ Adjusted prices for ${result.updated} stocks`);
  });

  app.listen(PORT, () => console.log(`🚀 Service 2 — Price Adjustment Engine running on :${PORT}`));
};
start();
