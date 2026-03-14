require('dotenv').config();
const express = require('express');
const helmet  = require('helmet');
const morgan  = require('morgan');
const cors    = require('cors');
const cron    = require('node-cron');
const { testConnection } = require('./src/config/db');
const routes  = require('./src/routes/index');
const { StockAvailability, ExchangeHours } = require('./src/models/index');

const app  = express();
const PORT = process.env.PORT || 3006;

app.use(helmet()); app.use(cors()); app.use(express.json()); app.use(morgan('dev'));
app.use('/api/v1', routes);
app.use((_, res) => res.status(404).json({ success:false, message:'Not found' }));

const start = async () => {
  await testConnection();

  // Every minute — expire past schedules
  cron.schedule('* * * * *', async () => {
    const result = await StockAvailability.expirePastSchedules();
    if (result.expired > 0) console.log(`⏰ Expired ${result.expired} availability schedules`);
  });

  // Every hour — log exchange statuses
  cron.schedule('0 * * * *', async () => {
    const statuses = await ExchangeHours.checkAllExchanges();
    statuses.forEach(s => console.log(`📊 ${s.exchange_name}: ${s.is_open ? '🟢 OPEN' : '🔴 CLOSED'}`));
  });

  app.listen(PORT, () => console.log(`🚀 Service 6 — Availability Scheduler running on :${PORT}`));
};
start();
