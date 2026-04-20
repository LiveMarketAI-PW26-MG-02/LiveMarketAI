require('dotenv').config();
const express = require('express');
const helmet  = require('helmet');
const morgan  = require('morgan');
const cors    = require('cors');

const { testConnection } = require('./src/config/db');
const routes             = require('./src/routes/index');

const app  = express();
const PORT = process.env.PORT || 3007;

// ── Middleware ──────────────────────────────────────────────────────────────
app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

// ── Routes ──────────────────────────────────────────────────────────────────
app.use('/api/v1', routes);

// ── 404 ─────────────────────────────────────────────────────────────────────
app.use((_, res) => res.status(404).json({ success: false, message: 'Route not found' }));

// ── Global error handler ─────────────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ success: false, message: 'Internal server error' });
});

// ── Start ────────────────────────────────────────────────────────────────────
const start = async () => {
  await testConnection();
  app.listen(PORT, () => {
    console.log(`🚀 Order Service running on port ${PORT}`);
    console.log(`   Base URL: http://localhost:${PORT}/api/v1`);
    console.log('');
    console.log('   Available endpoints:');
    console.log(`   POST   /api/v1/orders`);
    console.log(`   GET    /api/v1/orders/investor/:id`);
    console.log(`   GET    /api/v1/orders/investor/:id/sorted`);
    console.log(`   GET    /api/v1/orders/investor/:id/history`);
    console.log(`   GET    /api/v1/orders/investor/:id/capital`);
    console.log(`   GET    /api/v1/orders/investor/:id/holdings`);
    console.log(`   GET    /api/v1/orders/investor/:id/activity`);
    console.log(`   GET    /api/v1/orders/investor/:id/summary`);
    console.log(`   GET    /api/v1/orders/investor/:id/portfolio`);
    console.log(`   GET    /api/v1/orders/verify/:stock_id`);
    console.log(`   GET    /api/v1/orders/:order_id`);
  });
};

start();
