require('dotenv').config();
const express = require('express');
const helmet = require('helmet');
const morgan = require('morgan');
const cors = require('cors');
const { testConnection } = require('./src/config/db');
const routes = require('./src/routes/index');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(helmet()); app.use(cors()); app.use(express.json()); app.use(morgan('dev'));
app.use('/api/v1', routes);
app.use((_, res) => res.status(404).json({ success: false, message: 'Not found' }));

const start = async () => {
  await testConnection();
  app.listen(PORT, () => console.log(`🚀 Service 1 — Stock Catalog running on :${PORT}`));
};
start();
