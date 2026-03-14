-- ============================================================
-- STOCK PLATFORM — Master Schema
-- All 6 services
-- ============================================================

CREATE DATABASE IF NOT EXISTS stock_platform;
USE stock_platform;

-- -----------------------------------------------
-- SERVICE 1: Stock Catalog
-- -----------------------------------------------
CREATE TABLE IF NOT EXISTS stock_categories (
  category_id    BIGINT        NOT NULL AUTO_INCREMENT PRIMARY KEY,
  category_name  VARCHAR(100)  NOT NULL UNIQUE,
  description    TEXT,
  created_at     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stocks (
  stock_id       BIGINT         NOT NULL AUTO_INCREMENT PRIMARY KEY,
  stock_symbol   VARCHAR(20)    NOT NULL UNIQUE,
  stock_name     VARCHAR(255)   NOT NULL,
  category_id    BIGINT         DEFAULT NULL,
  description    TEXT,
  sector         VARCHAR(100)   DEFAULT NULL,
  industry       VARCHAR(100)   DEFAULT NULL,
  exchange       VARCHAR(50)    DEFAULT NULL,
  currency       VARCHAR(10)    NOT NULL DEFAULT 'USD',
  isin           VARCHAR(12)    DEFAULT NULL UNIQUE,
  market_cap     DECIMAL(20,2)  DEFAULT NULL,
  ipo_date       DATE           DEFAULT NULL,
  country        VARCHAR(100)   DEFAULT NULL,
  tags           JSON           DEFAULT NULL,
  status         ENUM('active','suspended','delisted') NOT NULL DEFAULT 'active',
  created_at     TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at     TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_stock_category FOREIGN KEY (category_id) REFERENCES stock_categories(category_id) ON DELETE SET NULL,
  INDEX idx_symbol   (stock_symbol),
  INDEX idx_status   (status),
  INDEX idx_sector   (sector),
  INDEX idx_exchange (exchange),
  INDEX idx_country  (country)
);

-- -----------------------------------------------
-- SERVICE 2: Price Adjustment Engine
-- -----------------------------------------------
CREATE TABLE IF NOT EXISTS stock_prices (
  price_id        BIGINT         NOT NULL AUTO_INCREMENT PRIMARY KEY,
  stock_id        BIGINT         NOT NULL,
  base_price      DECIMAL(15,4)  NOT NULL DEFAULT 0.0000,
  adjusted_price  DECIMAL(15,4)  NOT NULL DEFAULT 0.0000,
  currency        VARCHAR(10)    NOT NULL DEFAULT 'USD',
  last_updated    TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_price_stock FOREIGN KEY (stock_id) REFERENCES stocks(stock_id) ON DELETE CASCADE,
  UNIQUE KEY uq_stock_price (stock_id),
  INDEX idx_price_stock (stock_id)
);

CREATE TABLE IF NOT EXISTS price_adjustment_rules (
  rule_id           BIGINT         NOT NULL AUTO_INCREMENT PRIMARY KEY,
  rule_name         VARCHAR(100)   NOT NULL,
  adjustment_type   ENUM('discount','multiplier','override') NOT NULL,
  adjustment_value  DECIMAL(10,4)  NOT NULL,
  valid_from        DATETIME       NOT NULL,
  valid_to          DATETIME       NOT NULL,
  status            ENUM('active','inactive') NOT NULL DEFAULT 'active',
  created_at        TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at        TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_rule_status (status),
  INDEX idx_rule_dates  (valid_from, valid_to)
);

-- -----------------------------------------------
-- SERVICE 6: Availability Scheduler
-- -----------------------------------------------
CREATE TABLE IF NOT EXISTS exchange_hours (
  exchange_id    BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  exchange_name  VARCHAR(50)  NOT NULL UNIQUE,
  open_time      TIME         NOT NULL,
  close_time     TIME         NOT NULL,
  timezone       VARCHAR(50)  NOT NULL DEFAULT 'UTC',
  created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_availability (
  availability_id  BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  stock_id         BIGINT       NOT NULL,
  available_from   DATETIME     NOT NULL,
  available_until  DATETIME     NOT NULL,
  event_name       VARCHAR(100) DEFAULT NULL,
  status           ENUM('active','paused') NOT NULL DEFAULT 'active',
  created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_avail_stock FOREIGN KEY (stock_id) REFERENCES stocks(stock_id) ON DELETE CASCADE,
  INDEX idx_avail_stock  (stock_id),
  INDEX idx_avail_status (status)
);

-- -----------------------------------------------
-- SEED DATA
-- -----------------------------------------------
INSERT IGNORE INTO stock_categories (category_name, description) VALUES
  ('Technology',     'Technology sector stocks'),
  ('Finance',        'Banking and financial stocks'),
  ('Healthcare',     'Pharmaceutical and health stocks'),
  ('Energy',         'Oil, gas, and renewable energy stocks'),
  ('Consumer Goods', 'Retail and consumer product stocks'),
  ('Automobile',     'Automobile and ancillary stocks'),
  ('Infrastructure', 'Infrastructure and construction stocks');

INSERT IGNORE INTO exchange_hours (exchange_name, open_time, close_time, timezone) VALUES
  ('NSE',    '09:15:00', '15:30:00', 'Asia/Kolkata'),
  ('BSE',    '09:15:00', '15:30:00', 'Asia/Kolkata'),
  ('NASDAQ', '09:30:00', '16:00:00', 'America/New_York'),
  ('NYSE',   '09:30:00', '16:00:00', 'America/New_York'),
  ('LSE',    '08:00:00', '16:30:00', 'Europe/London');

INSERT IGNORE INTO stocks (stock_symbol, stock_name, category_id, sector, industry, exchange, currency, isin, market_cap, ipo_date, country, tags, status) VALUES
  ('AAPL',     'Apple Inc.',                  1, 'Technology',  'Consumer Electronics', 'NASDAQ', 'USD', 'US0378331005', 2800000000000.00, '1980-12-12', 'USA',   '["tech","nasdaq","blue-chip"]', 'active'),
  ('TCS',      'Tata Consultancy Services',   1, 'Technology',  'IT Services',          'NSE',    'INR', 'INE467B01029', 1300000000000.00, '2004-08-25', 'India', '["tech","nse","it"]',           'active'),
  ('HDFCBANK', 'HDFC Bank Ltd',               2, 'Finance',     'Banking',              'NSE',    'INR', 'INE040A01034', 800000000000.00,  '1995-05-19', 'India', '["finance","banking","nse"]',   'active'),
  ('RELIANCE', 'Reliance Industries Ltd',     4, 'Energy',      'Oil & Gas',            'NSE',    'INR', 'INE002A01018', 1700000000000.00, '1977-11-09', 'India', '["energy","nse","conglomerate"]','active'),
  ('TSLA',     'Tesla Inc.',                  6, 'Automobile',  'Electric Vehicles',    'NASDAQ', 'USD', 'US88160R1014', 800000000000.00,  '2010-06-29', 'USA',   '["ev","nasdaq","automobile"]',  'active');

INSERT IGNORE INTO stock_prices (stock_id, base_price, adjusted_price, currency) VALUES
  (1, 189.5000, 189.5000, 'USD'),
  (2, 3800.0000, 3800.0000, 'INR'),
  (3, 1620.0000, 1620.0000, 'INR'),
  (4, 2450.0000, 2450.0000, 'INR'),
  (5, 245.0000, 245.0000, 'USD');
