-- ============================================================
-- Order Service — Schema
-- Connects to existing stock_platform database
-- ============================================================

USE stock_platform;

CREATE TABLE IF NOT EXISTS orders (
  order_id       BIGINT         NOT NULL AUTO_INCREMENT PRIMARY KEY,
  investor_id    BIGINT         NOT NULL,                     -- who is buying
  stock_id       BIGINT         NOT NULL,                     -- which stock (FK → stocks)
  closing_price  DECIMAL(15,4)  NOT NULL,                     -- price at time of order
  quantity       INT            NOT NULL CHECK (quantity > 0),-- how many units bought
  position       INT            NOT NULL,                     -- sequential order number per investor
  status         ENUM('confirmed','cancelled','pending')
                                NOT NULL DEFAULT 'confirmed',
  created_at     TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at     TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

  CONSTRAINT fk_order_stock
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
    ON DELETE RESTRICT,

  INDEX idx_investor    (investor_id),
  INDEX idx_stock       (stock_id),
  INDEX idx_position    (investor_id, position),
  INDEX idx_status      (status),
  INDEX idx_created_at  (created_at)
);
