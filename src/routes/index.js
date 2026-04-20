const { Router } = require('express');
const router = Router();
const ctrl = require('../controllers/orderController');
const { validate, createOrderSchema } = require('../middleware/validate');

// ── Health ─────────────────────────────────────────────────────────────────
router.get('/health', (_, res) => res.json({
  status:  'ok',
  service: 'order-service',
  port:    process.env.PORT || 3007,
}));

// ── Step 1: Place order ───────────────────────────────────────────────────
// POST /api/v1/orders
// Body: { investor_id, stock_id, quantity }
router.post('/orders', validate(createOrderSchema), ctrl.createOrder);

// ── Step 3: Get all orders for investor ──────────────────────────────────
// GET /api/v1/orders/investor/:investor_id
router.get('/orders/investor/:investor_id', ctrl.getAllOrders);

// ── Step 6: Orders sorted by position ────────────────────────────────────
// GET /api/v1/orders/investor/:investor_id/sorted
router.get('/orders/investor/:investor_id/sorted', ctrl.getOrdersByPosition);

// ── Step 7: Paginated order history ──────────────────────────────────────
// GET /api/v1/orders/investor/:investor_id/history?page=1&limit=10
router.get('/orders/investor/:investor_id/history', ctrl.getPaginatedOrders);

// ── Step 4: Total capital spent ───────────────────────────────────────────
// GET /api/v1/orders/investor/:investor_id/capital
router.get('/orders/investor/:investor_id/capital', ctrl.getTotalCapital);

// ── Step 5 & 8: Holdings — total units per stock ──────────────────────────
// GET /api/v1/orders/investor/:investor_id/holdings
router.get('/orders/investor/:investor_id/holdings', ctrl.getHoldings);

// ── Step 11: Activity — order count per stock ─────────────────────────────
// GET /api/v1/orders/investor/:investor_id/activity
router.get('/orders/investor/:investor_id/activity', ctrl.getActivity);

// ── Step 12: Summary — quantity + order count per stock ───────────────────
// GET /api/v1/orders/investor/:investor_id/summary
router.get('/orders/investor/:investor_id/summary', ctrl.getSummary);

// ── Step 13: Full portfolio ────────────────────────────────────────────────
// GET /api/v1/orders/investor/:investor_id/portfolio
router.get('/orders/investor/:investor_id/portfolio', ctrl.getPortfolio);

// ── Step 10: Verify stock before order ────────────────────────────────────
// GET /api/v1/orders/verify/:stock_id
router.get('/orders/verify/:stock_id', ctrl.verifyStock);

// ── Get single order ───────────────────────────────────────────────────────
// GET /api/v1/orders/:order_id
router.get('/orders/:order_id', ctrl.getOrderById);

module.exports = router;
