const Order = require('../models/Order');

// ── Step 1: Place a new order ─────────────────────────────────────────────────
const createOrder = async (req, res) => {
  try {
    const order = await Order.create(req.body);
    return res.status(201).json({
      success: true,
      message: 'Order placed successfully',
      data:    order,
    });
  } catch (err) {
    const status = err.message.includes('not found') || err.message.includes('No price') ? 404 : 400;
    return res.status(status).json({ success: false, message: err.message });
  }
};

// ── Step 3: Get all orders for an investor ────────────────────────────────────
const getAllOrders = async (req, res) => {
  try {
    const orders = await Order.findAllByInvestor(req.params.investor_id);
    return res.json({ success: true, count: orders.length, data: orders });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ── Get single order by ID ────────────────────────────────────────────────────
const getOrderById = async (req, res) => {
  try {
    const order = await Order.findById(req.params.order_id);
    if (!order) return res.status(404).json({ success: false, message: 'Order not found' });
    return res.json({ success: true, data: order });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ── Step 4: Total capital spent by investor ───────────────────────────────────
const getTotalCapital = async (req, res) => {
  try {
    const data = await Order.getTotalCapital(req.params.investor_id);
    return res.json({ success: true, data });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ── Step 5 & 8: Total units held per stock ────────────────────────────────────
const getHoldings = async (req, res) => {
  try {
    const data = await Order.getHoldingsPerStock(req.params.investor_id);
    return res.json({ success: true, count: data.length, data });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ── Step 6: Orders sorted by position ────────────────────────────────────────
const getOrdersByPosition = async (req, res) => {
  try {
    const data = await Order.getOrdersSortedByPosition(req.params.investor_id);
    return res.json({ success: true, count: data.length, data });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ── Step 7: Paginated order history ──────────────────────────────────────────
const getPaginatedOrders = async (req, res) => {
  try {
    const page  = parseInt(req.query.page  || 1);
    const limit = Math.min(parseInt(req.query.limit || 10), 100);
    const result = await Order.getPaginated(req.params.investor_id, { page, limit });
    return res.json({ success: true, ...result });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ── Step 11: Order activity per stock ────────────────────────────────────────
const getActivity = async (req, res) => {
  try {
    const data = await Order.getActivityPerStock(req.params.investor_id);
    return res.json({ success: true, count: data.length, data });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ── Step 12: Summary — quantity + order count per stock ──────────────────────
const getSummary = async (req, res) => {
  try {
    const data = await Order.getSummary(req.params.investor_id);
    return res.json({ success: true, count: data.length, data });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ── Step 13: Full portfolio ───────────────────────────────────────────────────
const getPortfolio = async (req, res) => {
  try {
    const data = await Order.getPortfolio(req.params.investor_id);
    return res.json({ success: true, count: data.length, data });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

// ── Step 10: Pre-execution verification only (without placing order) ──────────
const verifyStock = async (req, res) => {
  try {
    const result = await Order.verify(req.params.stock_id);
    return res.json({ success: true, data: result });
  } catch (err) {
    return res.status(500).json({ success: false, message: err.message });
  }
};

module.exports = {
  createOrder, getAllOrders,   getOrderById,
  getTotalCapital, getHoldings, getOrdersByPosition,
  getPaginatedOrders, getActivity, getSummary,
  getPortfolio, verifyStock,
};
