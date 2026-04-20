const Joi = require('joi');

const validate = (schema) => (req, res, next) => {
  const { error, value } = schema.validate(req.body, {
    abortEarly:    false,
    stripUnknown:  true,
  });
  if (error) {
    return res.status(400).json({
      success: false,
      message: 'Validation failed',
      errors:  error.details.map((d) => ({
        field:   d.path.join('.'),
        message: d.message,
      })),
    });
  }
  req.body = value;
  next();
};

const createOrderSchema = Joi.object({
  investor_id: Joi.number().integer().positive().required(),
  stock_id:    Joi.number().integer().positive().required(),
  quantity:    Joi.number().integer().min(1).required(),
});

module.exports = { validate, createOrderSchema };
