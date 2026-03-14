const Redis = require('ioredis');

const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379,
  retryStrategy: (times) => Math.min(times * 100, 3000),
});

redis.on('connect', () => console.log('✅ Redis connected'));
redis.on('error',   (e) => console.error('❌ Redis error:', e.message));

module.exports = redis;
