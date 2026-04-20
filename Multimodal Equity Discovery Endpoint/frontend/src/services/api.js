import axios from 'axios';

const API = axios.create({ baseURL: '/api/v1' });

export const seedInstruments = () => API.post('/instruments/seed');

export const listInstruments = (page = 1, pageSize = 20) =>
  API.get('/instruments', { params: { page, page_size: pageSize } });

export const getMultimodalProfile = (symbol) =>
  API.get(`/instruments/${symbol}/profile`);

export const getEnrichedProfile = (symbol) =>
  API.get(`/instruments/${symbol}/enriched`);

export const getMarketDepth = (symbol) =>
  API.get(`/instruments/${symbol}/depth`);
