
import React, { useMemo } from 'react';
import { 
  ComposedChart, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Bar,
  Cell
} from 'recharts';
import { Timeframe } from '../types';

interface MainChartProps {
  timeframe?: Timeframe;
}

const generateMockData = (tf: Timeframe) => {
  const data = [];
  const now = new Date();
  let startDate = new Date(now);

  // Define start dates based on exact requirements
  switch (tf) {
    case '1D':
      startDate.setHours(0, 0, 0, 0); // Start of today
      break;
    case '1M':
      startDate.setDate(1); // Start of current month
      startDate.setHours(0, 0, 0, 0);
      break;
    case '3M':
      startDate.setMonth(now.getMonth() - 3);
      break;
    case '6M':
      startDate.setMonth(now.getMonth() - 6);
      break;
    case 'YTD':
      startDate = new Date(now.getFullYear(), 0, 1); // January 1st
      break;
    case '1Y':
      startDate.setFullYear(now.getFullYear() - 1);
      break;
    case '5Y':
      startDate.setFullYear(now.getFullYear() - 5);
      break;
    default:
      startDate.setDate(now.getDate() - 7);
  }

  // Calculate points and intervals based on duration
  const diffTime = Math.abs(now.getTime() - startDate.getTime());
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  
  let points = 20;
  let interval: 'hour' | 'day' | 'week' | 'month' = 'day';

  if (tf === '1D') {
    points = 24; // Hourly for one day
    interval = 'hour';
  } else if (tf === '1M' || tf === '3M') {
    points = diffDays > 0 ? diffDays : 30;
    interval = 'day';
  } else if (tf === '6M' || tf === 'YTD') {
    points = Math.max(15, Math.floor(diffDays / 3)); // Every 3 days
    interval = 'day';
  } else if (tf === '1Y') {
    points = 52; // Weekly
    interval = 'week';
  } else if (tf === '5Y') {
    points = 60; // Monthly
    interval = 'month';
  }

  let basePrice = 180 + Math.random() * 20;
  
  for (let i = 0; i <= points; i++) {
    const date = new Date(startDate);
    if (interval === 'hour') date.setHours(startDate.getHours() + i);
    else if (interval === 'day') date.setDate(startDate.getDate() + (i * (tf === '6M' || tf === 'YTD' ? 3 : 1)));
    else if (interval === 'week') date.setDate(startDate.getDate() + (i * 7));
    else if (interval === 'month') date.setMonth(startDate.getMonth() + i);

    if (date > now) break; // Don't generate future data

    const open = basePrice + (Math.random() - 0.5) * 4;
    const close = open + (Math.random() - 0.5) * 6;
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;
    
    let timeLabel = '';
    if (tf === '1D') {
      // Showcase date alongside time for 1D view
      timeLabel = `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })}, ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    } else if (['1M', '3M', '6M', 'YTD'].includes(tf)) {
      timeLabel = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } else {
      timeLabel = date.toLocaleDateString([], { month: 'short', year: '2-digit' });
    }

    data.push({ time: timeLabel, open, high, low, close });
    basePrice = close;
  }
  return data;
};

const Candlestick = (props: any) => {
  const { x, y, width, height, open, close, high, low } = props;
  
  if (
    x === undefined || y === undefined || 
    width === undefined || height === undefined || 
    width <= 0 || isNaN(x) || isNaN(y)
  ) return null;
  
  const isBullish = close >= open;
  const color = isBullish ? '#0A8A3A' : '#C0342B';

  const diff = Math.abs(open - close);
  const ratio = height / Math.max(diff, 0.001);
  const wickTop = y - (high - Math.max(open, close)) * ratio;
  const wickBottom = y + height + (Math.min(open, close) - low) * ratio;

  return (
    <g>
      <line
        x1={x + width / 2}
        y1={wickTop}
        x2={x + width / 2}
        y2={wickBottom}
        stroke={color}
        strokeWidth={1}
      />
      <rect
        x={x}
        y={y}
        width={width}
        height={Math.max(height, 1)}
        fill={isBullish ? 'white' : color}
        stroke={color}
        strokeWidth={1.5}
      />
    </g>
  );
};

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const { time, open, high, low, close } = payload[0].payload;
    const isBullish = close >= open;
    return (
      <div className="bg-white p-3 rounded-lg shadow-xl border border-[#F0F0F0] text-[11px] font-bold animate-in fade-in duration-200">
        <p className="text-[#999] mb-2 uppercase tracking-widest">{time}</p>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <div className="flex justify-between gap-4"><span>O</span> <span className="text-[#222]">{open.toFixed(2)}</span></div>
          <div className="flex justify-between gap-4"><span>H</span> <span className="text-[#222]">{high.toFixed(2)}</span></div>
          <div className="flex justify-between gap-4"><span>L</span> <span className="text-[#222]">{low.toFixed(2)}</span></div>
          <div className="flex justify-between gap-4"><span>C</span> <span className={isBullish ? 'text-[#0A8A3A]' : 'text-[#C0342B]'}>{close.toFixed(2)}</span></div>
        </div>
      </div>
    );
  }
  return null;
};

const MainChart: React.FC<MainChartProps> = ({ timeframe = '1D' }) => {
  const chartData = useMemo(() => generateMockData(timeframe), [timeframe]);

  return (
    <div className="w-full h-full min-h-[300px] min-w-0 relative select-none">
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F0F0F0" />
          <XAxis 
            dataKey="time" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#999', fontSize: 10, fontWeight: 600 }}
            minTickGap={30}
            dy={10}
          />
          <YAxis 
            domain={['auto', 'auto']} 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#999', fontSize: 10, fontWeight: 600 }}
            orientation="right"
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,0,0,0.02)' }} />
          <Bar
            dataKey="close"
            shape={<Candlestick />}
            isAnimationActive={false}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.close >= entry.open ? '#0A8A3A' : '#C0342B'} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MainChart;
