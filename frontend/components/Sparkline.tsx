
import React from 'react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

interface SparklineProps {
  data: number[];
  color: string;
}

const Sparkline: React.FC<SparklineProps> = ({ data, color }) => {
  const chartData = data.map((val, i) => ({ value: val, index: i }));
  
  return (
    <div className="w-16 h-8 min-h-[32px] min-w-[64px] relative">
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <LineChart data={chartData}>
          <Line 
            type="monotone" 
            dataKey="value" 
            stroke={color} 
            strokeWidth={1.5} 
            dot={false} 
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default Sparkline;
