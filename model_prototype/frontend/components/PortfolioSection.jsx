"use client";

import { useState } from "react";
import { TrendingUp, TrendingDown, Plus, X } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// 샘플 주식 데이터
const availableStocks = [
  { symbol: "TSLA", name: "테슬라", currentPrice: 242.8 },
  { symbol: "NVDA", name: "엔비디아", currentPrice: 495.2 },
  { symbol: "AAPL", name: "애플", currentPrice: 195.5 },
  { symbol: "AMZN", name: "아마존", currentPrice: 178.35 },
  { symbol: "MSFT", name: "마이크로소프트", currentPrice: 378.2 },
  { symbol: "META", name: "메타", currentPrice: 474.3 },
  { symbol: "GOOGL", name: "구글", currentPrice: 141.8 },  
  { symbol: "NFLX", name: "넷플릭스", currentPrice: 485.5 },
];

// 주가 차트 데이터 생성
const generateChartData = (currentPrice) => {
  const data = [];
  let price = currentPrice * 0.85;

  for (let i = 0; i < 29; i++) {
    const change = (Math.random() - 0.48) * currentPrice * 0.03;
    price += change;
    data.push({
      day: i + 1,
      price: parseFloat(price.toFixed(2)),
    });
  }

  // 마지막 데이터는 현재가로
  data.push({
    day: 30,
    price: currentPrice,
  });

  return data;
};

export default function PortfolioSection() {
  const [portfolio, setPortfolio] = useState([]);
  const [selectedStock, setSelectedStock] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [formData, setFormData] = useState({
    symbol: "",
    shares: "",
    purchasePrice: "",
  });

  const addStock = () => {
    if (!formData.symbol || !formData.shares || !formData.purchasePrice) {
      alert("모든 항목을 입력해주세요");
      return;
    }

    const stock = availableStocks.find((s) => s.symbol === formData.symbol);
    const newStock = {
      ...stock,
      shares: parseFloat(formData.shares),
      purchasePrice: parseFloat(formData.purchasePrice),
      id: Date.now(),
    };

    setPortfolio([...portfolio, newStock]);
    setFormData({ symbol: "", shares: "", purchasePrice: "" });
    setShowAddModal(false);
  };

  const removeStock = (id) => {
    setPortfolio(portfolio.filter((s) => s.id !== id));
    if (selectedStock?.id === id) {
      setSelectedStock(null);
    }
  };

  const calculateReturn = (stock) => {
    const totalReturn =
      (stock.currentPrice - stock.purchasePrice) * stock.shares;
    const returnPercent =
      ((stock.currentPrice - stock.purchasePrice) / stock.purchasePrice) * 100;
    return { totalReturn, returnPercent };
  };

  const totalValue = portfolio.reduce(
    (sum, stock) => sum + stock.currentPrice * stock.shares,
    0
  );
  const totalCost = portfolio.reduce(
    (sum, stock) => sum + stock.purchasePrice * stock.shares,
    0
  );
  const totalReturn = totalValue - totalCost;
  const totalReturnPercent =
    totalCost > 0 ? (totalReturn / totalCost) * 100 : 0;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8">
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-lg font-semibold transition-colors text-white"
        >
          <Plus size={20} />
          관심 종목 추가
        </button>
      </div>

      {/* 전체 수익률 요약 */}
      {portfolio.length > 0 && (
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-slate-800 rounded-lg p-4">
            <p className="text-slate-400 text-sm mb-1">총 평가금액</p>
            <p className="text-white text-xl font-bold">
              ${totalValue.toFixed(2)}
            </p>
          </div>
          <div className="bg-slate-800 rounded-lg p-4">
            <p className="text-slate-400 text-sm mb-1">총 수익/손실</p>
            <p
              className={`text-xl font-bold ${
                totalReturn >= 0 ? "text-green-400" : "text-red-400"
              }`}
            >
              ${totalReturn.toFixed(2)}
            </p>
          </div>
          <div className="bg-slate-800 rounded-lg p-4">
            <p className="text-slate-400 text-sm mb-1">수익률</p>
            <p
              className={`text-xl font-bold ${
                totalReturnPercent >= 0 ? "text-green-400" : "text-red-400"
              }`}
            >
              {totalReturnPercent >= 0 ? "+" : ""}
              {totalReturnPercent.toFixed(2)}%
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        
        {/* 왼쪽: 관심 종목 목록 */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-4">관심 종목</h3>
          {portfolio.length === 0 ? (
            <div className="bg-slate-800 rounded-lg p-8 text-center">
              <p className="text-slate-400">아직 추가된 종목이 없습니다</p>
            </div>
          ) : (
            <div className="space-y-3">
              {portfolio.map((stock) => {
                const { totalReturn, returnPercent } = calculateReturn(stock);
                const isSelected = selectedStock?.id === stock.id;

                return (
                  <div
                    key={stock.id}
                    onClick={() => setSelectedStock(stock)}
                    className={`bg-slate-800 rounded-lg p-4 cursor-pointer transition-all ${
                      isSelected
                        ? "ring-2 ring-blue-500"
                        : "hover:bg-slate-750"
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="text-white font-semibold">
                            {stock.symbol}
                          </h4>
                          <span className="text-slate-400 text-sm">
                            {stock.name}
                          </span>
                        </div>
                        <p className="text-slate-400 text-sm mt-1">
                          {stock.shares}주 보유
                        </p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeStock(stock.id);
                        }}
                        className="text-slate-400 hover:text-red-400 transition-colors"
                      >
                        <X size={18} />
                      </button>
                    </div>

                    <div className="flex items-end justify-between">
                      <div>
                        <p className="text-white text-lg font-bold">
                          ${stock.currentPrice}
                        </p>
                        <p className="text-slate-400 text-xs">
                          매수가: ${stock.purchasePrice}
                        </p>
                      </div>
                      <div className="text-right">
                        <div
                          className={`flex items-center gap-1 ${
                            returnPercent >= 0
                              ? "text-green-400"
                              : "text-red-400"
                          }`}
                        >
                          {returnPercent >= 0 ? (
                            <TrendingUp size={16} />
                          ) : (
                            <TrendingDown size={16} />
                          )}
                          <span className="font-semibold">
                            {returnPercent >= 0 ? "+" : ""}
                            {returnPercent.toFixed(2)}%
                          </span>
                        </div>
                        <p
                          className={`text-sm ${
                            totalReturn >= 0
                              ? "text-green-400"
                              : "text-red-400"
                          }`}
                        >
                          {totalReturn >= 0 ? "+" : ""}$
                          {totalReturn.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* 오른쪽: 주가 차트 */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-4">주가 차트</h3>
          {selectedStock ? (
            <div className="bg-slate-800 rounded-lg p-6">
              <div className="mb-4">
                <h4 className="text-white text-xl font-bold">
                  {selectedStock.symbol}
                </h4>
                <p className="text-slate-400">{selectedStock.name}</p>
                <p className="text-white text-2xl font-bold mt-2">
                  ${selectedStock.currentPrice}
                </p>
              </div>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={generateChartData(selectedStock.currentPrice)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="day"
                    stroke="#94a3b8"
                    tick={{ fill: "#94a3b8" }}
                  />
                  <YAxis
                    stroke="#94a3b8"
                    tick={{ fill: "#94a3b8" }}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1e293b",
                      border: "1px solid #334155",
                      borderRadius: "8px",
                      color: "#fff",
                    }}
                    formatter={(value) => [`$${value}`, "주가"]}
                    labelFormatter={(label) => `${label}일차`}
                  />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="bg-slate-800 rounded-lg p-8 h-[400px] flex items-center justify-center">
              <p className="text-slate-400">
                종목을 선택하면 차트가 표시됩니다
              </p>
            </div>
          )}
        </div>
      </div>

      {/* 종목 추가 모달 */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-2xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white">종목 추가</h3>
              <button
                onClick={() => setShowAddModal(false)}
                className="text-slate-400 hover:text-white"
              >
                <X size={24} />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-slate-300 text-sm mb-2">
                  종목
                </label>
                <select
                  value={formData.symbol}
                  onChange={(e) =>
                    setFormData({ ...formData, symbol: e.target.value })
                  }
                  className="w-full bg-slate-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">종목 선택</option>
                  {availableStocks.map((stock) => (
                    <option key={stock.symbol} value={stock.symbol}>
                      {stock.symbol} - {stock.name} (${stock.currentPrice})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-slate-300 text-sm mb-2">
                  보유 수량
                </label>
                <input
                  type="number"
                  value={formData.shares}
                  onChange={(e) =>
                    setFormData({ ...formData, shares: e.target.value })
                  }
                  placeholder="10"
                  className="w-full bg-slate-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-slate-300 text-sm mb-2">
                  매수 단가 ($)
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={formData.purchasePrice}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      purchasePrice: e.target.value,
                    })
                  }
                  placeholder="150.00"
                  className="w-full bg-slate-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <button
                onClick={addStock}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 rounded-lg transition-colors"
              >
                추가하기
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}