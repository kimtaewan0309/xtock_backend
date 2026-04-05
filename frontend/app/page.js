"use client";

import { useState } from "react";
import {
  Search as SearchIcon,
  Settings,
  LayoutDashboard,
  User,
  Menu,
  X,
  BookOpen,
} from "lucide-react";

import DashboardSection from "@/components/DashboardSection";
import LearningCenter from "@/components/LearningCenter";
import PortfolioSection from "@/components/PortfolioSection";
import SearchSection from "@/components/SearchSection";
import SettingsSection from "@/components/SettingsSection";
import ThemeProvider from "@/components/ThemeProvider";

export default function Home() {
  const [activeMenu, setActiveMenu] = useState("dashboard");
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  const handleSearch = async (query) => {
    setSearchQuery(query);
    setIsLoading(true);
    setAnalysisResult(null);
      /*
    const handleSearch = (query) => {
      console.log("검색:", query);
      // 실제로는 router.push(`/analysis/${encodeURIComponent(query)}`);
      alert(`검색 쿼리: ${query}`);
    };
    */

    // 더미 데이터
    setTimeout(() => {
      const isPositive = Math.random() > 0.5;

      const mockTweet = {
        author: "김가천",
        handle: "kimgc",
        time: "2시간 전",
        text: isPositive
          ? `${query}의 최근 실적이 시장 예상치를 넘어 투자자들의 관심이 집중되고 있습니다. 특히 신규 사업 부문의 성장세가 가팔라지고 있으며, 기존 핵심 사업에서도 안정적인 매출이 이어지고 있습니다. 또한 글로벌 시장 수요가 점차 회복되고 있어 향후 분기 실적에 대한 기대감도 높아지는 상황입니다.`
          : `${query}에 대한 시장의 우려가 커지고 있습니다. 핵심 사업 부문에서 경쟁 심화로 인해 수익성이 둔화되고 있습니다. 또한 글로벌 경기의 영향으로 수요 회복 속도도 예상보다 더딘 모습입니다. 단기적으로 변동성이 확대될 가능성이 높아 투자자들은 보수적인 접근이 필요하다는 의견이 많습니다.`,
        sentiment: isPositive ? "Positive" : "Negative",
        score: isPositive
          ? 0.85 + Math.random() * 0.15
          : 0.65 + Math.random() * 0.25,
      };

      const basePrice = 50000 + Math.random() * 50000;
      const stockData = [];
      const postIndex = 7;

      for (let i = 0; i < 15; i++) {
        let price;
        let change;

        if (i < postIndex) {
          const variation = (Math.random() - 0.5) * 0.02;
          price = basePrice * (1 + variation * (i / postIndex));
          change = variation * 100;
        } else if (i === postIndex) {
          price = basePrice;
          change = 0;
        } else {
          const daysSincePost = i - postIndex;
          const trendStrength = isPositive ? 0.03 : -0.025;
          const variation = (Math.random() - 0.5) * 0.015;
          price = basePrice * (1 + trendStrength * daysSincePost + variation);
          change = ((price - basePrice) / basePrice) * 100;
        }

        const date = new Date();
        date.setDate(date.getDate() - (14 - i));

        stockData.push({
          date: `${date.getMonth() + 1}/${date.getDate()}`,
          price: Math.round(price),
          change: parseFloat(change.toFixed(2)),
        });
      }

      setAnalysisResult({
        tweet: mockTweet,
        stockData,
        postIndex,
      });

      setIsLoading(false);
    }, 1500);
  };

  const menuItems = [
    { id: "dashboard", icon: LayoutDashboard, label: "대시보드" },
    { id: "search", icon: SearchIcon, label: "종목 분석" },
    { id: "learn", icon: BookOpen, label: "학습 센터" },
    { id: "portfolio", icon: User, label: "내 포트폴리오" },
    { id: "settings", icon: Settings, label: "설정" },
  ];

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100">

      {/* 모바일 메뉴 버튼 */}
      <button
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-slate-800 rounded-lg border border-slate-700"
      >
        {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* 사이드바 */}
      <aside
        className={`fixed lg:static inset-y-0 left-0 z-40 w-64 bg-slate-900 border-r border-slate-800 flex flex-col transition-transform duration-300 ${
          isMobileMenuOpen
            ? "translate-x-0"
            : "-translate-x-full lg:translate-x-0"
        }`}
      >
        
        {/* 사용자 정보 */}
        <div className="pt-6 px-4 pb-4 border-t border-slate-800">
          <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-slate-800">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
              <span className="text-sm font-bold">U</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold truncate">사용자</p>
              <p className="text-xs text-slate-400 truncate">user@gachon.ac.kr</p>
            </div>
          </div>
        </div>

        {/* 메뉴 */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {menuItems.map((item) => {
              const Icon = item.icon;
              return (
                <li key={item.id}>
                  <button
                    onClick={() => {
                      setActiveMenu(item.id);
                      setIsMobileMenuOpen(false);
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                      activeMenu === item.id
                        ? "bg-blue-600 text-white"
                        : "text-slate-400 hover:bg-slate-800 hover:text-slate-100"
                    }`}
                  >
                    <Icon size={20} />
                    <span className="font-medium">{item.label}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>
      </aside>

      {/* 메인 컨텐츠 */}
      <main className="flex-1 overflow-auto">
        {/* 상단 서비스 타이틀 */}
        <div className="w-full text-center mt-6">
          <h1 className="inline-flex items-center gap-2 font-extrabold">
            <span className="text-8xl leading-none">X</span>
            <div className="flex flex-col text-left leading-tight">
              <span className="text-4xl">tock</span>
              <span className="text-4xl">ignal</span>
            </div>
          </h1>
        </div>
        <div className="max-w-7xl mx-auto px-4 py-10 lg:px-8">
          {/* 헤더 */}
          <header className="mb-5">
            <div className="flex items-center justify-between mb-2">
              <h1 className="text-3xl md:text-4xl font-extrabold">
                {activeMenu === "dashboard" && "메인 대시보드"}
                {activeMenu === "search" && "종목 분석"}
                {activeMenu === "learn" && "학습 센터"}
                {activeMenu === "portfolio" && "내 포트폴리오"}
                {activeMenu === "settings" && "설정"}
              </h1>
              <p className="text-sm text-slate-400">
                {new Date().toLocaleDateString("ko-KR", {
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                  weekday: "long",
                })}
              </p>
            </div>
            <p className="text-sm md:text-base text-slate-400">
              {activeMenu === "dashboard" && "오늘의 시장 동향 및 예측 요약"}
              {activeMenu === "search" && "기업 및 인물의 온라인 반응과 주가 변동 함께 분석"}
              {activeMenu === "learn" && "주식 기초, 기술적·기본적 분석, AI 기반 투자 학습"}
              {activeMenu === "portfolio" && "나의 관심 종목 및 포트폴리오 관리"}
              {activeMenu === "settings" && "앱 설정 및 개인화"}
            </p>
          </header>

          {/* 메뉴별 컨텐츠 */}
          {activeMenu === "dashboard" && <DashboardSection />}

          {activeMenu === "search" && (
            <SearchSection
              searchQuery={searchQuery}
              isLoading={isLoading}
              analysisResult={analysisResult}
              onSearch={handleSearch}
            />
          )}

          {activeMenu === "learn" && <LearningCenter />}

          {activeMenu === "portfolio" && <PortfolioSection />}

          {activeMenu === "settings" && <SettingsSection />}
        </div>
      </main>

      {/* 모바일 오버레이 */}
      {isMobileMenuOpen && (
        <div
          onClick={() => setIsMobileMenuOpen(false)}
          className="lg:hidden fixed inset-0 bg-black/50 z-30"
        />
      )}
    </div>
  );
}