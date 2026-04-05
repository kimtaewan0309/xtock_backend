import { Search as SearchIcon } from "lucide-react";

export default function DashboardSection() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <p className="text-sm text-slate-400 mb-2">오늘의 인기 추천</p>
          <h3 className="text-2xl font-bold mb-1">Tesla (TSLA)</h3>
          <p className="text-emerald-400 text-lg font-semibold">+3.12%</p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <p className="text-sm text-slate-400 mb-2">NASDAQ 지수</p>
          <h3 className="text-2xl font-bold mb-1">16,920.45</h3>
          <p className="text-emerald-400 text-lg font-semibold">+1.05%</p>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <p className="text-sm text-slate-400 mb-2">시장 예측 동향</p>
          <h3 className="text-2xl font-bold mb-1">긍정적</h3>
          <p className="text-emerald-400 text-lg font-semibold">↗ 상승 예측 우세</p>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">실시간 주가 변동 및 예측</h2>
          <div className="relative">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
              size={20}
            />
            <input
              type="text"
              placeholder="종목 검색"
              className="pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-sm text-slate-400 border-b border-slate-800">
                <th className="pb-3 font-medium">종목명</th>
                <th className="pb-3 font-medium">현재가</th>
                <th className="pb-3 font-medium">변동률</th>
                <th className="pb-3 font-medium">AI 예측</th>
                <th className="pb-3 font-medium">관련 키워드</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b border-slate-800">
                <td className="py-4">Tesla</td>
                <td className="py-4">$218.30</td>
                <td className="py-4 text-blue-400">+3.12%</td>
                <td className="py-4 text-blue-400">+1.8%</td>
                <td className="py-4 text-slate-400">#EV #일론머스크 #전기차</td>
              </tr>
              <tr className="border-b border-slate-800">
                <td className="py-4">NVIDIA</td>
                <td className="py-4">$1,020.55</td>
                <td className="py-4 text-blue-400">+2.45%</td>
                <td className="py-4 text-blue-400">+1.2%</td>
                <td className="py-4 text-slate-400">#AI #GPU #데이터센터</td>
              </tr>
              <tr className="border-b border-slate-800">
                <td className="py-4">Apple</td>
                <td className="py-4">$197.40</td>
                <td className="py-4 text-rose-400">-0.75%</td>
                <td className="py-4 text-rose-400">-0.3%</td>
                <td className="py-4 text-slate-400">#iPhone #서비스 #하드웨어</td>
              </tr>
              <tr className="border-b border-slate-800">
                <td className="py-4">Amazon</td>
                <td className="py-4">$185.22</td>
                <td className="py-4 text-blue-400">+1.05%</td>
                <td className="py-4 text-blue-400">+0.7%</td>
                <td className="py-4 text-slate-400">#AWS #전자상거래 #클라우드</td>
              </tr>
              <tr className="border-b border-slate-800">
                <td className="py-4">Microsoft</td>
                <td className="py-4">$412.67</td>
                <td className="py-4 text-rose-400">-0.92%</td>
                <td className="py-4 text-rose-400">-0.5%</td>
                <td className="py-4 text-slate-400">#Azure #AI #오피스365</td>
              </tr>
              <tr className="border-b border-slate-800">
                <td className="py-4">Meta Platforms</td>
                <td className="py-4">$488.34</td>
                <td className="py-4 text-blue-400">+1.55%</td>
                <td className="py-4 text-blue-400">+1.1%</td>
                <td className="py-4 text-slate-400">#SNS #AI #메타버스</td>
              </tr>
              <tr>
                <td className="py-4">Alphabet (Google)</td>
                <td className="py-4">$153.20</td>
                <td className="py-4 text-blue-400">+0.68%</td>
                <td className="py-4 text-blue-400">+0.4%</td>
                <td className="py-4 text-slate-400">#검색엔진 #광고 #클라우드</td>
              </tr>
              <tr>
                <td className="py-4">Netflix</td>
                <td className="py-4">$489.50</td>
                <td className="py-4 text-blue-400">+1.3%</td>
                <td className="py-4 text-blue-400">+0.49%</td>
                <td className="py-4 text-slate-400">#스트리밍 #오리지널콘텐츠 #구독서비스</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}