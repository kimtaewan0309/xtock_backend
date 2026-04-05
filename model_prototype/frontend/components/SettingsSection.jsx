import { useTheme } from "@/components/ThemeProvider";

export default function SettingsSection() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="bg-slate-900 dark:bg-white border border-slate-800 rounded-2xl p-8">
      <div className="space-y-4">
        <div className="flex items-center justify-between py-3 border-b border-slate-800">
          <span className="text-white dark:text-black">알림 설정</span>
          <button className="px-4 py-2 bg-slate-800 dark:bg-gray-200 rounded-lg text-sm text-white dark:text-black">
            설정
          </button>
        </div>

        <div className="flex items-center justify-between py-3 border-b border-slate-800">
          <span className="text-white dark:text-black">테마 변경</span>
          <button
            onClick={toggleTheme}
            className="px-4 py-2 bg-slate-800 dark:bg-gray-200 rounded-lg text-sm text-white dark:text-black"
          >
            {theme === "dark" ? "라이트 모드" : "다크 모드"}
          </button>
        </div>
      </div>
    </div>
  );
}