import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { DashboardPage } from './pages/DashboardPage';
import { AnalysisPage } from './pages/AnalysisPage';
import { Menu } from 'lucide-react';

export function App() {
  const [currentTab, setCurrentTab] = useState<'dashboard' | 'analysis'>('dashboard');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

  return (
    <div className="flex min-h-screen bg-[#E6F2FC] text-[#203B5D]">
      {/* Sidebar Component */}
      <Sidebar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        isCollapsed={isSidebarCollapsed}
        setIsCollapsed={setIsSidebarCollapsed}
        isMobileOpen={isMobileMenuOpen}
        setIsMobileOpen={setIsMobileMenuOpen}
      />

      {/* Main Content Area */}
      <div
        className={`flex-1 flex flex-col min-h-screen min-w-0 transition-all duration-300 ease-in-out ${
          isSidebarCollapsed ? 'md:ml-20' : 'md:ml-64'
        } ml-0`}
      >
        {/* Mobile Top Navbar (Visible on screens < 768px) */}
        <header className="md:hidden flex items-center justify-between px-4 py-3 bg-[#FFFFFF] border-b border-[#C5DCF2] sticky top-0 z-30 shadow-sm">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setIsMobileMenuOpen(true)}
              className="p-1.5 rounded-md text-[#71849A] hover:text-[#203B5D] hover:bg-[#E6F2FC] transition-colors cursor-pointer"
              aria-label="Open navigation menu"
            >
              <Menu className="w-5 h-5 text-[#203B5D]" />
            </button>
            <span className="text-[14px] font-bold text-[#203B5D] tracking-tight">
              EEG Sentiment Lab
            </span>
          </div>

          <span className="text-[11px] font-medium text-[#00A9D6] bg-[#E6F2FC] px-2.5 py-0.5 rounded-full border border-[#C5DCF2]">
            Academic Software
          </span>
        </header>

        {/* Tab 1: Dashboard View */}
        {currentTab === 'dashboard' && (
          <DashboardPage onStartAnalysis={() => setCurrentTab('analysis')} />
        )}

        {/* Tab 2: EEG Sentiment Analysis & Results */}
        {currentTab === 'analysis' && <AnalysisPage />}
      </div>
    </div>
  );
}

export default App;

