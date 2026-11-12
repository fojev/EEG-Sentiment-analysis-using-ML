import React from 'react';
import { LayoutDashboard, Activity, ChevronLeft, ChevronRight, X, Brain } from 'lucide-react';

interface SidebarProps {
  currentTab: 'dashboard' | 'analysis';
  setCurrentTab: (tab: 'dashboard' | 'analysis') => void;
  isCollapsed: boolean;
  setIsCollapsed: React.Dispatch<React.SetStateAction<boolean>>;
  isMobileOpen: boolean;
  setIsMobileOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  setCurrentTab,
  isCollapsed,
  setIsCollapsed,
  isMobileOpen,
  setIsMobileOpen,
}) => {
  const handleNavClick = (tab: 'dashboard' | 'analysis') => {
    setCurrentTab(tab);
    setIsMobileOpen(false); // Close mobile drawer on selection
  };

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isMobileOpen && (
        <div
          onClick={() => setIsMobileOpen(false)}
          className="fixed inset-0 bg-[#203B5D]/40 backdrop-blur-sm z-40 md:hidden transition-opacity"
          aria-hidden="true"
        />
      )}

      {/* Sidebar Container: White Surface with Subtle Blue Border */}
      <aside
        className={`fixed top-0 left-0 h-screen z-50 flex flex-col py-6 bg-[#FFFFFF] border-r border-[#C5DCF2] select-none transition-all duration-300 ease-in-out ${
          isCollapsed ? 'w-20 px-3' : 'w-64 px-4'
        } ${
          isMobileOpen ? 'translate-x-0 shadow-xl' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Top Header: Icon + Toggle */}
        <div
          className={`flex items-center pb-5 border-b border-[#E6F2FC] ${
            isCollapsed ? 'flex-col gap-3 justify-center' : 'justify-between px-1'
          }`}
        >
          <div className="w-10 h-10 rounded-lg bg-[#E6F2FC] flex items-center justify-center text-[#00A9D6] shrink-0 border border-[#C5DCF2]">
            <Brain className="w-5 h-5 text-[#00A9D6]" />
          </div>

          {/* Desktop Collapse / Expand Toggle Button */}
          <button
            type="button"
            onClick={() => setIsCollapsed((prev) => !prev)}
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="hidden md:flex items-center justify-center w-7 h-7 rounded-md text-[#71849A] hover:text-[#203B5D] hover:bg-[#E6F2FC] border border-[#C5DCF2] transition-colors cursor-pointer"
          >
            {isCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </button>

          {/* Mobile Close Button */}
          <button
            type="button"
            onClick={() => setIsMobileOpen(false)}
            title="Close navigation"
            className="flex md:hidden items-center justify-center w-7 h-7 rounded-md text-[#71849A] hover:text-[#203B5D] hover:bg-[#E6F2FC] transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Navigation Links */}
        <nav className="flex flex-col gap-2 mt-6">
          {/* Tab 1: Dashboard */}
          <button
            type="button"
            onClick={() => handleNavClick('dashboard')}
            title="Dashboard"
            className={`flex items-center gap-3 py-2.5 rounded-lg transition-all text-[13.5px] text-left w-full cursor-pointer ${
              isCollapsed ? 'justify-center px-0' : 'px-3.5'
            } ${
              currentTab === 'dashboard'
                ? 'bg-[#00A9D6] text-white font-semibold shadow-sm'
                : 'text-[#203B5D] hover:text-[#00A9D6] hover:bg-[#E6F2FC] font-medium'
            }`}
          >
            <LayoutDashboard className="w-[19px] h-[19px] shrink-0" />
            {!isCollapsed && <span className="truncate">Dashboard</span>}
          </button>

          {/* Tab 2: EEG Sentiment Analysis */}
          <button
            type="button"
            onClick={() => handleNavClick('analysis')}
            title="EEG Sentiment Analysis"
            className={`flex items-center gap-3 py-2.5 rounded-lg transition-all text-[13.5px] text-left w-full cursor-pointer ${
              isCollapsed ? 'justify-center px-0' : 'px-3.5'
            } ${
              currentTab === 'analysis'
                ? 'bg-[#00A9D6] text-white font-semibold shadow-sm'
                : 'text-[#203B5D] hover:text-[#00A9D6] hover:bg-[#E6F2FC] font-medium'
            }`}
          >
            <Activity className="w-[19px] h-[19px] shrink-0" />
            {!isCollapsed && <span className="truncate">EEG Sentiment Analysis</span>}
          </button>
        </nav>
      </aside>
    </>
  );
};
