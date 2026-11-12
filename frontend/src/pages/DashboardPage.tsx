import React from 'react';
import { ArrowRight, Upload, CheckCircle2, PieChart, Smile, Meh, Frown, Info } from 'lucide-react';

interface DashboardPageProps {
  onStartAnalysis: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ onStartAnalysis }) => {
  return (
    <div className="flex-1 flex flex-col min-h-screen bg-[#E6F2FC] text-[#203B5D]">
      {/* Main Content Area */}
      <main className="flex-1 p-6 sm:p-10 lg:p-12 flex items-start justify-start">
        <div className="w-full max-w-5xl space-y-8">
          
          {/* 1. PRESERVE EXISTING HERO SECTION */}
          <section className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-8 sm:p-10 shadow-sm space-y-6">
            <div className="space-y-3">
              <h1 className="text-2xl sm:text-3xl lg:text-[34px] font-bold text-[#203B5D] tracking-tight leading-tight">
                EEG Sentiment Analysis
              </h1>
              <p className="text-[15px] sm:text-[16px] text-[#71849A] leading-relaxed max-w-2xl">
                Classify EEG feature data into Positive, Neutral, or Negative sentiment.
              </p>
            </div>

            <div className="pt-1">
              <button
                type="button"
                onClick={onStartAnalysis}
                className="inline-flex items-center gap-2.5 px-6 py-3 rounded-lg bg-[#00A9D6] hover:bg-[#008FB5] active:scale-[0.99] text-white font-semibold text-[14px] shadow-sm transition-all cursor-pointer"
              >
                <span>Start Analysis</span>
                <ArrowRight className="w-4 h-4 text-white" />
              </button>
            </div>
          </section>

          {/* 2. HOW TO GET STARTED SECTION */}
          <section className="space-y-4">
            <h2 className="text-lg sm:text-xl font-bold text-[#203B5D] tracking-tight">
              How to Get Started
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {/* CARD 01 */}
              <div className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-6 shadow-sm hover:border-[#00A9D6] transition-colors flex flex-col justify-between space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold tracking-wider text-[#00A9D6] bg-[#E6F2FC] px-2.5 py-1 rounded-md uppercase">
                      CARD 01
                    </span>
                    <div className="w-8 h-8 rounded-lg bg-[#E6F2FC] flex items-center justify-center">
                      <Upload className="w-4 h-4 text-[#00A9D6]" />
                    </div>
                  </div>
                  <h3 className="text-[16px] font-semibold text-[#203B5D]">
                    Upload Dataset
                  </h3>
                  <p className="text-[13px] sm:text-[14px] text-[#71849A] leading-relaxed">
                    Select and upload your EEG feature CSV file.
                  </p>
                </div>
              </div>

              {/* CARD 02 */}
              <div className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-6 shadow-sm hover:border-[#00A9D6] transition-colors flex flex-col justify-between space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold tracking-wider text-[#00A9D6] bg-[#E6F2FC] px-2.5 py-1 rounded-md uppercase">
                      CARD 02
                    </span>
                    <div className="w-8 h-8 rounded-lg bg-[#E6F2FC] flex items-center justify-center">
                      <CheckCircle2 className="w-4 h-4 text-[#00A9D6]" />
                    </div>
                  </div>
                  <h3 className="text-[16px] font-semibold text-[#203B5D]">
                    Review Dataset
                  </h3>
                  <p className="text-[13px] sm:text-[14px] text-[#71849A] leading-relaxed">
                    Check the uploaded dataset and verify its structure.
                  </p>
                </div>
              </div>

              {/* CARD 03 */}
              <div className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-6 shadow-sm hover:border-[#00A9D6] transition-colors flex flex-col justify-between space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold tracking-wider text-[#00A9D6] bg-[#E6F2FC] px-2.5 py-1 rounded-md uppercase">
                      CARD 03
                    </span>
                    <div className="w-8 h-8 rounded-lg bg-[#E6F2FC] flex items-center justify-center">
                      <PieChart className="w-4 h-4 text-[#00A9D6]" />
                    </div>
                  </div>
                  <h3 className="text-[16px] font-semibold text-[#203B5D]">
                    View Results
                  </h3>
                  <p className="text-[13px] sm:text-[14px] text-[#71849A] leading-relaxed">
                    Explore the predicted sentiment and classification results.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* 3. SENTIMENT CATEGORIES SECTION */}
          <section className="space-y-4">
            <h2 className="text-lg sm:text-xl font-bold text-[#203B5D] tracking-tight">
              Sentiment Categories
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {/* Positive */}
              <div className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-5 shadow-sm space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-lg bg-[#E6F2FC] border border-[#C5DCF2] flex items-center justify-center">
                    <Smile className="w-5 h-5 text-[#00A9D6]" />
                  </div>
                  <div>
                    <h3 className="text-[15px] font-semibold text-[#203B5D]">
                      Positive
                    </h3>
                  </div>
                </div>
                <p className="text-[13px] sm:text-[14px] text-[#71849A] leading-relaxed">
                  Positive sentiment classification.
                </p>
              </div>

              {/* Neutral */}
              <div className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-5 shadow-sm space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-lg bg-[#E6F2FC] border border-[#C5DCF2] flex items-center justify-center">
                    <Meh className="w-5 h-5 text-[#71849A]" />
                  </div>
                  <div>
                    <h3 className="text-[15px] font-semibold text-[#203B5D]">
                      Neutral
                    </h3>
                  </div>
                </div>
                <p className="text-[13px] sm:text-[14px] text-[#71849A] leading-relaxed">
                  Neutral sentiment classification.
                </p>
              </div>

              {/* Negative */}
              <div className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-5 shadow-sm space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-lg bg-[#E6F2FC] border border-[#C5DCF2] flex items-center justify-center">
                    <Frown className="w-5 h-5 text-[#203B5D]" />
                  </div>
                  <div>
                    <h3 className="text-[15px] font-semibold text-[#203B5D]">
                      Negative
                    </h3>
                  </div>
                </div>
                <p className="text-[13px] sm:text-[14px] text-[#71849A] leading-relaxed">
                  Negative sentiment classification.
                </p>
              </div>
            </div>
          </section>

          {/* 4. ABOUT THE ANALYSIS SECTION */}
          <section className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-6 sm:p-8 shadow-sm">
            <div className="flex items-start gap-4">
              <div className="w-9 h-9 rounded-lg bg-[#E6F2FC] border border-[#C5DCF2] flex items-center justify-center shrink-0 mt-0.5">
                <Info className="w-5 h-5 text-[#00A9D6]" />
              </div>
              <div className="space-y-1.5">
                <h2 className="text-[16px] font-bold text-[#203B5D] tracking-tight">
                  About the Analysis
                </h2>
                <p className="text-[14px] text-[#71849A] leading-relaxed">
                  Analyze EEG feature data and explore sentiment classification results through a simple and easy-to-use interface.
                </p>
              </div>
            </div>
          </section>

        </div>
      </main>
    </div>
  );
};
