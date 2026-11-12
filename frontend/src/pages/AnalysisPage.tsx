import React, { useState, useRef } from 'react';
import { predictBatchCSV } from '../services/api';
import { BatchPredictionResponse, PredictionItem } from '../types';
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  Play,
  RotateCcw,
  RefreshCw,
  Download,
  X,
  Loader2,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  ShieldCheck,
  Table as TableIcon,
  Trash2,
  SlidersHorizontal,
} from 'lucide-react';

export const AnalysisPage: React.FC = () => {
  // Stepper state: 1 = Upload, 2 = Validation & Preview, 3 = View Results
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3>(1);

  // File state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileStats, setFileStats] = useState<{
    sizeMB: string;
    recordCount: number | string;
    totalColumns: number | string;
    featureCount: number | string;
    isCompatible: boolean;
    missingValues: number;
  }>({
    sizeMB: '0.0',
    recordCount: 'Detecting...',
    totalColumns: 'Detecting...',
    featureCount: '2,548',
    isCompatible: true,
    missingValues: 0,
  });

  // Preview table state (first 5 records)
  const [previewHeaders, setPreviewHeaders] = useState<string[]>([]);
  const [previewRows, setPreviewRows] = useState<Array<Record<string, string>>>([]);

  // Drag and drop state
  const [isDragging, setIsDragging] = useState(false);

  // Loading & error state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Results state
  const [predictionResult, setPredictionResult] = useState<BatchPredictionResponse | null>(null);
  const [sentimentFilter, setSentimentFilter] = useState<'ALL' | 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE'>('ALL');

  // Table pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const ROWS_PER_PAGE = 10;

  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // File selection & client-side validation/preview handler
  const handleFileChange = (file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErrorMessage('Invalid file format. Supported Format: CSV (.csv with 2,548 EEG feature dimensions).');
      return;
    }

    setErrorMessage(null);
    setSelectedFile(file);
    const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);

    // Read file in memory for inspection and preview
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      if (!text || text.trim().length === 0) {
        setErrorMessage('Uploaded CSV file is empty. Please provide a valid EEG dataset.');
        setSelectedFile(null);
        setCurrentStep(1);
        return;
      }

      const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
      if (lines.length <= 1) {
        setErrorMessage('Uploaded CSV contains a header but no data records.');
        setSelectedFile(null);
        setCurrentStep(1);
        return;
      }

      const rawHeaders = lines[0].split(',').map((h) => h.trim().replace(/^#\s*/, ''));
      const totalCols = rawHeaders.length;
      const featureCols = rawHeaders.filter((c) => c.toLowerCase() !== 'label');
      const rowCount = lines.length - 1;

      // Extract first 5 rows for the preview table (displaying first 7 feature columns)
      const displayHeaders = featureCols.slice(0, 7);
      const rowsPreview: Array<Record<string, string>> = [];

      for (let i = 1; i <= Math.min(5, rowCount); i++) {
        const values = lines[i].split(',');
        const rowObj: Record<string, string> = { _rowNum: String(i) };
        displayHeaders.forEach((col) => {
          const colIdx = rawHeaders.indexOf(col);
          const val = values[colIdx];
          const num = parseFloat(val);
          rowObj[col] = !isNaN(num) ? num.toFixed(3) : val?.trim() || '—';
        });
        rowsPreview.push(rowObj);
      }

      // Check model compatibility
      const isCompat = featureCols.length === 2548;

      setPreviewHeaders(displayHeaders);
      setPreviewRows(rowsPreview);
      setFileStats({
        sizeMB: sizeInMB,
        recordCount: rowCount.toLocaleString(),
        totalColumns: totalCols.toLocaleString(),
        featureCount: featureCols.length.toLocaleString(),
        isCompatible: isCompat,
        missingValues: 0,
      });

      if (!isCompat) {
        setErrorMessage(
          `Dataset feature count mismatch: found ${featureCols.length} features, but the Random Forest model requires exactly 2,548 EEG features.`
        );
      }
    };

    reader.onerror = () => {
      setErrorMessage('Failed to read the selected file. Please verify file permissions.');
      setSelectedFile(null);
      setCurrentStep(1);
    };

    reader.readAsText(file);
    setCurrentStep(2);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const resetUpload = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setIsAnalyzing(false);
    setSelectedFile(null);
    setPredictionResult(null);
    setErrorMessage(null);
    setPreviewHeaders([]);
    setPreviewRows([]);
    setCurrentStep(1);
    setCurrentPage(1);
    setSentimentFilter('ALL');
    setFileStats({
      sizeMB: '0.0',
      recordCount: 'Detecting...',
      totalColumns: 'Detecting...',
      featureCount: '2,548',
      isCompatible: true,
      missingValues: 0,
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const resetAllToInitial = () => {
    resetUpload();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Run Real Machine Learning Analysis via Flask Backend
  const handleRunAnalysis = async () => {
    if (isAnalyzing || !selectedFile) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      // Send actual file to Flask backend endpoint POST /api/predict/batch
      const result = await predictBatchCSV(selectedFile);

      if (result && result.status === 'success') {
        setPredictionResult(result);
        setCurrentStep(3);
        setCurrentPage(1);
        setSentimentFilter('ALL');

        // Update file stats record count to strictly match API response
        setFileStats((prev) => ({
          ...prev,
          recordCount: result.total_samples.toLocaleString(),
        }));

        // Smooth scroll to results
        setTimeout(() => {
          resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
      } else {
        setErrorMessage(
          result?.message || 'Analysis failed. Please verify the EEG dataset format.'
        );
      }
    } catch (err: any) {
      console.error('Batch prediction error:', err);
      const serverMsg =
        err.response?.data?.message ||
        'Failed to connect to the backend server at http://127.0.0.1:5000. Please ensure Flask is running.';
      setErrorMessage(serverMsg);
    } finally {
      // Ensure loading state is immediately cleared in all cases
      setIsAnalyzing(false);
    }
  };

  // Export actual prediction records to CSV directly through browser
  const handleExportCSV = () => {
    if (!predictionResult?.predictions || predictionResult.predictions.length === 0) return;

    const headers = [
      'Record ID',
      'Predicted Sentiment',
      'Positive Probability',
      'Neutral Probability',
      'Negative Probability',
    ];

    const rows = predictionResult.predictions.map((p) => [
      `Record_${p.row_index + 1}`,
      p.predicted_sentiment,
      p.probabilities ? (p.probabilities.POSITIVE ?? 0).toFixed(4) : 'N/A',
      p.probabilities ? (p.probabilities.NEUTRAL ?? 0).toFixed(4) : 'N/A',
      p.probabilities ? (p.probabilities.NEGATIVE ?? 0).toFixed(4) : 'N/A',
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `EEG_Sentiment_Predictions_${predictionResult.filename || 'export'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Calculate actual counts and percentage distribution from live API response
  const totalAnalyzed = predictionResult?.total_samples || 0;
  const posCount = predictionResult?.class_summary?.POSITIVE || 0;
  const neuCount = predictionResult?.class_summary?.NEUTRAL || 0;
  const negCount = predictionResult?.class_summary?.NEGATIVE || 0;

  const posPct = totalAnalyzed > 0 ? (posCount / totalAnalyzed) * 100 : 0;
  const neuPct = totalAnalyzed > 0 ? (neuCount / totalAnalyzed) * 100 : 0;
  const negPct = totalAnalyzed > 0 ? (negCount / totalAnalyzed) * 100 : 0;

  // Donut SVG circumference calculation (r = 62)
  const circumference = 2 * Math.PI * 62; // ~389.56
  const posStroke = (posPct / 100) * circumference;
  const neuStroke = (neuPct / 100) * circumference;
  const negStroke = (negPct / 100) * circumference;

  const posOffset = 0;
  const neuOffset = -posStroke;
  const negOffset = -(posStroke + neuStroke);

  // Filtered prediction items based on user selection
  const allPredictions = predictionResult?.predictions || [];
  const filteredPredictions = allPredictions.filter((item) => {
    if (sentimentFilter === 'ALL') return true;
    return item.predicted_sentiment === sentimentFilter;
  });

  // Table pagination calculations
  const totalPages = Math.max(1, Math.ceil(filteredPredictions.length / ROWS_PER_PAGE));
  const paginatedPredictions: PredictionItem[] = filteredPredictions.slice(
    (currentPage - 1) * ROWS_PER_PAGE,
    currentPage * ROWS_PER_PAGE
  );

  return (
    <div className="flex-1 flex flex-col min-h-screen bg-[#E6F2FC] text-[#203B5D]">
      {/* Page Header */}
      <header className="px-6 lg:px-10 pt-8 pb-4">
        <div className="max-w-5xl mx-auto">
          <h1 className="text-2xl lg:text-[30px] font-bold text-[#203B5D] tracking-tight">
            EEG Sentiment Analysis
          </h1>
          <p className="text-[14px] text-[#71849A] mt-1 font-normal">
            Upload an EEG feature dataset to classify records into Positive, Neutral, or Negative sentiment.
          </p>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 p-6 lg:p-10 pt-2">
        <div className="max-w-5xl mx-auto space-y-6">
          {/* Always-mounted Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                handleFileChange(e.target.files[0]);
              }
            }}
          />

          {/* 3-Step Refined Workflow Indicator */}
          <div className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl px-6 py-4 shadow-sm">
            <div className="flex items-center justify-between max-w-xl mx-auto relative">
              {/* Connector Line 1 */}
              <div
                className={`absolute left-[20%] top-1/2 -translate-y-1/2 w-[28%] h-[2px] transition-colors duration-300 ${
                  currentStep >= 2 ? 'bg-[#00A9D6]' : 'bg-[#D5E8F8]'
                }`}
              />

              {/* Connector Line 2 */}
              <div
                className={`absolute left-[54%] top-1/2 -translate-y-1/2 w-[28%] h-[2px] transition-colors duration-300 ${
                  currentStep === 3 ? 'bg-[#00A9D6]' : 'bg-[#D5E8F8]'
                }`}
              />

              {/* Step 1: Upload Dataset */}
              <div className="flex items-center gap-2 bg-[#FFFFFF] px-2 z-10">
                <span
                  className={`w-6 h-6 rounded-full text-[12px] font-bold flex items-center justify-center transition-colors ${
                    currentStep > 1
                      ? 'bg-[#00A9D6] text-white'
                      : 'bg-[#00A9D6] text-white ring-2 ring-[#00A9D6]/30'
                  }`}
                >
                  {currentStep > 1 ? <CheckCircle2 className="w-4 h-4" /> : '1'}
                </span>
                <span
                  className={`text-[13px] ${
                    currentStep === 1
                      ? 'font-bold text-[#203B5D]'
                      : 'font-medium text-[#71849A]'
                  }`}
                >
                  Upload Dataset
                </span>
              </div>

              {/* Step 2: Validate & Preview */}
              <div className="flex items-center gap-2 bg-[#FFFFFF] px-2 z-10">
                <span
                  className={`w-6 h-6 rounded-full text-[12px] font-bold flex items-center justify-center transition-colors ${
                    currentStep === 2
                      ? 'bg-[#00A9D6] text-white ring-2 ring-[#00A9D6]/30'
                      : currentStep === 3
                      ? 'bg-[#00A9D6] text-white'
                      : 'bg-[#E6F2FC] border border-[#C5DCF2] text-[#71849A]'
                  }`}
                >
                  {currentStep === 3 ? <CheckCircle2 className="w-4 h-4" /> : '2'}
                </span>
                <span
                  className={`text-[13px] ${
                    currentStep === 2
                      ? 'font-bold text-[#203B5D]'
                      : currentStep === 3
                      ? 'font-medium text-[#203B5D]'
                      : 'font-medium text-[#71849A]'
                  }`}
                >
                  Validate & Preview
                </span>
              </div>

              {/* Step 3: View Results */}
              <div className="flex items-center gap-2 bg-[#FFFFFF] px-2 z-10">
                <span
                  className={`w-6 h-6 rounded-full text-[12px] font-bold flex items-center justify-center transition-colors ${
                    currentStep === 3
                      ? 'bg-[#00A9D6] text-white ring-2 ring-[#00A9D6]/30'
                      : 'bg-[#E6F2FC] border border-[#C5DCF2] text-[#71849A]'
                  }`}
                >
                  3
                </span>
                <span
                  className={`text-[13px] ${
                    currentStep === 3
                      ? 'font-bold text-[#203B5D]'
                      : 'font-medium text-[#71849A]'
                  }`}
                >
                  View Results
                </span>
              </div>
            </div>
          </div>

          {/* Error Message Alert Banner */}
          {errorMessage && (
            <div className="flex items-center gap-3 p-4 rounded-xl bg-[#FFF5F5] border border-[#FECACA] text-[#991B1B] text-[13px] shadow-sm animate-in fade-in duration-200">
              <AlertCircle className="w-5 h-5 text-[#DC2626] shrink-0" />
              <p className="flex-1 font-medium">{errorMessage}</p>
              <button
                type="button"
                onClick={() => setErrorMessage(null)}
                className="text-[#991B1B] hover:text-[#7F1D1D] transition-colors cursor-pointer"
                title="Dismiss error"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* SECTION 1: DATASET UPLOAD AREA (Visible when no file is chosen) */}
          {!selectedFile && (
            <section className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-6 lg:p-8 shadow-sm space-y-5">
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl py-12 px-6 text-center flex flex-col items-center justify-center transition-all cursor-pointer bg-[#F8FAFC] ${
                  isDragging
                    ? 'border-[#00A9D6] bg-[#E6F2FC]'
                    : 'border-[#C5DCF2] hover:border-[#00A9D6] hover:bg-[#F0F7FD]'
                }`}
              >
                <div className="w-14 h-14 rounded-xl bg-[#E6F2FC] flex items-center justify-center text-[#00A9D6] border border-[#C5DCF2] mb-3.5 shadow-sm">
                  <UploadCloud className="w-7 h-7 text-[#00A9D6]" />
                </div>
                <h3 className="text-[16px] font-bold text-[#203B5D]">
                  Upload EEG Dataset
                </h3>
                <p className="text-[13px] text-[#71849A] mt-1 max-w-sm">
                  Drag and drop your EEG feature CSV file here, or click to browse.
                </p>

                {/* Explicit Supported Format Badge */}
                <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
                  <span className="inline-block text-[11.5px] font-semibold text-[#203B5D] bg-[#E6F2FC] px-3.5 py-1.5 rounded-md border border-[#C5DCF2] font-mono">
                    Supported Format: CSV (.csv)
                  </span>
                  <span className="inline-block text-[11.5px] font-mono text-[#00A9D6] bg-[#FFFFFF] px-3 py-1.5 rounded-md border border-[#C5DCF2]">
                    Requires 2,548 EEG feature dimensions
                  </span>
                </div>

                <div className="mt-3.5 flex items-center gap-1.5 text-[11.5px] text-[#71849A]">
                  <ShieldCheck className="w-4 h-4 text-[#00A9D6]" />
                  <span>Strict Data Privacy: In-memory transient analysis (no permanent disk or server storage)</span>
                </div>

                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  className="mt-5 px-6 py-2.5 rounded-lg bg-[#00A9D6] hover:bg-[#008FB5] text-white text-[13.5px] font-semibold transition-colors shadow-sm cursor-pointer"
                >
                  Choose CSV File
                </button>
              </div>
            </section>
          )}

          {/* SECTION 2: DATASET VALIDATION & COMPACT PREVIEW (Visible after file selection) */}
          {selectedFile && (
            <section className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-6 lg:p-8 shadow-sm space-y-6">
              {/* Selected File Header */}
              <div className="bg-[#F8FAFC] border border-[#C5DCF2] rounded-xl p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#E2EDF8]">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-[#E6F2FC] text-[#00A9D6] flex items-center justify-center shrink-0 border border-[#C5DCF2]">
                      <FileText className="w-5 h-5 text-[#00A9D6]" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-[14.5px] font-bold font-mono text-[#203B5D]">
                          {selectedFile.name}
                        </span>
                        <span
                          className={`inline-flex items-center gap-1 text-[11px] font-semibold px-2 py-0.5 rounded border ${
                            fileStats.isCompatible
                              ? 'text-[#00A9D6] bg-[#E6F2FC] border-[#00A9D6]/30'
                              : 'text-[#DC2626] bg-[#FFF5F5] border-[#FECACA]'
                          }`}
                        >
                          {fileStats.isCompatible ? (
                            <>
                              <CheckCircle2 className="w-3 h-3 text-[#00A9D6]" />
                              Valid CSV
                            </>
                          ) : (
                            <>
                              <AlertCircle className="w-3 h-3 text-[#DC2626]" />
                              Dimension Mismatch
                            </>
                          )}
                        </span>
                      </div>
                      <p className="text-[12px] text-[#71849A] mt-0.5">
                        {fileStats.isCompatible
                          ? '✓ Fully compatible with trained Random Forest model'
                          : '⚠️ Dataset dimensions must match the 2,548 trained EEG features'}
                      </p>
                    </div>
                  </div>

                  {/* Actions: Replace / Remove File */}
                  <div className="flex items-center gap-2 self-start sm:self-center">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium text-[#203B5D] hover:bg-[#E6F2FC] rounded-md transition-colors border border-[#C5DCF2] bg-[#FFFFFF] cursor-pointer"
                      title="Replace current file"
                    >
                      <RefreshCw className="w-3.5 h-3.5 text-[#00A9D6]" />
                      <span>Change File</span>
                    </button>
                    <button
                      type="button"
                      onClick={resetUpload}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium text-[#DC2626] hover:bg-[#FFF5F5] rounded-md transition-colors border border-[#FECACA] bg-[#FFFFFF] cursor-pointer"
                      title="Remove file"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-[#DC2626]" />
                      <span>Remove File</span>
                    </button>
                  </div>
                </div>

                {/* Validation Metadata Statistics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
                  <div className="bg-[#FFFFFF] rounded-lg p-3 border border-[#C5DCF2]">
                    <span className="text-[10.5px] font-medium tracking-wider uppercase text-[#71849A]">
                      File Size
                    </span>
                    <p className="text-[15px] font-bold font-mono text-[#203B5D] mt-0.5">
                      {fileStats.sizeMB} MB
                    </p>
                  </div>
                  <div className="bg-[#FFFFFF] rounded-lg p-3 border border-[#C5DCF2]">
                    <span className="text-[10.5px] font-medium tracking-wider uppercase text-[#71849A]">
                      Detected Records
                    </span>
                    <p className="text-[15px] font-bold font-mono text-[#203B5D] mt-0.5">
                      {fileStats.recordCount}
                    </p>
                  </div>
                  <div className="bg-[#FFFFFF] rounded-lg p-3 border border-[#C5DCF2]">
                    <span className="text-[10.5px] font-medium tracking-wider uppercase text-[#71849A]">
                      Feature Columns
                    </span>
                    <p className="text-[15px] font-bold font-mono text-[#203B5D] mt-0.5">
                      {fileStats.featureCount}
                    </p>
                  </div>
                  <div className="bg-[#FFFFFF] rounded-lg p-3 border border-[#C5DCF2]">
                    <span className="text-[10.5px] font-medium tracking-wider uppercase text-[#71849A]">
                      Missing Values
                    </span>
                    <p className="text-[15px] font-bold font-mono text-[#00A9D6] mt-0.5">
                      {fileStats.missingValues} (None)
                    </p>
                  </div>
                </div>

                {/* Compact Table Preview: First 5 records */}
                {previewRows.length > 0 && (
                  <div className="pt-2 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5 text-[12.5px] font-bold text-[#203B5D]">
                        <TableIcon className="w-3.5 h-3.5 text-[#00A9D6]" />
                        <span>Dataset Preview (First 5 Records)</span>
                      </div>
                      <span className="text-[11px] text-[#71849A] font-mono">
                        Showing 7 of {fileStats.featureCount} feature columns
                      </span>
                    </div>

                    <div className="border border-[#C5DCF2] rounded-lg overflow-hidden bg-[#FFFFFF]">
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-[11.5px]">
                          <thead className="bg-[#E6F2FC] text-[#203B5D] border-b border-[#C5DCF2] font-mono text-[10.5px] uppercase tracking-wider">
                            <tr>
                              <th className="px-3 py-2 w-16">Row #</th>
                              {previewHeaders.map((col) => (
                                <th key={col} className="px-3 py-2 whitespace-nowrap">
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-[#E2EDF8] font-mono">
                            {previewRows.map((r, i) => (
                              <tr key={i} className="hover:bg-[#F8FAFC]">
                                <td className="px-3 py-2 text-[#71849A] font-bold">
                                  #{r._rowNum}
                                </td>
                                {previewHeaders.map((col) => (
                                  <td key={col} className="px-3 py-2 text-[#203B5D] whitespace-nowrap">
                                    {r[col]}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Primary Analyze Action Button */}
              {!predictionResult && (
                <div className="flex justify-center pt-1">
                  <button
                    type="button"
                    onClick={handleRunAnalysis}
                    disabled={isAnalyzing || !fileStats.isCompatible}
                    className={`inline-flex items-center gap-2.5 px-8 py-3 rounded-lg bg-[#00A9D6] hover:bg-[#008FB5] text-white font-semibold text-[14px] shadow-sm transition-all cursor-pointer ${
                      isAnalyzing ? 'opacity-70 cursor-wait' : ''
                    } ${!fileStats.isCompatible ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin text-white" />
                        <span>Analyzing EEG Dataset...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-current text-white" />
                        <span>Analyze Dataset</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </section>
          )}

          {/* SECTION 3: PROFESSIONAL RESULTS DASHBOARD (Visible after successful prediction) */}
          {predictionResult && (
            <section
              ref={resultsRef}
              className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-6 lg:p-8 shadow-sm space-y-7 animate-in fade-in duration-300"
            >
              {/* Results Top Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-[#E2EDF8]">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-[#E6F2FC] text-[#00A9D6] flex items-center justify-center border border-[#C5DCF2]">
                    <BarChart3 className="w-5 h-5 text-[#00A9D6]" />
                  </div>
                  <div>
                    <h2 className="text-[19px] font-bold text-[#203B5D]">
                      Prediction Results
                    </h2>
                    <p className="text-[12.5px] text-[#71849A]">
                      Supervised classification evaluated with the trained Random Forest model
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 self-start sm:self-center">
                  <span className="inline-flex items-center gap-1.5 text-[12px] font-bold text-[#00A9D6] bg-[#E6F2FC] px-3 py-1.5 rounded-md border border-[#C5DCF2] font-mono">
                    <span className="w-2 h-2 rounded-full bg-[#00A9D6]"></span>
                    {totalAnalyzed.toLocaleString()} Records Analyzed
                  </span>
                  <button
                    type="button"
                    onClick={handleExportCSV}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-[12.5px] font-semibold text-[#203B5D] bg-[#FFFFFF] hover:bg-[#E6F2FC] border border-[#C5DCF2] rounded-md transition-colors cursor-pointer shadow-sm"
                    title="Export predictions as CSV"
                  >
                    <Download className="w-3.5 h-3.5 text-[#00A9D6]" />
                    <span>Download Results CSV</span>
                  </button>
                </div>
              </div>

              {/* Dynamic Analysis Summary Sentence */}
              <div className="p-4 rounded-lg bg-[#E6F2FC] border border-[#C5DCF2] text-[13.5px] text-[#203B5D] leading-relaxed">
                The uploaded dataset contains <strong>{totalAnalyzed}</strong> records. The model classified{' '}
                <strong className="text-[#00A9D6]">{posCount}</strong> records as Positive,{' '}
                <strong className="text-[#71849A]">{neuCount}</strong> as Neutral, and{' '}
                <strong className="text-[#203B5D]">{negCount}</strong> as Negative.
              </div>

              {/* Visualization Grid: Donut Chart + 3 Sentiment Cards */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center py-2">
                {/* SVG Donut Chart + Legend */}
                <div className="lg:col-span-5 flex flex-col items-center justify-center gap-6">
                  {/* Donut Chart with Centered Total */}
                  <div className="relative w-48 h-48 shrink-0 flex items-center justify-center">
                    <svg className="w-48 h-48 transform -rotate-90" viewBox="0 0 160 160">
                      {/* Background Track */}
                      <circle
                        cx="80"
                        cy="80"
                        fill="none"
                        r="62"
                        stroke="#E6F2FC"
                        strokeWidth="16"
                      />
                      {/* Positive Segment (Bright Cyan) */}
                      <circle
                        cx="80"
                        cy="80"
                        fill="none"
                        r="62"
                        stroke="#00A9D6"
                        strokeWidth="16"
                        strokeDasharray={`${posStroke} ${circumference}`}
                        strokeDashoffset={posOffset}
                        className="transition-all duration-700"
                      />
                      {/* Neutral Segment (Slate Grey) */}
                      <circle
                        cx="80"
                        cy="80"
                        fill="none"
                        r="62"
                        stroke="#71849A"
                        strokeWidth="16"
                        strokeDasharray={`${neuStroke} ${circumference}`}
                        strokeDashoffset={neuOffset}
                        className="transition-all duration-700"
                      />
                      {/* Negative Segment (Navy Blue) */}
                      <circle
                        cx="80"
                        cy="80"
                        fill="none"
                        r="62"
                        stroke="#203B5D"
                        strokeWidth="16"
                        strokeDasharray={`${negStroke} ${circumference}`}
                        strokeDashoffset={negOffset}
                        className="transition-all duration-700"
                      />
                    </svg>
                    {/* Centered Total Count */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-center select-none pointer-events-none">
                      <span className="text-2xl font-bold font-mono text-[#203B5D] leading-none">
                        {totalAnalyzed}
                      </span>
                      <span className="text-[10px] text-[#71849A] font-medium uppercase tracking-wider mt-1">
                        Total Records
                      </span>
                    </div>
                  </div>

                  {/* Clean Legend */}
                  <div className="flex items-center justify-center gap-5 text-[12.5px]">
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#00A9D6]" />
                      <span className="text-[#203B5D] font-medium">Positive</span>
                      <span className="font-mono text-[#71849A] ml-0.5">
                        {posPct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#71849A]" />
                      <span className="text-[#203B5D] font-medium">Neutral</span>
                      <span className="font-mono text-[#71849A] ml-0.5">
                        {neuPct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#203B5D]" />
                      <span className="text-[#203B5D] font-medium">Negative</span>
                      <span className="font-mono text-[#71849A] ml-0.5">
                        {negPct.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>

                <div className="hidden lg:block lg:col-span-1 h-32 w-[1px] bg-[#E2EDF8] mx-auto" />

                {/* 3 Sentiment Highlight Cards */}
                <div className="lg:col-span-6 grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {/* Positive Card */}
                  <div className="bg-[#FFFFFF] border border-[#00A9D6]/30 rounded-xl p-4 flex flex-col justify-between shadow-sm">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[12.5px] font-semibold text-[#203B5D]">
                          Positive Sentiment
                        </span>
                        <span className="w-2 h-2 rounded-full bg-[#00A9D6]" />
                      </div>
                      <span className="text-[22px] font-bold font-mono text-[#00A9D6] block">
                        {posPct.toFixed(1)}%
                      </span>
                    </div>
                    <span className="text-[11px] font-mono text-[#71849A] mt-3 block pt-2 border-t border-[#E2EDF8]">
                      {posCount.toLocaleString()} records
                    </span>
                  </div>

                  {/* Neutral Card */}
                  <div className="bg-[#FFFFFF] border border-[#C5DCF2] rounded-xl p-4 flex flex-col justify-between shadow-sm">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[12.5px] font-semibold text-[#203B5D]">
                          Neutral Sentiment
                        </span>
                        <span className="w-2 h-2 rounded-full bg-[#71849A]" />
                      </div>
                      <span className="text-[22px] font-bold font-mono text-[#71849A] block">
                        {neuPct.toFixed(1)}%
                      </span>
                    </div>
                    <span className="text-[11px] font-mono text-[#71849A] mt-3 block pt-2 border-t border-[#E2EDF8]">
                      {neuCount.toLocaleString()} records
                    </span>
                  </div>

                  {/* Negative Card */}
                  <div className="bg-[#FFFFFF] border border-[#203B5D]/30 rounded-xl p-4 flex flex-col justify-between shadow-sm">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[12.5px] font-semibold text-[#203B5D]">
                          Negative Sentiment
                        </span>
                        <span className="w-2 h-2 rounded-full bg-[#203B5D]" />
                      </div>
                      <span className="text-[22px] font-bold font-mono text-[#203B5D] block">
                        {negPct.toFixed(1)}%
                      </span>
                    </div>
                    <span className="text-[11px] font-mono text-[#71849A] mt-3 block pt-2 border-t border-[#E2EDF8]">
                      {negCount.toLocaleString()} records
                    </span>
                  </div>
                </div>
              </div>

              {/* INDIVIDUAL PREDICTION RESULTS TABLE */}
              {allPredictions.length > 0 && (
                <div className="pt-2 space-y-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <h3 className="text-[15px] font-bold text-[#203B5D]">
                        Individual Prediction Results
                      </h3>
                      <p className="text-[12px] text-[#71849A]">
                        Record-by-record classifications generated by the Random Forest model
                      </p>
                    </div>

                    {/* Filter Pills */}
                    <div className="flex items-center gap-1.5 self-start sm:self-center bg-[#F8FAFC] p-1 rounded-lg border border-[#C5DCF2] text-[12px]">
                      <SlidersHorizontal className="w-3.5 h-3.5 text-[#71849A] ml-1.5 mr-0.5" />
                      {(['ALL', 'POSITIVE', 'NEUTRAL', 'NEGATIVE'] as const).map((filter) => {
                        const count =
                          filter === 'ALL'
                            ? totalAnalyzed
                            : filter === 'POSITIVE'
                            ? posCount
                            : filter === 'NEUTRAL'
                            ? neuCount
                            : negCount;

                        const isActive = sentimentFilter === filter;
                        return (
                          <button
                            key={filter}
                            type="button"
                            onClick={() => {
                              setSentimentFilter(filter);
                              setCurrentPage(1);
                            }}
                            className={`px-2.5 py-1 rounded-md font-medium text-[11.5px] transition-colors cursor-pointer ${
                              isActive
                                ? 'bg-[#00A9D6] text-white font-semibold'
                                : 'text-[#71849A] hover:text-[#203B5D]'
                            }`}
                          >
                            {filter === 'ALL' ? 'All' : filter.charAt(0) + filter.slice(1).toLowerCase()} ({count})
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Table Container */}
                  <div className="border border-[#C5DCF2] rounded-lg overflow-hidden bg-[#FFFFFF]">
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-[12.5px]">
                        <thead className="bg-[#E6F2FC] text-[#203B5D] border-b border-[#C5DCF2] uppercase text-[11px] font-mono tracking-wider">
                          <tr>
                            <th className="px-4 py-2.5">Record ID</th>
                            <th className="px-4 py-2.5">Predicted Sentiment</th>
                            <th className="px-4 py-2.5">Probability (Pos / Neu / Neg)</th>
                            <th className="px-4 py-2.5 text-right">Model Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#E2EDF8] font-mono">
                          {paginatedPredictions.map((row) => {
                            const isPos = row.predicted_sentiment === 'POSITIVE';
                            const isNeu = row.predicted_sentiment === 'NEUTRAL';

                            return (
                              <tr key={row.row_index} className="hover:bg-[#F0F7FD] transition-colors">
                                <td className="px-4 py-2.5 text-[#203B5D] font-medium">
                                  #{String(row.row_index + 1).padStart(3, '0')}
                                </td>
                                <td className="px-4 py-2.5">
                                  <span
                                    className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-bold ${
                                      isPos
                                        ? 'bg-[#E6F2FC] text-[#00A9D6] border border-[#00A9D6]/30'
                                        : isNeu
                                        ? 'bg-[#F1F5F9] text-[#71849A] border border-[#CBD5E1]'
                                        : 'bg-[#EBF2FA] text-[#203B5D] border border-[#203B5D]/30'
                                    }`}
                                  >
                                    {row.predicted_sentiment}
                                  </span>
                                </td>
                                <td className="px-4 py-2.5 text-[#71849A] text-[11.5px]">
                                  {row.probabilities ? (
                                    <span>
                                      {(row.probabilities.POSITIVE * 100).toFixed(0)}% /{' '}
                                      {(row.probabilities.NEUTRAL * 100).toFixed(0)}% /{' '}
                                      {(row.probabilities.NEGATIVE * 100).toFixed(0)}%
                                    </span>
                                  ) : (
                                    '—'
                                  )}
                                </td>
                                <td className="px-4 py-2.5 text-right">
                                  <span className="inline-flex items-center gap-1 text-[11px] text-[#71849A]">
                                    <span className="w-1.5 h-1.5 rounded-full bg-[#00A9D6]"></span>
                                    Classified
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>

                    {/* Pagination Controls */}
                    <div className="flex items-center justify-between px-4 py-2.5 bg-[#E6F2FC] border-t border-[#C5DCF2] text-[12px]">
                      <span className="text-[#71849A]">
                        Showing {filteredPredictions.length === 0 ? 0 : (currentPage - 1) * ROWS_PER_PAGE + 1}–
                        {Math.min(currentPage * ROWS_PER_PAGE, filteredPredictions.length)} of{' '}
                        {filteredPredictions.length} records
                      </span>
                      {totalPages > 1 && (
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                            disabled={currentPage === 1}
                            className="p-1 rounded bg-[#FFFFFF] border border-[#C5DCF2] text-[#71849A] hover:text-[#203B5D] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                          >
                            <ChevronLeft className="w-4 h-4" />
                          </button>
                          <span className="px-2 text-[#203B5D] font-mono text-[11.5px]">
                            {currentPage} / {totalPages}
                          </span>
                          <button
                            type="button"
                            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                            disabled={currentPage === totalPages}
                            className="p-1 rounded bg-[#FFFFFF] border border-[#C5DCF2] text-[#71849A] hover:text-[#203B5D] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                          >
                            <ChevronRight className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Bottom Actions: Analyze Another Dataset & Reset Analysis */}
              <div className="pt-4 border-t border-[#E2EDF8] flex flex-wrap items-center justify-center gap-3">
                <button
                  type="button"
                  onClick={() => {
                    setPredictionResult(null);
                    setCurrentStep(1);
                    setSelectedFile(null);
                    setPreviewHeaders([]);
                    setPreviewRows([]);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-[#00A9D6] hover:bg-[#008FB5] text-white font-semibold text-[13px] shadow-sm transition-colors cursor-pointer"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Analyze Another Dataset</span>
                </button>

                <button
                  type="button"
                  onClick={resetAllToInitial}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-[#C5DCF2] text-[#71849A] hover:text-[#203B5D] hover:bg-[#E6F2FC] font-medium text-[13px] transition-colors cursor-pointer bg-[#FFFFFF]"
                >
                  <RotateCcw className="w-4 h-4" />
                  <span>Reset Analysis</span>
                </button>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
};
