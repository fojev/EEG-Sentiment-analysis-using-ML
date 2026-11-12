/**
 * TypeScript Type Definitions for EEG Sentiment Analysis Application
 */

export type SentimentClass = 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';

export interface ClassProbabilities {
  POSITIVE: number;
  NEUTRAL: number;
  NEGATIVE: number;
}

export interface PredictionItem {
  row_index: number;
  predicted_sentiment: SentimentClass;
  probabilities?: ClassProbabilities;
}

export interface ClassSummary {
  POSITIVE: number;
  NEUTRAL: number;
  NEGATIVE: number;
}

export interface BatchPredictionResponse {
  status: 'success' | 'error';
  filename?: string;
  total_samples: number;
  class_summary: ClassSummary;
  predictions: PredictionItem[];
  latency_ms: number;
  algorithm_used: string;
  message?: string;
}

export interface HealthResponse {
  status: string;
  backend: string;
  project_name: string;
  model_loaded: boolean;
  selected_algorithm: string;
  expected_features_count: number;
  classes: SentimentClass[];
}

export interface MetricsResponse {
  status: string;
  selected_algorithm: string;
  classes: SentimentClass[];
  cross_validation_accuracy: number;
  cross_validation_std: number;
  final_test_accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
  top_10_features?: Array<{ feature: string; importance: number }>;
}

export interface DatasetPreviewResponse {
  status: string;
  total_records: number;
  total_features: number;
  target_labels: string[];
  class_distribution: Record<string, number>;
  first_five_records: Array<{
    row_index: number;
    label: string;
    sample_features: Record<string, number>;
  }>;
}

