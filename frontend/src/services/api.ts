/**
 * API Service for communicating with the Flask backend.
 * Uses Axios to handle HTTP requests and typed responses.
 */

import axios from 'axios';
import {
  HealthResponse,
  MetricsResponse,
  DatasetPreviewResponse,
  BatchPredictionResponse,
} from '../types';

// Use relative URL so Vite proxy directs /api to http://127.0.0.1:5000 in dev
// with fallback to direct host if running standalone
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60s timeout for large batch CSV uploads
});

/**
 * Health check to verify Flask backend status and model readiness
 */
export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
};

/**
 * Fetch evaluated model performance metrics from metrics.json
 */
export const getMetrics = async (): Promise<MetricsResponse> => {
  const response = await apiClient.get<MetricsResponse>('/metrics');
  return response.data;
};

/**
 * Fetch dataset preview and high-level column metadata
 */
export const getDatasetPreview = async (): Promise<DatasetPreviewResponse> => {
  const response = await apiClient.get<DatasetPreviewResponse>('/dataset/preview');
  return response.data;
};

/**
 * Upload a CSV file for batch sentiment prediction using the trained Random Forest model.
 * Sends multipart/form-data with the 'file' field.
 */
export const predictBatchCSV = async (file: File): Promise<BatchPredictionResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<BatchPredictionResponse>('/predict/batch', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

