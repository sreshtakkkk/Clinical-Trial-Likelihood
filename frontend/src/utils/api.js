import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    console.error("API Error:", err?.response?.data || err.message);
    return Promise.reject(err);
  }
);

export const healthCheck = () => api.get("/health");
export const getModels = () => api.get("/models");
export const getMetrics = () => api.get("/metrics");
export const getCVResults = () => api.get("/cv-results");
export const getVisualizations = () => api.get("/visualizations");

export const predictSingle = (payload) => api.post("/predict", payload);

export const predictBatch = (file, model = "XGBoost") => {
  const formData = new FormData();
  formData.append("file", file);
  return api.post(`/predict/batch?model=${model}`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const getShapExplanation = (trialId) => api.get(`/shap/${trialId}`);

export default api;
