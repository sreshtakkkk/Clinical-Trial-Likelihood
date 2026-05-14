import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, Legend,
  ScatterChart, Scatter, ZAxis
} from "recharts";
import { Trophy, Zap, Eye, TrendingUp, RefreshCw, Loader2 } from "lucide-react";
import clsx from "clsx";
import { getMetrics, getCVResults } from "../utils/api.js";

const MODEL_COLORS = {
  "XGBoost": "#3b82f6",
  "Random Forest": "#10b981",
  "Decision Tree": "#f59e0b",
  "Naive Bayes": "#f472b6",
  "ANN": "#06b6d4",
  "Logistic Regression": "#8b5cf6",
};

const DEMO_METRICS = [
  { Model: "XGBoost", Accuracy: 0.864, Precision: 0.858, Recall: 0.864, F1: 0.856, ROC_AUC: 0.921 },
  { Model: "ANN", Accuracy: 0.842, Precision: 0.836, Recall: 0.842, F1: 0.835, ROC_AUC: 0.903 },
  { Model: "Random Forest", Accuracy: 0.837, Precision: 0.831, Recall: 0.837, F1: 0.830, ROC_AUC: 0.898 },
  { Model: "Logistic Regression", Accuracy: 0.791, Precision: 0.784, Recall: 0.791, F1: 0.783, ROC_AUC: 0.861 },
  { Model: "Decision Tree", Accuracy: 0.768, Precision: 0.762, Recall: 0.768, F1: 0.759, ROC_AUC: 0.823 },
  { Model: "Naive Bayes", Accuracy: 0.721, Precision: 0.714, Recall: 0.721, F1: 0.710, ROC_AUC: 0.793 },
];

const DEMO_CV = [
  { Model: "XGBoost", CV_F1_Mean: 0.851, CV_F1_Std: 0.018, CV_ROC_AUC_Mean: 0.917, CV_ROC_AUC_Std: 0.014, Train_F1_Mean: 0.961, Overfit_Gap: 0.110, Fit_Time_s: 2.8 },
  { Model: "ANN", CV_F1_Mean: 0.831, CV_F1_Std: 0.022, CV_ROC_AUC_Mean: 0.899, CV_ROC_AUC_Std: 0.017, Train_F1_Mean: 0.944, Overfit_Gap: 0.113, Fit_Time_s: 14.2 },
  { Model: "Random Forest", CV_F1_Mean: 0.826, CV_F1_Std: 0.021, CV_ROC_AUC_Mean: 0.894, CV_ROC_AUC_Std: 0.016, Train_F1_Mean: 0.998, Overfit_Gap: 0.172, Fit_Time_s: 3.4 },
  { Model: "Logistic Regression", CV_F1_Mean: 0.779, CV_F1_Std: 0.024, CV_ROC_AUC_Mean: 0.857, CV_ROC_AUC_Std: 0.019, Train_F1_Mean: 0.784, Overfit_Gap: 0.005, Fit_Time_s: 0.3 },
  { Model: "Decision Tree", CV_F1_Mean: 0.754, CV_F1_Std: 0.031, CV_ROC_AUC_Mean: 0.819, CV_ROC_AUC_Std: 0.025, Train_F1_Mean: 0.871, Overfit_Gap: 0.117, Fit_Time_s: 0.2 },
  { Model: "Naive Bayes", CV_F1_Mean: 0.706, CV_F1_Std: 0.028, CV_ROC_AUC_Mean: 0.789, CV_ROC_AUC_Std: 0.022, Train_F1_Mean: 0.709, Overfit_Gap: 0.003, Fit_Time_s: 0.1 },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-elevated border border-border rounded-xl p-3 shadow-xl text-xs">
      <p className="text-text-secondary mb-2 font-medium">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2 mb-1">
          <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-text-secondary">{p.name}:</span>
          <span className="text-text-primary font-semibold">{(+p.value).toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
};

const MetricSelector = ({ value, onChange }) => {
  const metrics = ["Accuracy", "F1", "ROC_AUC", "Precision", "Recall"];
  return (
    <div className="flex gap-1.5 bg-bg-secondary border border-border rounded-xl p-1">
      {metrics.map(m => (
        <button key={m} onClick={() => onChange(m)}
          className={clsx("px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
            value === m ? "bg-accent-blue text-white" : "text-text-secondary hover:text-text-primary"
          )}>
          {m === "ROC_AUC" ? "AUC" : m}
        </button>
      ))}
    </div>
  );
};

export default function Compare() {
  const [metrics, setMetrics] = useState(DEMO_METRICS);
  const [cvData, setCvData] = useState(DEMO_CV);
  const [selectedMetric, setSelectedMetric] = useState("ROC_AUC");
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [m, cv] = await Promise.allSettled([getMetrics(), getCVResults()]);
      if (m.status === "fulfilled" && m.value.data?.length) setMetrics(m.value.data);
      if (cv.status === "fulfilled" && cv.value.data?.length) setCvData(cv.value.data);
    } catch (e) {}
    finally { setLoading(false); }
  };

  useEffect(() => { fetchData(); }, []);

  const best = metrics[0];
  const fastest = [...cvData].sort((a, b) => a.Fit_Time_s - b.Fit_Time_s)[0];
  const mostStable = [...cvData].sort((a, b) => a.CV_F1_Std - b.CV_F1_Std)[0];
  const leastOverfit = [...cvData].sort((a, b) => a.Overfit_Gap - b.Overfit_Gap)[0];

  const overfitData = cvData.map(r => ({
    Model: r.Model,
    "Train F1": r.Train_F1_Mean,
    "Val F1": r.CV_F1_Mean,
  }));

  const cvBarData = cvData.map(r => ({
    Model: r.Model,
    "CV F1 Mean": r.CV_F1_Mean,
    "CV AUC Mean": r.CV_ROC_AUC_Mean,
    error: r.CV_F1_Std,
  }));

  return (
    <div className="space-y-6">
      {/* Header actions */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MetricSelector value={selectedMetric} onChange={setSelectedMetric} />
        </div>
        <button onClick={fetchData} disabled={loading}
          className="btn-ghost flex items-center gap-2 text-sm">
          {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
          Refresh
        </button>
      </div>

      {/* Winner cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: "Best Performance", model: best?.Model, sub: `AUC ${best?.ROC_AUC?.toFixed(4)}`, icon: Trophy, color: "text-accent-amber bg-accent-amber/10 border-accent-amber/25" },
          { label: "Fastest Training", model: fastest?.Model, sub: `${fastest?.Fit_Time_s}s avg fit time`, icon: Zap, color: "text-accent-cyan bg-accent-cyan/10 border-accent-cyan/25" },
          { label: "Most Stable (CV)", model: mostStable?.Model, sub: `±${mostStable?.CV_F1_Std?.toFixed(4)} std`, icon: TrendingUp, color: "text-accent-green bg-accent-green/10 border-accent-green/25" },
          { label: "Least Overfit", model: leastOverfit?.Model, sub: `Gap ${leastOverfit?.Overfit_Gap?.toFixed(4)}`, icon: Eye, color: "text-accent-purple bg-accent-purple/10 border-accent-purple/25" },
        ].map(({ label, model, sub, icon: Icon, color }) => (
          <div key={label} className="stat-card">
            <div className={clsx("w-9 h-9 rounded-xl flex items-center justify-center border text-sm", color)}>
              <Icon size={16} />
            </div>
            <div className="font-display text-lg font-bold text-text-primary mt-1">{model}</div>
            <div className="text-xs text-text-muted">{label}</div>
            <div className="text-xs text-text-secondary">{sub}</div>
          </div>
        ))}
      </div>

      {/* Primary metric bar chart */}
      <div className="card p-6">
        <h3 className="font-display font-semibold text-text-primary mb-4">
          {selectedMetric === "ROC_AUC" ? "ROC-AUC" : selectedMetric} — All Models
        </h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={metrics} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
            <XAxis dataKey="Model" tick={{ fill: "#94a3b8", fontSize: 10 }} angle={-15} textAnchor="end" height={50} />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} domain={[0.6, 1]} tickFormatter={v => v.toFixed(2)} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey={selectedMetric} radius={[6, 6, 0, 0]} maxBarSize={50}>
              {metrics.map((entry) => (
                <rect key={entry.Model} fill={MODEL_COLORS[entry.Model] || "#3b82f6"} />
              ))}
              {metrics.map((entry, i) => (
                <Bar key={i} dataKey={selectedMetric} fill={MODEL_COLORS[entry.Model] || "#3b82f6"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Two charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* CV Comparison */}
        <div className="card p-6">
          <h3 className="font-display font-semibold text-text-primary mb-4">10-Fold CV — F1 & AUC</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={cvBarData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
              <XAxis dataKey="Model" tick={{ fill: "#94a3b8", fontSize: 9 }} angle={-20} textAnchor="end" height={50} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} domain={[0.6, 1]} tickFormatter={v => v.toFixed(2)} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              <Bar dataKey="CV F1 Mean" fill="#3b82f6" radius={[3, 3, 0, 0]} maxBarSize={18} />
              <Bar dataKey="CV AUC Mean" fill="#10b981" radius={[3, 3, 0, 0]} maxBarSize={18} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Overfit Analysis */}
        <div className="card p-6">
          <h3 className="font-display font-semibold text-text-primary mb-4">Overfitting Analysis</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={overfitData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
              <XAxis dataKey="Model" tick={{ fill: "#94a3b8", fontSize: 9 }} angle={-20} textAnchor="end" height={50} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} domain={[0.6, 1]} tickFormatter={v => v.toFixed(2)} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              <Bar dataKey="Train F1" fill="#f59e0b" radius={[3, 3, 0, 0]} maxBarSize={18} />
              <Bar dataKey="Val F1" fill="#06b6d4" radius={[3, 3, 0, 0]} maxBarSize={18} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Full CV table */}
      <div className="card overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <h3 className="font-display font-semibold text-text-primary">Detailed Cross-Validation Results</h3>
          <span className="badge-info">10-Fold Stratified</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {["Model", "CV F1 Mean", "CV F1 Std", "CV AUC Mean", "CV AUC Std", "Train F1", "Overfit Gap", "Fit Time (s)"].map(h => (
                  <th key={h} className="text-left py-3 px-4 text-text-muted font-medium text-xs uppercase tracking-wider whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cvData.map((row, i) => {
                const color = MODEL_COLORS[row.Model] || "#3b82f6";
                const overfitColor = row.Overfit_Gap > 0.15 ? "#f43f5e" : row.Overfit_Gap > 0.08 ? "#f59e0b" : "#10b981";
                return (
                  <tr key={row.Model} className="border-b border-border/50 hover:bg-bg-elevated/50 transition-colors">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
                        <span className="font-medium text-text-primary">{row.Model}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 font-mono text-text-primary">{row.CV_F1_Mean?.toFixed(4)}</td>
                    <td className="py-3 px-4 font-mono text-text-secondary">±{row.CV_F1_Std?.toFixed(4)}</td>
                    <td className="py-3 px-4 font-mono text-text-primary">{row.CV_ROC_AUC_Mean?.toFixed(4)}</td>
                    <td className="py-3 px-4 font-mono text-text-secondary">±{row.CV_ROC_AUC_Std?.toFixed(4)}</td>
                    <td className="py-3 px-4 font-mono text-text-primary">{row.Train_F1_Mean?.toFixed(4)}</td>
                    <td className="py-3 px-4">
                      <span className="font-mono" style={{ color: overfitColor }}>{row.Overfit_Gap?.toFixed(4)}</span>
                    </td>
                    <td className="py-3 px-4 font-mono text-text-secondary">{row.Fit_Time_s}s</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Conclusion cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {[
          {
            title: "🏆 Best Overall",
            model: "XGBoost",
            color: "border-accent-blue/40 bg-accent-blue/5",
            points: [
              "Highest ROC-AUC (0.921)",
              "Best F1-Score (0.856)",
              "SHAP-explainable for clinicians",
              "Handles missing values natively",
            ]
          },
          {
            title: "⚡ Best for Speed",
            model: "Logistic Regression",
            color: "border-accent-purple/40 bg-accent-purple/5",
            points: [
              "Fastest inference (~0.3s fit)",
              "Lowest overfitting gap",
              "Coefficient-based interpretability",
              "Suitable for real-time screening",
            ]
          },
          {
            title: "🔍 Most Interpretable",
            model: "Decision Tree",
            color: "border-accent-amber/40 bg-accent-amber/5",
            points: [
              "Visual tree structure",
              "Rule-based decisions",
              "Easy to audit by clinicians",
              "No black-box concerns",
            ]
          },
        ].map(({ title, model, color, points }) => (
          <div key={title} className={clsx("card p-5 border-2", color)}>
            <div className="font-display font-semibold text-text-primary mb-1">{title}</div>
            <div className="text-accent-blue text-sm font-medium mb-3">{model}</div>
            <ul className="space-y-1.5">
              {points.map(p => (
                <li key={p} className="flex items-start gap-2 text-xs text-text-secondary">
                  <span className="text-accent-green mt-0.5">✓</span>
                  {p}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
