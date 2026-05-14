import React, { useState, useEffect } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend, Cell
} from "recharts";
import {
  FlaskConical, TrendingUp, Award, Zap, Shield,
  Brain, GitBranch, Activity, ChevronUp, ChevronDown
} from "lucide-react";
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

const MODEL_ICONS = {
  "XGBoost": Brain,
  "Random Forest": GitBranch,
  "Decision Tree": Activity,
  "Naive Bayes": Zap,
  "ANN": FlaskConical,
  "Logistic Regression": TrendingUp,
};

// Demo data (used when API is offline)
const DEMO_METRICS = [
  { Model: "XGBoost", Accuracy: 0.864, Precision: 0.858, Recall: 0.864, F1: 0.856, ROC_AUC: 0.921 },
  { Model: "ANN", Accuracy: 0.842, Precision: 0.836, Recall: 0.842, F1: 0.835, ROC_AUC: 0.903 },
  { Model: "Random Forest", Accuracy: 0.837, Precision: 0.831, Recall: 0.837, F1: 0.830, ROC_AUC: 0.898 },
  { Model: "Logistic Regression", Accuracy: 0.791, Precision: 0.784, Recall: 0.791, F1: 0.783, ROC_AUC: 0.861 },
  { Model: "Decision Tree", Accuracy: 0.768, Precision: 0.762, Recall: 0.768, F1: 0.759, ROC_AUC: 0.823 },
  { Model: "Naive Bayes", Accuracy: 0.721, Precision: 0.714, Recall: 0.721, F1: 0.710, ROC_AUC: 0.793 },
];

const STAT_CARDS = [
  { label: "Best Model", value: "XGBoost", sub: "ROC-AUC 0.921", icon: Award, color: "blue" },
  { label: "Avg Accuracy", value: "80.4%", sub: "Across 6 models", icon: TrendingUp, color: "green" },
  { label: "Training Samples", value: "~45K", sub: "After SMOTE", icon: FlaskConical, color: "purple" },
  { label: "Feature Dims", value: "80+", sub: "TF-IDF + molecular", icon: Shield, color: "amber" },
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

const colorMap = {
  blue: "text-accent-blue bg-accent-blue/10 border-accent-blue/25",
  green: "text-accent-green bg-accent-green/10 border-accent-green/25",
  purple: "text-accent-purple bg-accent-purple/10 border-accent-purple/25",
  amber: "text-accent-amber bg-accent-amber/10 border-accent-amber/25",
};

export default function Dashboard() {
  const [metrics, setMetrics] = useState(DEMO_METRICS);
  const [loading, setLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState("demo");

  useEffect(() => {
    getMetrics()
      .then(r => {
        if (r.data?.length) {
          setMetrics(r.data);
          setApiStatus("live");
        }
      })
      .catch(() => setApiStatus("demo"))
      .finally(() => setLoading(false));
  }, []);

  const best = metrics[0] || {};
  const radarData = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"].map(k => ({
    metric: k === "ROC_AUC" ? "AUC" : k,
    XGBoost: metrics.find(m => m.Model === "XGBoost")?.[k] || 0,
    ANN: metrics.find(m => m.Model === "ANN")?.[k] || 0,
    "Random Forest": metrics.find(m => m.Model === "Random Forest")?.[k] || 0,
  }));

  return (
    <div className="space-y-6">
      {/* Status banner */}
      <div className={clsx(
        "flex items-center gap-2 text-xs px-4 py-2 rounded-lg border w-fit",
        apiStatus === "live"
          ? "bg-accent-green/10 border-accent-green/25 text-accent-green"
          : "bg-accent-amber/10 border-accent-amber/25 text-accent-amber"
      )}>
        <span className={clsx("w-1.5 h-1.5 rounded-full animate-pulse", apiStatus === "live" ? "bg-accent-green" : "bg-accent-amber")} />
        {apiStatus === "live" ? "Live API data" : "Demo mode — connect API for live data"}
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {STAT_CARDS.map(({ label, value, sub, icon: Icon, color }) => (
          <div key={label} className="stat-card">
            <div className={clsx("w-10 h-10 rounded-xl flex items-center justify-center border", colorMap[color])}>
              <Icon size={18} />
            </div>
            <div className="font-display text-2xl font-bold text-text-primary mt-1">{value}</div>
            <div className="text-xs text-text-muted">{label}</div>
            <div className="text-xs text-text-secondary">{sub}</div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Grouped bar chart */}
        <div className="card p-6">
          <h3 className="font-display font-semibold text-text-primary mb-4">Performance Metrics Comparison</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={metrics} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
              <XAxis dataKey="Model" tick={{ fill: "#94a3b8", fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} domain={[0.5, 1]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
              <Bar dataKey="Accuracy" fill="#3b82f6" radius={[3, 3, 0, 0]} maxBarSize={14} />
              <Bar dataKey="F1" fill="#10b981" radius={[3, 3, 0, 0]} maxBarSize={14} />
              <Bar dataKey="ROC_AUC" fill="#8b5cf6" radius={[3, 3, 0, 0]} maxBarSize={14} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar chart */}
        <div className="card p-6">
          <h3 className="font-display font-semibold text-text-primary mb-4">Top 3 Models — Radar View</h3>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#1e2d4a" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0.6, 1]} tick={{ fill: "#475569", fontSize: 9 }} />
              <Radar name="XGBoost" dataKey="XGBoost" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} strokeWidth={2} />
              <Radar name="ANN" dataKey="ANN" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.1} strokeWidth={2} />
              <Radar name="Random Forest" dataKey="Random Forest" stroke="#10b981" fill="#10b981" fillOpacity={0.1} strokeWidth={2} />
              <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model leaderboard table */}
      <div className="card overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <h3 className="font-display font-semibold text-text-primary">Model Leaderboard</h3>
          <span className="badge-info">Ranked by ROC-AUC</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {["Rank", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"].map(h => (
                  <th key={h} className="text-left py-3 px-4 text-text-muted font-medium text-xs uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.map((row, i) => {
                const Icon = MODEL_ICONS[row.Model] || Activity;
                const color = MODEL_COLORS[row.Model] || "#3b82f6";
                return (
                  <tr key={row.Model} className="border-b border-border/50 hover:bg-bg-elevated/50 transition-colors">
                    <td className="py-3 px-4">
                      <span className={clsx(
                        "w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold",
                        i === 0 ? "bg-accent-amber/20 text-accent-amber" :
                        i === 1 ? "bg-text-muted/20 text-text-secondary" :
                        "bg-bg-elevated text-text-muted"
                      )}>
                        {i + 1}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: `${color}20`, border: `1px solid ${color}40` }}>
                          <Icon size={13} style={{ color }} />
                        </div>
                        <span className="font-medium text-text-primary">{row.Model}</span>
                      </div>
                    </td>
                    {["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"].map(k => (
                      <td key={k} className="py-3 px-4">
                        <div className="flex items-center gap-1.5">
                          <span className="font-mono text-text-primary">{(+(row[k] || 0)).toFixed(4)}</span>
                          <div className="w-12 h-1 bg-bg-elevated rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{ width: `${(row[k] || 0) * 100}%`, background: color }}
                            />
                          </div>
                        </div>
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
