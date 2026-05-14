import React, { useState, useEffect } from "react";
import { Image, RefreshCw, Loader2, Download, Search, ZoomIn } from "lucide-react";
import clsx from "clsx";
import { getVisualizations } from "../utils/api.js";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const CATEGORIES = {
  "EDA": ["class_distribution", "phase_distribution", "wordcloud", "top_drugs", "top_diseases", "correlation_heatmap"],
  "Model Eval": ["cm_", "roc_", "nb_posterior"],
  "Comparison": ["all_roc_curves", "metrics_comparison", "radar_chart", "cv_f1_comparison", "overfitting_analysis"],
  "Features": ["feat_imp_", "lr_coefficients", "ann_training_history"],
  "Trees": ["decision_tree_structure", "learning_curve_"],
};

function categorize(filename) {
  for (const [cat, patterns] of Object.entries(CATEGORIES)) {
    if (patterns.some(p => filename.includes(p))) return cat;
  }
  return "Other";
}

function formatName(filename) {
  return filename
    .replace(".png", "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, l => l.toUpperCase());
}

const PLACEHOLDER_CHARTS = [
  { filename: "class_distribution.png", label: "Class Distribution" },
  { filename: "phase_distribution.png", label: "Phase Distribution" },
  { filename: "all_roc_curves.png", label: "All ROC Curves" },
  { filename: "metrics_comparison.png", label: "Metrics Comparison" },
  { filename: "correlation_heatmap.png", label: "Correlation Heatmap" },
  { filename: "wordcloud.png", label: "Criteria Word Cloud" },
  { filename: "radar_chart.png", label: "Radar Chart" },
  { filename: "cv_f1_comparison.png", label: "CV F1 Comparison" },
  { filename: "feat_imp_xgboost.png", label: "XGBoost Feature Importance" },
  { filename: "ann_training_history.png", label: "ANN Training History" },
  { filename: "overfitting_analysis.png", label: "Overfitting Analysis" },
  { filename: "lr_coefficients.png", label: "LR Coefficients" },
];

export default function Visualizations() {
  const [vizList, setVizList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);

  const fetchViz = async () => {
    setLoading(true);
    try {
      const { data } = await getVisualizations();
      if (data?.length) {
        setVizList(data);
        setApiOnline(true);
      } else {
        setVizList([]);
        setApiOnline(false);
      }
    } catch {
      setApiOnline(false);
      setVizList([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchViz(); }, []);

  const displayList = apiOnline
    ? vizList.map(v => ({ ...v, category: categorize(v.filename), label: formatName(v.filename) }))
    : PLACEHOLDER_CHARTS.map(v => ({ ...v, category: categorize(v.filename), url: null }));

  const categories = ["All", ...new Set(displayList.map(v => v.category))];

  const filtered = displayList.filter(v => {
    const matchCat = filter === "All" || v.category === filter;
    const matchSearch = !search || v.label.toLowerCase().includes(search.toLowerCase());
    return matchCat && matchSearch;
  });

  return (
    <div className="space-y-5">
      {/* Toolbar */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="flex gap-1.5 bg-bg-secondary border border-border rounded-xl p-1 flex-wrap">
          {categories.map(cat => (
            <button key={cat} onClick={() => setFilter(cat)}
              className={clsx("px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap",
                filter === cat ? "bg-accent-blue text-white" : "text-text-secondary hover:text-text-primary"
              )}>
              {cat}
            </button>
          ))}
        </div>
        <div className="flex gap-2 flex-1">
          <div className="relative flex-1 max-w-xs">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input value={search} onChange={e => setSearch(e.target.value)}
              className="input-field w-full pl-9 text-sm" placeholder="Search visualizations..." />
          </div>
          <button onClick={fetchViz} disabled={loading}
            className="btn-ghost flex items-center gap-2 text-sm">
            {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            Refresh
          </button>
        </div>
      </div>

      {/* Status */}
      {!apiOnline && (
        <div className="flex items-center gap-2 p-3 bg-accent-amber/10 border border-accent-amber/25 rounded-xl text-xs text-accent-amber">
          <span className="w-1.5 h-1.5 rounded-full bg-accent-amber animate-pulse" />
          Demo mode — run the training pipeline to generate actual visualizations. Showing expected chart inventory.
        </div>
      )}

      {/* Count */}
      <div className="text-xs text-text-muted">
        Showing {filtered.length} of {displayList.length} visualizations
      </div>

      {/* Grid */}
      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="card aspect-video animate-pulse bg-bg-elevated" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="card p-16 text-center">
          <Image size={40} className="mx-auto text-text-muted mb-4" />
          <p className="text-text-secondary">No visualizations found.</p>
          <p className="text-text-muted text-sm mt-1">Run <code className="bg-bg-elevated px-1.5 py-0.5 rounded text-accent-blue">python app.py --mode train</code> to generate charts.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map(viz => (
            <div key={viz.filename}
              className="card group cursor-pointer hover:border-accent-blue/40 transition-all duration-200 overflow-hidden"
              onClick={() => setSelected(viz)}>
              {/* Image area */}
              <div className="aspect-video bg-bg-elevated relative overflow-hidden">
                {viz.url ? (
                  <img
                    src={`${API_BASE}${viz.url}`}
                    alt={viz.label}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    onError={e => { e.target.style.display = "none"; }}
                  />
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center gap-2">
                    <Image size={28} className="text-text-muted" />
                    <span className="text-xs text-text-muted text-center px-3">{viz.label}</span>
                  </div>
                )}
                {/* Hover overlay */}
                <div className="absolute inset-0 bg-accent-blue/0 group-hover:bg-accent-blue/10 transition-all duration-200 flex items-center justify-center opacity-0 group-hover:opacity-100">
                  <div className="w-10 h-10 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center">
                    <ZoomIn size={18} className="text-white" />
                  </div>
                </div>
              </div>
              {/* Meta */}
              <div className="p-3">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-medium text-text-primary leading-tight">{viz.label}</p>
                    <p className="text-xs text-text-muted mt-0.5">{viz.category}</p>
                  </div>
                  {viz.url && (
                    <a
                      href={`${API_BASE}${viz.url}`}
                      download={viz.filename}
                      onClick={e => e.stopPropagation()}
                      className="shrink-0 p-1.5 hover:bg-bg-elevated rounded-lg transition-colors text-text-muted hover:text-text-secondary"
                    >
                      <Download size={13} />
                    </a>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Lightbox */}
      {selected && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6"
          onClick={() => setSelected(null)}>
          <div className="max-w-5xl w-full card overflow-hidden" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-3 border-b border-border">
              <div>
                <h3 className="font-display font-semibold text-text-primary">{selected.label}</h3>
                <p className="text-xs text-text-muted">{selected.category} · {selected.filename}</p>
              </div>
              <div className="flex items-center gap-2">
                {selected.url && (
                  <a href={`${API_BASE}${selected.url}`} download={selected.filename}
                    className="btn-ghost text-xs flex items-center gap-1.5">
                    <Download size={13} /> Download
                  </a>
                )}
                <button onClick={() => setSelected(null)}
                  className="w-8 h-8 rounded-lg hover:bg-bg-elevated flex items-center justify-center text-text-muted hover:text-text-primary transition-colors">
                  ✕
                </button>
              </div>
            </div>
            <div className="p-4 bg-bg-elevated min-h-64 flex items-center justify-center">
              {selected.url ? (
                <img src={`${API_BASE}${selected.url}`} alt={selected.label}
                  className="max-h-[70vh] object-contain rounded-lg" />
              ) : (
                <div className="text-center">
                  <Image size={48} className="mx-auto text-text-muted mb-4" />
                  <p className="text-text-secondary">This chart will appear after running the training pipeline.</p>
                  <code className="text-xs text-accent-blue mt-2 block">python app.py --mode train</code>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
