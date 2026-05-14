import React, { useState, useRef } from "react";
import {
  FlaskConical, Upload, CheckCircle, XCircle, AlertTriangle,
  ChevronDown, Loader2, Sparkles, BarChart2, Info
} from "lucide-react";
import clsx from "clsx";
import { predictSingle, predictBatch } from "../utils/api.js";

const MODELS = ["XGBoost", "Random Forest", "Decision Tree", "Naive Bayes", "ANN", "Logistic Regression"];
const PHASES = ["Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4", "Early Phase 1", "N/A"];

const DEFAULT_FORM = {
  phase: "Phase 3",
  diseases: "Type 2 Diabetes|Hypertension",
  drugs: "Metformin|Linagliptin",
  icdcodes: "E11|I10",
  smiless: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
  criteria: "Inclusion Criteria: Adults aged 18-75 with confirmed diagnosis. Adequate organ function. Exclusion Criteria: Severe renal impairment. Pregnancy or lactation. Prior participation in similar trial.",
  why_stop: "",
  model: "XGBoost",
};

function RiskGauge({ probability }) {
  const angle = probability * 180 - 90;
  const color = probability >= 0.7 ? "#10b981" : probability >= 0.5 ? "#f59e0b" : "#f43f5e";
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-40 h-20 overflow-hidden">
        <svg viewBox="0 0 200 100" className="w-full">
          <path d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke="#1e2d4a" strokeWidth="16" strokeLinecap="round" />
          <path
            d="M10 100 A90 90 0 0 1 190 100"
            fill="none"
            stroke={color}
            strokeWidth="16"
            strokeLinecap="round"
            strokeDasharray={`${probability * 283} 283`}
            style={{ filter: `drop-shadow(0 0 6px ${color})` }}
          />
          <line
            x1="100" y1="100"
            x2={100 + 70 * Math.cos((angle * Math.PI) / 180)}
            y2={100 + 70 * Math.sin((angle * Math.PI) / 180)}
            stroke="white" strokeWidth="3" strokeLinecap="round"
          />
          <circle cx="100" cy="100" r="6" fill={color} />
        </svg>
      </div>
      <div className="font-display text-3xl font-bold mt-1" style={{ color }}>
        {(probability * 100).toFixed(1)}%
      </div>
      <div className="text-xs text-text-muted">Success Probability</div>
    </div>
  );
}

function ShapBar({ feature, value, impact }) {
  const abs = Math.abs(value);
  const max = 0.3;
  const pct = Math.min((abs / max) * 100, 100);
  const color = impact === "positive" ? "#10b981" : "#f43f5e";
  return (
    <div className="flex items-center gap-3 py-1.5">
      <div className="w-36 text-xs text-text-secondary truncate font-mono">{feature}</div>
      <div className="flex-1 h-4 bg-bg-elevated rounded-full overflow-hidden flex items-center">
        {impact === "positive" ? (
          <div className="h-full rounded-full ml-auto" style={{ width: `${pct}%`, background: color, opacity: 0.8 }} />
        ) : (
          <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color, opacity: 0.8 }} />
        )}
      </div>
      <div className="w-16 text-xs font-mono text-right" style={{ color }}>
        {value > 0 ? "+" : ""}{value.toFixed(4)}
      </div>
    </div>
  );
}

export default function Predict() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [batchFile, setBatchFile] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [tab, setTab] = useState("single");
  const fileRef = useRef();

  const handleChange = (e) => setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const { data } = await predictSingle(form);
      setResult(data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Prediction failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  const handleBatch = async () => {
    if (!batchFile) return;
    setBatchLoading(true);
    setBatchResult(null);
    try {
      const { data } = await predictBatch(batchFile, form.model);
      setBatchResult(data);
    } catch (err) {
      setError("Batch prediction failed.");
    } finally {
      setBatchLoading(false);
    }
  };

  const riskColor = (r) => ({
    low: "text-accent-green", moderate: "text-accent-amber",
    high: "text-accent-rose", critical: "text-red-400",
  }[r] || "text-text-secondary");

  const riskBadge = (r) => ({
    low: "badge-success", moderate: "badge-warning",
    high: "badge-danger", critical: "badge-danger",
  }[r] || "badge-info");

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Tab switcher */}
      <div className="flex gap-2 bg-bg-secondary border border-border rounded-xl p-1 w-fit">
        {["single", "batch"].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={clsx(
              "px-5 py-2 rounded-lg text-sm font-medium transition-all",
              tab === t ? "bg-accent-blue text-white shadow" : "text-text-secondary hover:text-text-primary"
            )}
          >
            {t === "single" ? "Single Trial" : "Batch CSV"}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Form */}
        <div className="lg:col-span-3 card p-6">
          <div className="flex items-center gap-2 mb-5">
            <FlaskConical size={18} className="text-accent-blue" />
            <h2 className="font-display font-semibold text-text-primary">
              {tab === "single" ? "Trial Parameters" : "Batch Upload"}
            </h2>
          </div>

          {tab === "single" ? (
            <form onSubmit={handlePredict} className="space-y-4">
              {/* Model selector */}
              <div>
                <label className="label">ML Model</label>
                <div className="relative">
                  <select name="model" value={form.model} onChange={handleChange}
                    className="input-field w-full appearance-none pr-10">
                    {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                  <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
                </div>
              </div>

              {/* Phase */}
              <div>
                <label className="label">Trial Phase</label>
                <div className="relative">
                  <select name="phase" value={form.phase} onChange={handleChange}
                    className="input-field w-full appearance-none pr-10">
                    {PHASES.map(p => <option key={p} value={p}>{p}</option>)}
                  </select>
                  <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
                </div>
              </div>

              {/* Diseases & Drugs */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="label">Diseases <span className="text-text-muted">(pipe-separated)</span></label>
                  <input name="diseases" value={form.diseases} onChange={handleChange}
                    className="input-field w-full" placeholder="Diabetes|Cancer" />
                </div>
                <div>
                  <label className="label">Drugs <span className="text-text-muted">(pipe-separated)</span></label>
                  <input name="drugs" value={form.drugs} onChange={handleChange}
                    className="input-field w-full" placeholder="DrugA|DrugB" />
                </div>
              </div>

              {/* ICD & SMILES */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="label">ICD Codes</label>
                  <input name="icdcodes" value={form.icdcodes} onChange={handleChange}
                    className="input-field w-full" placeholder="E11|I10" />
                </div>
                <div>
                  <label className="label">SMILES String</label>
                  <input name="smiless" value={form.smiless} onChange={handleChange}
                    className="input-field w-full font-mono text-xs" placeholder="CCO|CN..." />
                </div>
              </div>

              {/* Criteria */}
              <div>
                <label className="label">Eligibility Criteria</label>
                <textarea name="criteria" value={form.criteria} onChange={handleChange} rows={4}
                  className="input-field w-full resize-none"
                  placeholder="Inclusion Criteria: ... Exclusion Criteria: ..." />
              </div>

              {/* Why stop */}
              <div>
                <label className="label">Why Stop <span className="text-text-muted">(optional)</span></label>
                <input name="why_stop" value={form.why_stop} onChange={handleChange}
                  className="input-field w-full" placeholder="Leave empty if ongoing" />
              </div>

              {error && (
                <div className="flex items-start gap-2 p-3 bg-accent-rose/10 border border-accent-rose/25 rounded-xl text-sm text-accent-rose">
                  <AlertTriangle size={15} className="mt-0.5 shrink-0" />
                  {error}
                </div>
              )}

              <button type="submit" disabled={loading}
                className="btn-primary w-full flex items-center justify-center gap-2">
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                {loading ? "Predicting..." : "Predict Trial Outcome"}
              </button>
            </form>
          ) : (
            <div className="space-y-4">
              <div
                onClick={() => fileRef.current?.click()}
                className="border-2 border-dashed border-border hover:border-accent-blue rounded-2xl p-10 text-center cursor-pointer transition-colors group"
              >
                <Upload size={32} className="mx-auto text-text-muted group-hover:text-accent-blue mb-3 transition-colors" />
                <p className="text-text-secondary text-sm">
                  {batchFile ? batchFile.name : "Drop CSV file here or click to browse"}
                </p>
                <p className="text-text-muted text-xs mt-1">Must contain: phase, diseases, drugs, criteria</p>
                <input ref={fileRef} type="file" accept=".csv" className="hidden"
                  onChange={e => setBatchFile(e.target.files[0])} />
              </div>
              <div className="flex gap-3">
                <div className="flex-1 relative">
                  <select name="model" value={form.model} onChange={handleChange}
                    className="input-field w-full appearance-none pr-10">
                    {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                  <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
                </div>
                <button onClick={handleBatch} disabled={!batchFile || batchLoading}
                  className="btn-primary flex items-center gap-2 whitespace-nowrap">
                  {batchLoading ? <Loader2 size={15} className="animate-spin" /> : <Upload size={15} />}
                  Run Batch
                </button>
              </div>
              {batchResult && (
                <div className="grid grid-cols-2 gap-3 pt-2">
                  {[
                    { label: "Total Trials", value: batchResult.total_trials },
                    { label: "Predicted Success", value: batchResult.successful_trials, cls: "text-accent-green" },
                    { label: "Predicted Failure", value: batchResult.failed_trials, cls: "text-accent-rose" },
                    { label: "Avg Probability", value: `${(batchResult.avg_success_probability * 100).toFixed(1)}%` },
                  ].map(({ label, value, cls }) => (
                    <div key={label} className="bg-bg-elevated border border-border rounded-xl p-4">
                      <div className={clsx("font-display text-2xl font-bold", cls || "text-text-primary")}>{value}</div>
                      <div className="text-xs text-text-muted">{label}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Results panel */}
        <div className="lg:col-span-2 space-y-4">
          {result ? (
            <>
              {/* Main result card */}
              <div className={clsx(
                "card p-6 border-2",
                result.prediction === 1 ? "border-accent-green/30" : "border-accent-rose/30"
              )}>
                <div className="flex items-center justify-between mb-4">
                  {result.prediction === 1 ? (
                    <div className="flex items-center gap-2 text-accent-green">
                      <CheckCircle size={20} />
                      <span className="font-display font-semibold">Success Likely</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-accent-rose">
                      <XCircle size={20} />
                      <span className="font-display font-semibold">Failure Risk</span>
                    </div>
                  )}
                  <span className={riskBadge(result.risk_level)}>{result.risk_level} risk</span>
                </div>

                <RiskGauge probability={result.success_probability} />

                <div className="mt-4 grid grid-cols-2 gap-3">
                  <div className="bg-bg-elevated rounded-xl p-3">
                    <div className="text-xs text-text-muted">Confidence</div>
                    <div className="font-medium text-text-primary capitalize">{result.confidence}</div>
                  </div>
                  <div className="bg-bg-elevated rounded-xl p-3">
                    <div className="text-xs text-text-muted">Model</div>
                    <div className="font-medium text-text-primary">{result.model_used}</div>
                  </div>
                </div>

                <div className="mt-3 p-3 bg-bg-elevated rounded-xl border border-border">
                  <div className="flex items-start gap-2 text-xs text-text-secondary">
                    <Info size={13} className="mt-0.5 shrink-0" />
                    {result.interpretation}
                  </div>
                </div>
              </div>

              {/* SHAP panel */}
              {result.top_shap_features?.length > 0 && (
                <div className="card p-5">
                  <div className="flex items-center gap-2 mb-4">
                    <BarChart2 size={16} className="text-accent-purple" />
                    <h3 className="font-medium text-text-primary text-sm">SHAP Feature Impact</h3>
                  </div>
                  <div className="space-y-0.5">
                    {result.top_shap_features.map((f, i) => (
                      <ShapBar key={i} {...f} />
                    ))}
                  </div>
                  <div className="mt-3 flex items-center gap-4 text-xs text-text-muted">
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-1 bg-accent-green rounded" />Increases success
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-1 bg-accent-rose rounded" />Decreases success
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card p-10 flex flex-col items-center justify-center text-center h-64">
              <div className="w-14 h-14 bg-accent-blue/10 border border-accent-blue/25 rounded-2xl flex items-center justify-center mb-4">
                <Sparkles size={24} className="text-accent-blue" />
              </div>
              <p className="text-text-secondary text-sm">Fill in the form and click predict to see the trial outcome analysis.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
