import React, { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard, Microscope, BarChart3, Image,
  Activity, ChevronLeft, ChevronRight, Dna, Bell
} from "lucide-react";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/dashboard", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/predict", icon: Microscope, label: "Predict Trial" },
  { to: "/compare", icon: BarChart3, label: "Model Comparison" },
  { to: "/visualizations", icon: Image, label: "Visualizations" },
];

export default function Layout({ children }) {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  const currentPage = NAV_ITEMS.find(n => n.to === location.pathname)?.label || "Dashboard";

  return (
    <div className="flex h-screen overflow-hidden bg-bg-primary bg-grid">
      {/* Sidebar */}
      <aside
        className={clsx(
          "flex flex-col bg-bg-secondary border-r border-border transition-all duration-300 ease-in-out z-20 shrink-0",
          collapsed ? "w-16" : "w-60"
        )}
      >
        {/* Logo */}
        <div className={clsx(
          "flex items-center gap-3 px-4 h-16 border-b border-border shrink-0",
          collapsed && "justify-center px-0"
        )}>
          <div className="w-8 h-8 bg-gradient-to-br from-accent-blue to-accent-cyan rounded-lg flex items-center justify-center shrink-0">
            <Dna size={16} className="text-white" />
          </div>
          {!collapsed && (
            <div>
              <div className="font-display font-bold text-sm text-text-primary leading-tight">ClinicalAI</div>
              <div className="text-[10px] text-text-muted">Trial Predictor</div>
            </div>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 px-2 space-y-1">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150",
                collapsed && "justify-center px-2",
                isActive
                  ? "bg-accent-blue/15 text-accent-blue border border-accent-blue/25"
                  : "text-text-secondary hover:text-text-primary hover:bg-bg-elevated"
              )}
              title={collapsed ? label : undefined}
            >
              <Icon size={18} className="shrink-0" />
              {!collapsed && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Status indicator */}
        {!collapsed && (
          <div className="mx-3 mb-4 p-3 bg-bg-elevated rounded-xl border border-border">
            <div className="flex items-center gap-2 mb-1">
              <span className="glow-dot bg-accent-green" />
              <span className="text-xs text-accent-green font-medium">API Online</span>
            </div>
            <div className="text-[11px] text-text-muted">6 models loaded</div>
          </div>
        )}

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(c => !c)}
          className="flex items-center justify-center h-10 w-full border-t border-border text-text-muted hover:text-text-secondary transition-colors"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </aside>

      {/* Main content */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        {/* Topbar */}
        <header className="h-16 bg-bg-secondary border-b border-border flex items-center justify-between px-6 shrink-0">
          <div>
            <h1 className="font-display font-semibold text-text-primary">{currentPage}</h1>
            <p className="text-xs text-text-muted">Predictive Modeling for Clinical Trial Success</p>
          </div>
          <div className="flex items-center gap-3">
            <button className="relative p-2 text-text-muted hover:text-text-primary hover:bg-bg-elevated rounded-lg transition-colors">
              <Bell size={18} />
              <span className="absolute top-1 right-1 w-2 h-2 bg-accent-rose rounded-full" />
            </button>
            <div className="w-8 h-8 bg-gradient-to-br from-accent-purple to-accent-blue rounded-full flex items-center justify-center text-white text-xs font-bold">
              ML
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
