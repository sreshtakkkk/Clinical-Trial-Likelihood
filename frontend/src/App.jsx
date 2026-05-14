import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Predict from "./pages/Predict.jsx";
import Compare from "./pages/Compare.jsx";
import Visualizations from "./pages/Visualizations.jsx";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/visualizations" element={<Visualizations />} />
      </Routes>
    </Layout>
  );
}
