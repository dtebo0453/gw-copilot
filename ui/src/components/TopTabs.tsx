import React from "react";

export type TopTab = "Map" | "3D" | "Artifacts" | "Model Files" | "Plots";

export function TopTabs({
  tab,
  setTab,
  children,
}: {
  tab: TopTab;
  setTab: (t: TopTab) => void;
  children: React.ReactNode;
}) {
  return (
    <div className="topPanel">
      <div className="tabs">
        {(["Map", "3D", "Artifacts", "Model Files", "Plots"] as TopTab[]).map((t) => (
          <button
            key={t}
            className={t === tab ? "tab active" : "tab"}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="topContent">{children}</div>
    </div>
  );
}
