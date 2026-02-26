import React from "react";

type Props = {
  children: React.ReactNode;
  fallbackLabel?: string;
};

type State = {
  hasError: boolean;
  error: Error | null;
};

/**
 * React error boundary that catches render errors in any child tree
 * and shows a friendly recovery UI instead of a white screen.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      const label = this.props.fallbackLabel || "this section";
      return (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "30px 20px",
            textAlign: "center",
            color: "#666",
            gap: 10,
            minHeight: 120,
          }}
        >
          <div style={{ fontSize: 28, opacity: 0.5 }}>⚠️</div>
          <div style={{ fontWeight: 500, color: "#333" }}>
            Something went wrong in {label}
          </div>
          <div style={{ fontSize: 12, color: "#888", maxWidth: 400 }}>
            {this.state.error?.message || "An unexpected error occurred."}
          </div>
          <button
            className="btn"
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{ marginTop: 8 }}
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
