"""Centralized CSS constants for the Insight UI."""

from pygments.formatters import HtmlFormatter as _HtmlFormatter
from trajviz.converge.styles import CONVERGE_CSS as _CONVERGE_CSS

_pygments_css = _HtmlFormatter(style="github-dark").get_style_defs(".wf-code-hl")

APP_CSS = """
/* CJK and non-Latin font fallback */
.gradio-container, .gradio-container * {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI',
        'Noto Sans SC', 'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei',
        'Hiragino Sans GB', 'Noto Sans JP', 'Noto Sans KR', sans-serif;
}
:root {
    color-scheme: light;
    --ov-bg: #f6f8fc;
    --ov-card: #ffffff;
    --ov-border: #dce3ef;
    --ov-text: #0f172a;
    --ov-muted: #5b6473;
    --ov-accent: #1d4ed8;
    --ov-success: #059669;
    --ov-warn: #b45309;
    --ov-bad: #dc2626;
    /* Component-level light tokens */
    --ov-card-bg: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    --ov-card-shadow: rgba(15, 23, 42, 0.04);
    --ov-banner-bg: linear-gradient(135deg, #eaf2ff 0%, #effbf4 55%, #fffaf0 100%);
    --ov-banner-border: #c7d7f6;
    --ov-code-bg: #eef3ff;
    --ov-code-border: #dbe5ff;
    --ov-body-text: #1f2937;
    --ov-insight-bg: #f0f4ff;
    --ov-insight-border: #dbe5ff;
    --ov-insight-text: #374151;
    --ov-link-hover-bg: #eef3ff;
    --ov-anomaly-bg: #fef3c7;
    --ov-anomaly-border: #f59e0b;
    --ov-anomaly-text: #92400e;
    --ov-anomaly-hover: #fde68a;
    --ov-chart-ctrl-bg: #f3f6ff;
    --ov-table-header-bg: #f1f5f9;
    --ov-nav-bg: #f8fafc;
    --ov-acc-bg: #ffffff;
    /* Workflow card palette */
    --wf-bg-user: #dbeafe;
    --wf-border-user: #1e40af;
    --wf-bg-assistant: #fef3c7;
    --wf-border-assistant: #92400e;
    --wf-bg-error: #fee2e2;
    --wf-border-error: #dc2626;
    --wf-bg-system-error: #ffedd5;
    --wf-border-system-error: #d97706;
    --wf-bg-final: #d1fae5;
    --wf-border-final: #059669;
    --wf-bg-tool: #fef3c7;
    --wf-border-tool: #d97706;
    --wf-bg-reasoning: #ede9fe;
    --wf-border-reasoning: #7c3aed;
    --wf-bg-default: #f3f4f6;
    --wf-border-default: #6b7280;
    --wf-card-border: #e5e7eb;
    --wf-meta-color: #6b7280;
    --wf-preview-color: #374151;
    --wf-connector-from: #d1d5db;
    --wf-connector-to: #e5e7eb;
    --wf-scroll-thumb: #cbd5e1;
}
/* Type scale */
h2, .section-header { font-size: 18px; font-weight: 700; }
h3, .card-header { font-size: 14px; font-weight: 600; }
body, p, td, li { font-size: 13px; font-weight: 400; }
.muted, .ov-kpi-sub, .wf-meta { font-size: 12px; }

.summary-banner {
    background: var(--ov-banner-bg);
    border: 1px solid var(--ov-banner-border);
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(24, 47, 89, 0.06);
    padding: 10px 16px;
    margin-bottom: 12px;
    font-size: 13px;
    line-height: 1.5;
}
.summary-banner p { margin: 1px 0 !important; }
.overview-card {
    background: var(--ov-card-bg);
    border: 1px solid var(--ov-border);
    border-radius: 12px;
    box-shadow: 0 2px 8px var(--ov-card-shadow);
    padding: 14px 16px;
    min-height: 120px;
    margin-bottom: 20px;
}
.overview-card h3,
.overview-card h4 {
    color: var(--ov-text);
    margin-top: 0.1em;
}
.overview-card p,
.overview-card li,
.overview-card td {
    color: var(--ov-body-text);
}
.overview-card code {
    background: var(--ov-code-bg);
    border: 1px solid var(--ov-code-border);
    border-radius: 5px;
    padding: 1px 5px;
}
.overview-kpi-strip {
    margin: 4px 0 20px 0;
}
.ov-kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
}
.ov-kpi-card {
    background: var(--ov-card);
    border: 1px solid var(--ov-border);
    border-radius: 12px;
    padding: 12px 12px 10px;
    box-shadow: 0 1px 6px rgba(15, 23, 42, 0.04);
    transition: transform 0.15s, box-shadow 0.15s;
}
.ov-kpi-card--issues {
    cursor: pointer;
}
.ov-kpi-card--issues:focus-visible {
    outline: 2px solid var(--ov-accent, #1d4ed8);
    outline-offset: 2px;
}
.ov-kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.10);
}
.ov-kpi-label {
    color: var(--ov-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-weight: 700;
    margin-bottom: 5px;
}
.ov-kpi-value {
    color: var(--ov-text);
    font-size: 23px;
    font-weight: 700;
    line-height: 1.1;
}
.ov-kpi-sub {
    color: var(--ov-muted);
    font-size: 12px;
    margin-top: 4px;
}
.ov-kpi-breakdown {
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    max-height: 7.2em;
    overflow-y: auto;
}
.ov-kpi-breakdown-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    line-height: 1.25;
    min-width: 0;
}
.ov-kpi-breakdown-swatch {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
}
.ov-kpi-breakdown-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 600;
}
.ov-kpi-breakdown-count {
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
    color: var(--ov-text);
    font-weight: 600;
}

.insight-step-link {
    color: var(--ov-accent);
    text-decoration: underline;
    text-decoration-style: dotted;
    cursor: pointer;
    font-weight: 600;
}
.insight-step-link:hover {
    text-decoration-style: solid;
    background: var(--ov-link-hover-bg);
    border-radius: 3px;
    padding: 0 2px;
}
/* Workflow count */
.wf-count {
    font-size: 12px;
    color: var(--ov-muted);
    padding: 4px 0 8px 0;
    font-weight: 600;
}
.per-message-acc {
    border: 1px solid var(--ov-border);
    border-radius: 12px;
    background: var(--ov-acc-bg);
    margin-bottom: 20px;
    transition: background 0.15s, box-shadow 0.15s;
}
.per-message-acc:hover {
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
}
.detail-panel {
    position: sticky; top: 12px;
    max-height: 78vh; overflow-y: auto;
    scrollbar-width: thin;
}
#wf-detail-content {
    text-align: left;
}

/* KPI card verdict indicator */
.ov-kpi-card[data-status="good"] {
    border-left: 4px solid #059669;
}
.ov-kpi-card[data-status="warn"] {
    border-left: 4px solid #d97706;
}
.ov-kpi-card[data-status="bad"] {
    border-left: 4px solid #dc2626;
}
/* ===== Detail-panel (dp-*) ===== */
.dp-header {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 12px;
    font-size: 14px;
    font-weight: 700;
    color: white;
}
.dp-header .dp-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: rgba(255,255,255,0.25);
}
.dp-meta-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-bottom: 14px;
}
.dp-meta-table td {
    padding: 4px 10px;
    border-bottom: 1px solid var(--ov-border);
    color: var(--ov-body-text);
}
.dp-meta-table td:first-child {
    font-weight: 600;
    color: var(--ov-muted);
    white-space: nowrap;
    width: 110px;
}
.dp-meta-table code {
    background: var(--ov-code-bg);
    border: 1px solid var(--ov-code-border);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 11px;
}
.dp-section {
    border-left: 3px solid var(--ov-border);
    background: var(--ov-card);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.dp-section-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 6px 0;
    color: var(--ov-muted);
}
.dp-section-text { border-left-color: var(--wf-border-assistant); }
.dp-section-reasoning { border-left-color: var(--wf-border-reasoning); }
.dp-section-tool { border-left-color: var(--wf-border-tool); }
.dp-section-tool-error { border-left-color: var(--wf-border-error); }
.dp-section-patch { border-left-color: var(--wf-border-final); }
.dp-section-snapshot { border-left-color: var(--wf-border-default); }
.dp-details {
    background: var(--ov-code-bg);
    border: 1px solid var(--ov-code-border);
    border-radius: 8px;
    margin: 6px 0;
    font-size: 12px;
}
.dp-details summary {
    padding: 6px 12px;
    cursor: pointer;
    font-weight: 600;
    font-size: 12px;
    color: var(--ov-accent);
    user-select: none;
}
.dp-details[open] summary {
    border-bottom: 1px solid var(--ov-code-border);
}
.dp-details-body {
    padding: 8px 12px;
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
}
.dp-details-body pre {
    margin: 0;
    font-size: 11px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--ov-body-text);
}
.dp-tool-header {
    font-size: 13px;
    font-weight: 600;
    margin: 0 0 4px 0;
    color: var(--ov-text);
}
.dp-tool-meta {
    font-size: 11px;
    color: var(--ov-muted);
    margin-bottom: 6px;
}
.dp-content {
    font-size: 13px;
    line-height: 1.6;
    color: var(--ov-body-text);
    white-space: pre-wrap;
    word-break: break-word;
}
/* Diff rendering */
.dp-diff-toggle { margin-top: 8px; }
.dp-diff-toggle > summary {
    cursor: pointer; font-weight: 600; font-size: 12px;
    color: var(--ov-accent); padding: 4px 0; user-select: none;
}
.dp-diff-content { margin-top: 4px; }
.dp-diff-file { margin-bottom: 6px; }
.dp-diff-file > summary {
    cursor: pointer; font-size: 12px; font-weight: 600;
    color: var(--ov-body-text); padding: 4px 8px;
    background: var(--ov-code-bg); border-radius: 4px;
}
.dp-diff-pre {
    margin: 4px 0 0; padding: 8px 12px; font-size: 11px;
    line-height: 1.5; background: #0d1117; border-radius: 6px;
    overflow-x: auto; color: #c9d1d9;
}
.dp-diff-pre code { font-family: 'Fira Code','Consolas',monospace; white-space: pre; display: block; }
.diff-add { background: rgba(46,160,67,0.15); color: #3fb950; display: block; }
.diff-del { background: rgba(248,81,73,0.15); color: #f85149; display: block; }
.diff-hunk { color: #79c0ff; font-style: italic; display: block; }
.diff-ctx { color: #8b949e; display: block; }

/* ===== Filter chips ===== */
.filter-panel {
    display: flex; flex-direction: column; gap: 8px;
    position: sticky; top: 0; z-index: 10;
    background: var(--ov-insight-bg);
    border: 1px solid var(--ov-insight-border);
    border-radius: 10px;
    padding: 10px 12px;
}
.filter-group {
    display: flex; align-items: center; gap: 12px;
    min-height: 32px;
}
.filter-group + .filter-group {
    padding-top: 8px;
    border-top: 1px solid var(--ov-border);
}
.filter-group-label {
    flex: 0 0 116px;
    color: var(--ov-text);
    font-size: 12px;
    font-weight: 700;
    line-height: 1.2;
}
.filter-group-label span {
    display: block;
    margin-top: 2px;
    color: var(--ov-muted);
    font-size: 10px;
    font-weight: 400;
}
.filter-options {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.filter-chip {
    display: inline-flex; align-items: center;
    padding: 5px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 600; cursor: pointer;
    font-family: inherit; line-height: 1.4;
    border: 1px solid var(--ov-insight-border);
    background: var(--ov-card); color: var(--ov-muted);
    transition: background 0.15s, border-color 0.15s, color 0.15s;
    user-select: none;
}
.filter-chip:hover {
    border-color: var(--ov-accent);
    background: var(--ov-link-hover-bg);
}
.filter-chip.chip-active {
    background: var(--ov-accent); color: white;
    border-color: var(--ov-accent);
}
.filter-chip-all.chip-active {
    background: #64748b;
    border-color: #64748b;
}
@keyframes filterRequiredPulse {
    0%, 100% { background: transparent; }
    50% { background: rgba(239, 68, 68, 0.10); }
}
.filter-group-attention {
    border-radius: 6px;
    animation: filterRequiredPulse 0.9s ease;
}

/* ===== Agent summary cards ===== */
.agent-cards-grid {
    display: flex; gap: 16px; flex-wrap: wrap;
    padding: 8px 0;
}
.agent-card {
    flex: 1 1 260px; max-width: 400px;
    background: var(--ov-card); border: 1px solid var(--ov-border);
    border-radius: 8px; padding: 14px 16px;
    box-shadow: 0 1px 3px var(--ov-card-shadow);
}
.agent-card-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 6px;
}
.agent-card-label {
    font-weight: 700; font-size: 14px;
}
.agent-card-steps {
    font-size: 12px; color: var(--ov-muted);
}
.agent-card-spawn {
    font-size: 11px; margin-bottom: 8px;
}
.agent-card-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 6px 12px;
}
.agent-kpi-val {
    display: block; font-size: 15px; font-weight: 700;
    color: var(--ov-text);
}
.agent-kpi-label {
    display: block; font-size: 10px; color: var(--ov-muted);
    text-transform: uppercase; letter-spacing: 0.5px;
}

/* ===== TOC sidebar ===== */
.wf-toc-sidebar {
    width: 140px; max-height: 70vh; overflow-y: auto;
    position: sticky; top: 12px;
    background: var(--ov-nav-bg); border: 1px solid var(--ov-border);
    border-radius: 10px; padding: 8px 6px;
    scrollbar-width: thin; scrollbar-color: var(--wf-scroll-thumb) transparent;
}
.wf-toc-sidebar.toc-hidden { display: none; }
.toc-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.7px; color: var(--ov-muted); padding: 4px 6px 8px;
}
.toc-entry {
    display: flex; align-items: center; gap: 4px;
    padding: 4px 6px; border-radius: 6px; cursor: pointer;
    font-size: 11px; transition: background 0.1s;
}
.toc-entry:hover { background: var(--ov-link-hover-bg); }
.toc-num { font-weight: 700; color: var(--ov-text); min-width: 28px; }

/* ===== Upload row compact styling ===== */
.upload-row {
    padding: 0 !important;
    margin-bottom: 8px !important;
    gap: 4px !important;
}
.upload-row > .row {
    align-items: stretch !important;
    gap: 10px !important;
}
.upload-row .file-upload, .upload-row .upload-button {
    min-height: auto !important;
}
/* Format dropdown: align to top, let it size naturally */
.upload-row > .row > .form:first-child {
    align-self: flex-start !important;
    margin-top: 0 !important;
}
/* Smaller drop zone text and icon for compact layout */
.upload-row button[aria-label*="upload"] .wrap {
    font-size: 13px !important;
}
.upload-row button[aria-label*="upload"] .icon-wrap {
    transform: scale(0.75);
}
/* Compact load buttons — don't stretch full column width */
.upload-row button.primary, .upload-row button.secondary {
    max-width: 180px !important;
    align-self: center !important;
    margin-top: 4px !important;
}

/* File-interaction timeline: grow with Plotly's layout.height so every
   file row is visible in Diagnostics. Do not force Plotly graph divs to
   height:auto — that overrides the pixel height and squashes the scatter.
   Gradio's Plotly wrapper also sets autosize + Plots.resize(); a small
   JS expander in insight.build_ui restores the intended height. */
.resizable-chart {
    min-height: 0 !important;
    max-height: none !important;
    height: auto !important;
    overflow: visible !important;
    resize: none;
    border-bottom: none;
    cursor: default;
}
.resizable-chart > .wrap,
.resizable-chart .wrap,
.resizable-chart .plot-container {
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
}

/* ===== Responsive breakpoints ===== */
@media (max-width: 768px) {
    /* Workflow two-column stacks to single column */
    .detail-panel {
        position: relative !important;
        top: auto !important;
        max-height: none !important;
    }
    /* KPI grid: 2 columns on tablet */
    .ov-kpi-grid {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 10px !important;
    }
}
@media (max-width: 480px) {
    .filter-group {
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
    }
    .filter-group-label {
        flex-basis: auto;
    }
    .filter-summary {
        align-items: flex-start;
        gap: 8px;
    }
    /* KPI cards: smaller text */
    .ov-kpi-value { font-size: 18px !important; }
    .ov-kpi-label { font-size: 10px !important; }
    .ov-kpi-sub { font-size: 11px !important; }
    .ov-kpi-card { padding: 8px 10px 8px !important; }
    /* Single column KPI grid */
    .ov-kpi-grid {
        grid-template-columns: repeat(2, 1fr) !important;
    }
    /* Detail panel not sticky */
    .detail-panel {
        position: relative !important;
        top: auto !important;
        max-height: none !important;
    }
    /* Summary banner compact */
    .summary-banner {
        padding: 10px 14px !important;
        border-radius: 10px !important;
    }
    /* Workflow cards compact */
    .wf-container { max-width: 100% !important; }
}

/* ===== Tooltip styles (pure CSS via data-help attribute) ===== */
[data-help] {
    position: relative;
    cursor: help;
    border-bottom: 1px dotted var(--ov-muted);
}
[data-help]::after {
    content: attr(data-help);
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: #1e293b;
    color: #f1f5f9;
    font-size: 11px;
    font-weight: 400;
    line-height: 1.4;
    padding: 6px 10px;
    border-radius: 6px;
    white-space: normal;
    width: max-content;
    max-width: 260px;
    z-index: 100;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    text-transform: none;
    letter-spacing: normal;
}
[data-help]::before {
    content: '';
    position: absolute;
    bottom: calc(100% + 2px);
    left: 50%;
    transform: translateX(-50%);
    border: 4px solid transparent;
    border-top-color: #1e293b;
    z-index: 100;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
}
[data-help]:hover::after,
[data-help]:hover::before {
    opacity: 1;
}

/* ===== Section subtitle ===== */
.section-subtitle {
    font-size: 13px;
    color: var(--ov-muted);
    font-weight: 400;
    margin: -8px 0 12px 0;
    font-style: italic;
}

/* ===== Detail panel tabs ===== */
.dp-tabs {
    position: sticky;
    top: 0;
    z-index: 6;
    display: flex;
    gap: 0;
    background: var(--ov-card);
    border-bottom: 2px solid var(--ov-border);
    margin-bottom: 12px;
}
.dp-tab {
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    color: var(--ov-muted);
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: color 0.15s, border-color 0.15s;
    user-select: none;
}
.dp-tab:hover {
    color: var(--ov-text);
}
.dp-tab.dp-tab-active {
    color: var(--ov-accent);
    border-bottom-color: var(--ov-accent);
}
.dp-tab-content {
    display: none;
}
.dp-tab-content.dp-tab-visible {
    display: block;
}
.dp-metrics-unavailable {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin: 12px 0;
    padding: 16px;
    color: var(--ov-insight-text);
    background: var(--ov-insight-bg);
    border: 1px solid var(--ov-insight-border);
    border-radius: 10px;
}
.dp-metrics-unavailable-icon {
    display: inline-flex;
    flex: 0 0 24px;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    color: white;
    background: #64748b;
    border-radius: 50%;
    font-size: 13px;
    font-weight: 700;
    font-family: Georgia, serif;
}
.dp-metrics-unavailable-title {
    margin-bottom: 4px;
    color: var(--ov-text);
    font-size: 13px;
    font-weight: 700;
}
.dp-metrics-unavailable-description {
    font-size: 12px;
    line-height: 1.5;
}
.dp-metrics-unavailable-fields {
    margin-top: 6px;
    color: var(--ov-muted);
    font-size: 11px;
    line-height: 1.5;
}

/* ===== Breadcrumb ===== */
.dp-breadcrumb {
    font-size: 11px;
    color: var(--ov-muted);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
}
.dp-breadcrumb span {
    cursor: pointer;
    color: var(--ov-accent);
}
.dp-breadcrumb span:hover {
    text-decoration: underline;
}

/* ===== Filter summary bar ===== */
#wf-filter-hidden {
    display: none !important;
}

.filter-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 12px;
    font-size: 12px;
    color: var(--ov-muted);
    background: var(--ov-insight-bg);
    border: 1px solid var(--ov-insight-border);
    border-radius: 8px;
    margin: 4px 0 8px 0;
}
.filter-summary .reset-filters {
    cursor: pointer;
    color: var(--ov-accent);
    font-weight: 600;
    font-size: 11px;
    font-family: inherit;
    border: 0;
    padding: 2px 0;
    background: transparent;
}
.filter-summary .reset-filters:hover {
    text-decoration: underline;
}

/* ===== KPI sparkline ===== */
.ov-kpi-sparkline {
    margin-top: 4px;
    height: 20px;
}
.ov-kpi-sparkline svg {
    display: block;
}

.diag-rc-panel {
    display: flex; flex-direction: column; gap: 6px; margin: 8px 0;
}
.diag-rc-item {
    padding: 8px 12px; border-radius: 8px; font-size: 12px;
    line-height: 1.4; transition: background 0.15s;
}
.diag-rc-primary {
    background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b;
}
.diag-rc-primary:hover { background: #fee2e2; }
.diag-rc-secondary {
    background: var(--ov-insight-bg); border: 1px solid var(--ov-insight-border);
    color: var(--ov-insight-text);
}
.diag-rc-secondary:hover { background: var(--ov-link-hover-bg); }
.diag-rc-rank { font-weight: 700; margin-right: 4px; }
.diag-rc-text { }

/* Context-window usage breakdown (Diagnostics) */
.ctx-usage {
    margin: 4px 0 12px;
    padding: 12px 14px;
    background: var(--ov-card);
    border: 1px solid var(--ov-border);
    border-radius: 10px;
}
.ctx-usage-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.ctx-usage-pct {
    font-size: 18px;
    font-weight: 700;
    color: var(--ov-text);
}
.ctx-usage-counts {
    font-size: 12px;
    color: var(--ov-muted);
}
.ctx-usage-bar {
    display: flex;
    height: 10px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--ov-table-header-bg);
    margin: 4px 0 12px;
}
.ctx-usage-seg { height: 100%; min-width: 0; flex-shrink: 0; }
.ctx-usage-free { background: transparent; }
.ctx-usage-legend {
    display: grid;
    gap: 5px 12px;
    font-size: 12px;
    color: var(--ov-body-text);
    align-items: center;
}
.ctx-usage-legend-window {
    grid-template-columns: minmax(8em, 1fr) auto auto auto;
}
.ctx-usage-legend-loaded {
    grid-template-columns: minmax(8em, 1fr) auto auto;
}
.ctx-usage-head-cell {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--ov-muted);
}
.ctx-usage-name {
    display: flex;
    align-items: center;
    min-width: 0;
}
.ctx-usage-swatch {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    display: inline-block;
    margin-right: 8px;
    flex-shrink: 0;
}
.ctx-usage-tokens,
.ctx-usage-pct-cell {
    font-variant-numeric: tabular-nums;
    text-align: right;
    color: var(--ov-muted);
}

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
    :root {
        /* Core tokens */
        --ov-bg: #1a1b2e;
        --ov-card: #1e1f33;
        --ov-border: #2d2f45;
        --ov-text: #e2e8f0;
        --ov-muted: #94a3b8;
        --ov-accent: #60a5fa;
        --ov-success: #34d399;
        --ov-warn: #fbbf24;
        --ov-bad: #f87171;
        /* Component-level dark tokens */
        --ov-card-bg: linear-gradient(180deg, #1e1f33 0%, #1a1b2e 100%);
        --ov-card-shadow: rgba(0, 0, 0, 0.3);
        --ov-banner-bg: linear-gradient(135deg, #1e2a4a 0%, #1a2e2a 55%, #2a2418 100%);
        --ov-banner-border: #2d3a5c;
        --ov-code-bg: #161726;
        --ov-code-border: #2d2f45;
        --ov-body-text: #cbd5e1;
        --ov-insight-bg: #1a2040;
        --ov-insight-border: #2d3a5c;
        --ov-insight-text: #94a3b8;
        --ov-link-hover-bg: #1e2a4a;
        --ov-anomaly-bg: rgba(245,158,11,0.12);
        --ov-anomaly-border: rgba(245,158,11,0.3);
        --ov-anomaly-text: #fbbf24;
        --ov-anomaly-hover: rgba(245,158,11,0.2);
        --ov-chart-ctrl-bg: #1a2040;
        --ov-table-header-bg: #1e2040;
        --ov-nav-bg: #161726;
        --ov-acc-bg: #1e1f33;
        /* Workflow card palette (dark) */
        --wf-bg-user: rgba(30,64,175,0.15);
        --wf-border-user: #60a5fa;
        --wf-bg-assistant: rgba(245,158,11,0.1);
        --wf-border-assistant: #fbbf24;
        --wf-bg-error: rgba(220,38,38,0.12);
        --wf-border-error: #f87171;
        --wf-bg-system-error: rgba(217,119,6,0.12);
        --wf-border-system-error: #fbbf24;
        --wf-bg-final: rgba(5,150,105,0.12);
        --wf-border-final: #34d399;
        --wf-bg-tool: rgba(217,119,6,0.1);
        --wf-border-tool: #fbbf24;
        --wf-bg-reasoning: rgba(124,58,237,0.12);
        --wf-border-reasoning: #a78bfa;
        --wf-bg-default: rgba(107,114,128,0.1);
        --wf-border-default: #6b7280;
        --wf-card-border: #2d2f45;
        --wf-meta-color: #94a3b8;
        --wf-preview-color: #cbd5e1;
        --wf-connector-from: #374151;
        --wf-connector-to: #2d2f45;
        --wf-scroll-thumb: #4b5563;
    }
    .diag-rc-primary {
        background: rgba(220,38,38,0.1); border-color: rgba(220,38,38,0.3);
        color: #fca5a5;
    }
    .diag-rc-primary:hover { background: rgba(220,38,38,0.15); }
    /* Detail panel header stays light text on colored bg — no override needed */
    .dp-diff-pre { background: #0d1117; }
    .filter-chip { background: var(--ov-card); color: var(--ov-muted); }
    .filter-chip:hover { background: var(--ov-link-hover-bg); }
    .filter-chip.chip-active { background: var(--ov-accent); color: #0f172a; }
}

/* ===== Trajectory Quality Score ===== */
.score-dim-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;
}
.score-dim-card {
    background: var(--ov-card); border: 1px solid var(--ov-border);
    border-radius: 8px; padding: 10px 14px; transition: box-shadow 0.15s;
}
.score-dim-card:hover { box-shadow: 0 2px 8px var(--ov-card-shadow); }
.score-dim-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 4px;
}
.score-dim-name { font-size: 12px; font-weight: 600; color: var(--ov-text); }
.score-dim-badge {
    display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 7px;
    border-radius: 4px; color: white; text-transform: uppercase; letter-spacing: 0.5px;
}
.score-dim-score { font-size: 24px; font-weight: 800; line-height: 1.2; }
.score-dim-metric { font-size: 11px; color: var(--ov-muted); margin-top: 2px; }

.judge-panel {
    margin: 8px 0; border: 1px solid var(--ov-border); border-radius: 8px;
    overflow: hidden;
}
.judge-panel summary {
    padding: 8px 12px; cursor: pointer; font-size: 13px; font-weight: 600;
    color: var(--ov-text); display: flex; align-items: center; gap: 8px;
}

/* Overview Issues fold */
.overview-issues-panel {
    margin: 4px 0 10px;
    border: 1px solid var(--ov-border);
    border-radius: 6px;
    background: var(--ov-card);
    overflow: hidden;
}
.overview-issues-summary {
    list-style: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    padding: 8px 12px;
    font-size: 13px;
    font-weight: 600;
    color: var(--ov-text);
    user-select: none;
}
.overview-issues-summary::-webkit-details-marker { display: none; }
.overview-issues-summary::before {
    content: "▸";
    display: inline-block;
    width: 0.9em;
    color: var(--ov-muted);
    transition: transform 0.12s ease;
}
.overview-issues-panel[open] > .overview-issues-summary::before {
    transform: rotate(90deg);
}
.overview-issues-summary-meta {
    font-size: 12px;
    font-weight: 500;
    color: var(--ov-muted);
}
.overview-issues-body {
    padding: 0 12px 10px;
    border-top: 1px solid var(--ov-border);
}
.overview-issue-card {
    padding: 8px 10px;
    background: var(--ov-card);
    border-left: 3px solid var(--ov-border);
    border-radius: 4px;
    margin-bottom: 6px;
}
.overview-issue-head {
    display: flex;
    align-items: baseline;
    gap: 8px;
    flex-wrap: wrap;
}
.overview-issue-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--ov-text);
    line-height: 1.35;
}
.overview-issue-detail {
    font-size: 12px;
    color: var(--ov-muted);
}
.overview-issue-more {
    margin-top: 4px;
}
.overview-issue-more-summary {
    list-style: none;
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
    color: var(--ov-accent, #1d4ed8);
    user-select: none;
}
.overview-issue-more-summary::-webkit-details-marker { display: none; }
.overview-issue-more-summary::before {
    content: "▸ ";
    color: var(--ov-muted);
}
.overview-issue-more[open] > .overview-issue-more-summary::before {
    content: "▾ ";
}
.overview-issue-more-body {
    margin-top: 4px;
    padding-top: 2px;
}
.overview-issues-progress {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 10px 0 8px;
    padding: 8px 10px;
    border-radius: 4px;
    border: 1px solid var(--ov-banner-border, #c7d7f6);
    background: var(--ov-insight-bg, #f0f4ff);
    color: var(--ov-accent, #1d4ed8);
    font-size: 13px;
    font-weight: 600;
    line-height: 1.35;
}
.overview-issues-progress-label {
    flex-shrink: 0;
}
.judge-badge {
    display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 7px;
    border-radius: 4px; color: white; text-transform: uppercase;
}
.judge-reasoning {
    padding: 8px 12px; font-size: 12px; color: var(--ov-muted); line-height: 1.5;
    border-top: 1px solid var(--ov-border);
}

/* DECAF attribution tab */
.attr-banner {
    padding: 12px 14px; margin-bottom: 12px; border-radius: 8px;
    background: var(--ov-card); border: 1px solid var(--ov-border);
}
.attr-banner-head { font-size: 15px; font-weight: 600; color: var(--ov-text); }
.attr-banner-sub { font-size: 12px; color: var(--ov-muted); margin-top: 4px; }
.attr-link { display: flex; gap: 8px; padding: 4px 0; align-items: flex-start; }
.attr-link-icon { color: var(--ov-muted); font-weight: 700; min-width: 14px; }
.attr-link-body { flex: 1; color: var(--ov-text); }
.attr-link-src {
    display: block; font-size: 11px; color: var(--ov-muted);
    font-family: ui-monospace, monospace; margin-top: 2px;
}
.attr-quote {
    font-size: 11px; color: var(--ov-muted); border-left: 2px solid var(--ov-border);
    padding: 2px 8px; margin: 4px 0 4px 6px; font-style: italic;
}
.attr-audit {
    font-size: 11px; color: var(--ov-muted); margin-top: 6px;
    padding-top: 6px; border-top: 1px dashed var(--ov-border);
}

/* ===== Overview section navigation ===== */
.overview-content-layout {
    align-items: start;
    gap: 18px;
}
.overview-section-nav {
    position: sticky;
    top: 12px;
    flex: 0 0 160px !important;
    min-width: 160px !important;
    padding: 8px;
    background: var(--ov-nav-bg);
    border: 1px solid var(--ov-border);
    border-radius: 10px;
}
.overview-nav-title {
    padding: 6px 10px 8px;
    color: var(--ov-text);
    font-size: 13px;
    font-weight: 700;
    border-bottom: 1px solid var(--ov-border);
    margin-bottom: 4px;
}
.overview-section-radio {
    min-width: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    overflow: visible !important;
}
.overview-section-radio > .wrap {
    flex-direction: column;
    align-items: stretch;
    gap: 4px;
}
.overview-section-radio label {
    width: 100%;
    min-height: 32px;
    margin: 0;
    padding: 7px 10px;
    justify-content: flex-start !important;
    color: var(--ov-muted);
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
}
.overview-section-radio label:hover {
    background: var(--ov-link-hover-bg);
    color: var(--ov-accent);
}
.overview-section-radio label.selected {
    background: var(--ov-link-hover-bg) !important;
    color: var(--ov-accent) !important;
    font-weight: 600;
}
.overview-section-radio input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
}
.overview-section-radio input:focus-visible + span {
    outline: 2px solid var(--ov-accent);
    outline-offset: 3px;
    border-radius: 6px;
}
.overview-section-content {
    min-width: 0;
}
@media (max-width: 768px) {
    .overview-content-layout {
        display: block !important;
    }
    .overview-section-nav {
        position: static;
        width: 100%;
        margin-bottom: 12px;
    }
    .overview-section-radio > .wrap {
        flex-direction: row;
        flex-wrap: nowrap;
        overflow-x: auto;
    }
    .overview-section-radio label {
        width: auto;
        flex: 0 0 auto;
    }
}

/* ===== Responsive KPI grid ===== */
@media (max-width: 600px) {
    .ov-kpi-grid {
        grid-template-columns: 1fr !important;
    }
    .filter-summary {
        align-items: flex-start;
        gap: 8px;
    }
}

/* ===== Run-group file coverage matrix ===== */
.rg-file-scroll {
    overflow-x: auto;
    max-width: 100%;
    margin: 0.25rem 0 0.75rem;
}
.rg-matrix-toggle {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}
.rg-matrix-toggle:not(:checked) ~ .rg-file-scroll tr.rg-matrix-tail {
    display: none;
}
.rg-matrix-toggle-label {
    display: inline-block;
    margin: -0.35rem 0 0.85rem;
    font-size: 12px;
    color: var(--ov-accent);
    cursor: pointer;
    user-select: none;
}
.rg-matrix-toggle-label:hover {
    text-decoration: underline;
}
.rg-matrix-toggle ~ .rg-matrix-toggle-label .rg-matrix-hide {
    display: none;
}
.rg-matrix-toggle:checked ~ .rg-matrix-toggle-label .rg-matrix-show {
    display: none;
}
.rg-matrix-toggle:checked ~ .rg-matrix-toggle-label .rg-matrix-hide {
    display: inline;
}
.rg-file-table .rg-file-path code {
    font-size: 12px;
    word-break: break-all;
}
.rg-file-summary {
    font-size: 13px;
    color: var(--ov-text);
    margin: 0 0 8px;
}
.rg-dir-caret {
    display: inline-block;
    width: 0.65em;
    color: var(--ov-muted);
    font-size: 11px;
    flex-shrink: 0;
}
.rg-dir-caret::before {
    content: "▸";
}
.rg-tree-dir[open] > summary .rg-dir-caret::before {
    content: "▾";
}
.rg-dir-count {
    font-size: 11px;
    color: var(--ov-muted);
    white-space: nowrap;
}
.rg-mix-bar {
    display: inline-flex;
    width: 72px;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--ov-border);
    flex-shrink: 0;
}
.rg-mix-seg {
    display: block;
    height: 100%;
}
.rg-mix-shared { background: #059669; }
.rg-mix-partial { background: #6366f1; }
.rg-mix-run { background: var(--rg-run, #64748b); }
.rg-run-0 { --rg-run: #2563eb; --rg-run-bg: #dbeafe; --rg-run-fg: #1e40af; }
.rg-run-1 { --rg-run: #7c3aed; --rg-run-bg: #ede9fe; --rg-run-fg: #5b21b6; }
.rg-run-2 { --rg-run: #db2777; --rg-run-bg: #fce7f3; --rg-run-fg: #9d174d; }
.rg-run-3 { --rg-run: #0e7490; --rg-run-bg: #cffafe; --rg-run-fg: #155e75; }
.rg-run-4 { --rg-run: #c2410c; --rg-run-bg: #ffedd5; --rg-run-fg: #9a3412; }
.rg-run-5 { --rg-run: #4338ca; --rg-run-bg: #e0e7ff; --rg-run-fg: #3730a3; }
.rg-run-swatch {
    display: inline-block;
    width: 0.65em;
    height: 0.65em;
    border-radius: 999px;
    background: var(--rg-run, #64748b);
    vertical-align: -1px;
    flex-shrink: 0;
}
.rg-legend-run,
.rg-tree-runhead {
    color: var(--rg-run-fg, var(--ov-text));
    font-weight: 600;
    white-space: nowrap;
}
.rg-legend-run {
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.rg-file-root {
    font-size: 12px;
    color: var(--ov-muted);
    margin: 0 0 6px;
}
.rg-file-root code {
    font-size: 12px;
    color: var(--ov-text);
}
.rg-both-list {
    margin: 0 0 10px;
    font-size: 12px;
}
.rg-both-list summary {
    cursor: pointer;
    color: var(--ov-text);
    font-weight: 500;
}
.rg-both-paths {
    margin: 6px 0 0;
    padding-left: 1.2em;
    max-height: 12em;
    overflow: auto;
}
.rg-both-paths code {
    font-size: 12px;
}
.rg-both-more {
    color: var(--ov-muted);
    list-style: none;
    margin-left: -0.4em;
}
.rg-both-read {
    display: inline-block;
    margin-left: 6px;
    padding: 0 6px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    background: #d1fae5;
    color: #047857;
    vertical-align: middle;
}
.rg-tree {
    margin: 0.25rem 0 0.75rem;
    overflow-x: auto;
}
.rg-tree-head,
.rg-tree-summary,
.rg-tree-file {
    display: grid;
    grid-template-columns: minmax(12em, 1fr) repeat(var(--rg-run-cols, 2), 5.6em) minmax(7em, auto);
    gap: 6px 8px;
    align-items: center;
}
.rg-tree-head {
    font-size: 11px;
    color: var(--ov-muted);
    padding: 0 0 6px;
    border-bottom: 1px solid var(--ov-border);
    margin-bottom: 4px;
}
.rg-tree-cell {
    text-align: center;
    white-space: nowrap;
}
.rg-tree-runhead {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
}
.rg-tree-name {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
}
.rg-tree-name code {
    font-size: 12px;
    word-break: break-all;
}
.rg-tree-mix {
    font-size: 12px;
    color: var(--ov-muted);
    white-space: nowrap;
}
.rg-tree-dir {
    margin: 2px 0 2px 0.15em;
}
.rg-tree-dir > .rg-tree-dir,
.rg-tree-dir > .rg-tree-file {
    margin-left: 1em;
}
.rg-tree-summary {
    cursor: pointer;
    list-style: none;
}
.rg-tree-summary::-webkit-details-marker {
    display: none;
}
.rg-tree-file {
    padding: 2px 0 2px 6px;
    border-radius: 4px;
}
.rg-tree-both {
    background: rgba(5, 150, 105, 0.12);
    box-shadow: inset 3px 0 0 #059669;
}
.rg-tree-unique {
    background: var(--rg-run-bg, #f1f5f9);
    box-shadow: inset 3px 0 0 var(--rg-run, #64748b);
}
.rg-badge {
    display: inline-block;
    min-width: 1.35em;
    padding: 1px 5px;
    margin: 0 1px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    line-height: 1.4;
    text-align: center;
}
.rg-badge-r {
    background: #dbeafe;
    color: #1d4ed8;
}
.rg-badge-shared {
    background: #d1fae5;
    color: #047857;
}
.rg-badge-run {
    background: var(--rg-run-bg, #e2e8f0);
    color: var(--rg-run-fg, #334155);
}
.rg-badge-w {
    background: #ffedd5;
    color: #c2410c;
}
.rg-file-empty {
    color: var(--ov-muted);
    font-size: 12px;
}
.rg-kind {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 6px;
}
.rg-kind-shared {
    background: #d1fae5;
    color: #047857;
}
.rg-kind-partial {
    background: #e0e7ff;
    color: #4338ca;
}
.rg-kind-unique {
    background: #fef3c7;
    color: #b45309;
}
.rg-kind-run {
    background: var(--rg-run-bg, #f1f5f9);
    color: var(--rg-run-fg, #334155);
}
.rg-cov {
    font-size: 11px;
    color: var(--ov-muted);
}
.rg-row-unique td {
    background: rgba(245, 158, 11, 0.06);
}
.rg-row-consensus td {
    background: rgba(5, 150, 105, 0.04);
}
.rg-action-hit {
    color: var(--ov-success);
    font-weight: 700;
    font-size: 12px;
}
.rg-atype {
    display: inline-block;
    min-width: 3.2em;
    padding: 1px 5px;
    margin-right: 4px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-align: center;
    vertical-align: middle;
}
.rg-atype-read { background: #dbeafe; color: #1d4ed8; }
.rg-atype-write { background: #ffedd5; color: #c2410c; }
.rg-atype-search { background: #ede9fe; color: #6d28d9; }
.rg-atype-cmd { background: #e2e8f0; color: #334155; }
.rg-atype-spawn { background: #fce7f3; color: #be185d; }
.rg-atype-other { background: #f1f5f9; color: #475569; }
@media (prefers-color-scheme: dark) {
    .rg-badge-r { background: rgba(59,130,246,0.25); color: #93c5fd; }
    .rg-badge-w { background: rgba(234,88,12,0.25); color: #fdba74; }
    .rg-badge-shared { background: rgba(5,150,105,0.25); color: #6ee7b7; }
    .rg-kind-shared { background: rgba(5,150,105,0.25); color: #6ee7b7; }
    .rg-kind-partial { background: rgba(99,102,241,0.25); color: #a5b4fc; }
    .rg-kind-unique { background: rgba(245,158,11,0.25); color: #fcd34d; }
    .rg-mix-shared { background: #34d399; }
    .rg-mix-partial { background: #818cf8; }
    .rg-both-read { background: rgba(5,150,105,0.25); color: #6ee7b7; }
    .rg-tree-both { background: rgba(5,150,105,0.18); box-shadow: inset 3px 0 0 #34d399; }
    .rg-run-0 { --rg-run: #60a5fa; --rg-run-bg: rgba(37,99,235,0.28); --rg-run-fg: #bfdbfe; }
    .rg-run-1 { --rg-run: #c4b5fd; --rg-run-bg: rgba(124,58,237,0.28); --rg-run-fg: #ddd6fe; }
    .rg-run-2 { --rg-run: #f9a8d4; --rg-run-bg: rgba(219,39,119,0.28); --rg-run-fg: #fbcfe8; }
    .rg-run-3 { --rg-run: #67e8f9; --rg-run-bg: rgba(14,116,144,0.28); --rg-run-fg: #a5f3fc; }
    .rg-run-4 { --rg-run: #fdba74; --rg-run-bg: rgba(194,65,12,0.28); --rg-run-fg: #fed7aa; }
    .rg-run-5 { --rg-run: #a5b4fc; --rg-run-bg: rgba(67,56,202,0.28); --rg-run-fg: #c7d2fe; }
    .rg-atype-read { background: rgba(59,130,246,0.25); color: #93c5fd; }
    .rg-atype-write { background: rgba(234,88,12,0.25); color: #fdba74; }
    .rg-atype-search { background: rgba(139,92,246,0.25); color: #c4b5fd; }
    .rg-atype-cmd { background: rgba(148,163,184,0.25); color: #cbd5e1; }
    .rg-atype-spawn { background: rgba(236,72,153,0.25); color: #f9a8d4; }
}

/* ===== Converge comparison styles (merged) ===== */
""" + _CONVERGE_CSS + """

/* AI Trajectory Analysis sidebar */
#analysis-sidebar .analysis-panel-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--ov-text);
    letter-spacing: -0.01em;
    text-align: center;
    margin: 0 0 10px;
}
#analysis-sidebar .html-container:has(.analysis-panel-title) .prose {
    text-align: center;
}
#analysis-sidebar .analysis-status {
    font-size: 12px;
    color: var(--ov-muted);
    line-height: 1.45;
    margin-bottom: 10px;
    text-align: left;
}
#analysis-sidebar .analysis-status-ok { color: var(--ov-success); font-weight: 600; }
#analysis-sidebar .analysis-status-warn { color: var(--ov-warn); font-weight: 600; }
#analysis-sidebar .analysis-status code {
    background: var(--ov-code-bg);
    border: 1px solid var(--ov-code-border);
    border-radius: 4px;
    padding: 0 4px;
    font-size: 11px;
}
#analysis-sidebar .analysis-chatbot {
    border: 1px solid var(--ov-border);
    border-radius: 10px;
}
/* Closed right-edge tab: Gradio rotates this button 180deg, so undo that
   before adding a readable label next to the chevron. */
#analysis-sidebar.right:not(.open) .toggle-button {
    transform: none;
    width: auto;
    height: auto;
    padding: 7px 12px 7px 8px;
    gap: 8px;
    border-radius: 8px 0 0 8px;
    border: 1px solid var(--ov-border, var(--border-color-primary));
    border-right: none;
    box-shadow: -2px 2px 8px var(--ov-card-shadow, rgba(15, 23, 42, 0.08));
    background: var(--ov-card, var(--background-fill-secondary));
}
#analysis-sidebar.right:not(.open) .toggle-button .chevron {
    transform: rotate(180deg);
    padding: 0;
}
#analysis-sidebar.right:not(.open) .toggle-button::after {
    content: "AI Trajectory Analysis";
    font-size: 12px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--ov-text, var(--body-text-color));
    white-space: nowrap;
    line-height: 1.2;
}
"""

WORKFLOW_CSS = """
<style>
.wf-scroll {
    max-height: 75vh; overflow-y: auto; padding: 8px 4px 8px 0;
    scrollbar-width: thin; scrollbar-color: var(--wf-scroll-thumb) transparent;
}
.wf-scroll::-webkit-scrollbar { width: 6px; }
.wf-scroll::-webkit-scrollbar-thumb { background: var(--wf-scroll-thumb); border-radius: 3px; }
.wf-container { font-family: system-ui, -apple-system, 'Noto Sans SC', 'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei', 'Hiragino Sans GB', sans-serif; max-width: 680px; margin: 0 auto; }
.wf-card {
    border: 2px solid var(--wf-card-border); border-radius: 10px; padding: 12px 16px;
    cursor: pointer; transition: box-shadow 0.15s, transform 0.1s;
    position: relative;
}
.wf-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.10); transform: translateY(-1px); }
.wf-card.wf-active { border-left: 4px solid var(--ov-accent); box-shadow: 0 2px 10px rgba(29,78,216,0.15); }
/* Temporary pulse when jumping here from Overview charts / diagnostics. */
.wf-card.wf-flash {
    animation: wf-flash-pulse 2s ease-out 1;
    z-index: 2;
    border-color: var(--ov-accent) !important;
    border-width: 3px;
    border-left-width: 6px;
    background-color: color-mix(in srgb, var(--ov-accent) 22%, transparent);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--ov-accent) 45%, transparent),
                0 8px 28px rgba(29, 78, 216, 0.4);
}
@keyframes wf-flash-pulse {
    0% {
        transform: scale(1.02);
        background-color: color-mix(in srgb, var(--ov-accent) 38%, transparent);
        box-shadow: 0 0 0 8px color-mix(in srgb, var(--ov-accent) 50%, transparent),
                    0 10px 32px rgba(29, 78, 216, 0.5);
    }
    18% {
        transform: scale(1);
        background-color: color-mix(in srgb, var(--ov-accent) 28%, transparent);
        box-shadow: 0 0 0 5px color-mix(in srgb, var(--ov-accent) 42%, transparent),
                    0 8px 28px rgba(29, 78, 216, 0.42);
    }
    55% {
        background-color: color-mix(in srgb, var(--ov-accent) 22%, transparent);
        box-shadow: 0 0 0 4px color-mix(in srgb, var(--ov-accent) 35%, transparent),
                    0 6px 22px rgba(29, 78, 216, 0.35);
    }
    100% {
        transform: scale(1);
        background-color: color-mix(in srgb, var(--ov-accent) 0%, transparent);
        box-shadow: 0 0 0 0 color-mix(in srgb, var(--ov-accent) 0%, transparent),
                    0 2px 10px rgba(29, 78, 216, 0.15);
    }
}
.wf-connector {
    width: 2px; height: 20px; background: linear-gradient(to bottom, var(--wf-connector-from), var(--wf-connector-to));
    margin: 0 auto;
}
.wf-badge {
    display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px;
    border-radius: 6px; margin-right: 6px; text-transform: uppercase; letter-spacing: 0.5px;
}
.wf-header { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; }
.wf-meta { font-size: 12px; color: var(--wf-meta-color); margin-top: 5px; display: flex; gap: 8px; flex-wrap: wrap; }
.wf-meta span { white-space: nowrap; }
.wf-preview {
    font-size: 12px; color: var(--wf-preview-color); margin-top: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-height: 1.4em; line-height: 1.4;
}

.wf-code-block {
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    margin: 6px 0; overflow-x: auto; position: relative;
}
.wf-code-lang {
    position: absolute; top: 4px; right: 8px;
    font-size: 10px; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.5px; user-select: none;
}
.wf-code-block pre {
    margin: 0; padding: 10px 14px; overflow-x: auto;
}
.wf-code-block code {
    font-family: 'Fira Code', 'Cascadia Code', 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px; line-height: 1.5; color: #c9d1d9;
    white-space: pre; display: block;
}
/* Pygments syntax highlighting (github-dark) */
""" + _pygments_css + """
.wf-icons { font-size: 11px; color: var(--wf-meta-color); }
</style>
<!-- Click handling via Gradio js_on_load + trigger() -->
"""
