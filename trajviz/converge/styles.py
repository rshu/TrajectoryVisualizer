"""Centralized CSS constants for the Converge UI."""

CONVERGE_CSS = """
/* ── Converge theme tokens (light) ──────────────────────────── */
:root {
    --cvg-bg: #f6f8fc;
    --cvg-card: #ffffff;
    --cvg-border: #dce3ef;
    --cvg-text: #0f172a;
    --cvg-muted: #5b6473;
    --cvg-accent: #1d4ed8;
    --cvg-success: #059669;
    --cvg-warn: #b45309;
    --cvg-bad: #dc2626;
    --cvg-warning-bg: #fffbeb;
    --cvg-warning-border: #f59e0b;
    --cvg-warning-text: #92400e;
    --cvg-badge-bg: #eef3ff;
    --cvg-badge-text: #1d4ed8;
    --cvg-table-header-bg: #f1f5f9;
    --cvg-table-alt-bg: #f8fafc;
}

/* ── Converge theme tokens (dark) ───────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root {
        --cvg-bg: #0f172a;
        --cvg-card: #1e293b;
        --cvg-border: #334155;
        --cvg-text: #e2e8f0;
        --cvg-muted: #94a3b8;
        --cvg-accent: #60a5fa;
        --cvg-success: #34d399;
        --cvg-warn: #fbbf24;
        --cvg-bad: #f87171;
        --cvg-warning-bg: #422006;
        --cvg-warning-border: #b45309;
        --cvg-warning-text: #fde68a;
        --cvg-badge-bg: #1e3a5f;
        --cvg-badge-text: #93c5fd;
        --cvg-table-header-bg: #1e293b;
        --cvg-table-alt-bg: #0f172a;
    }
}

/* ── Report panel ───────────────────────────────────────────── */
.cvg-report {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI',
        'Noto Sans SC', 'Noto Sans CJK SC', sans-serif;
    background: var(--cvg-card);
    border: 1px solid var(--cvg-border);
    border-radius: 8px;
    padding: 1.5rem;
    color: var(--cvg-text);
    line-height: 1.6;
}

.cvg-report h2 {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 1.25rem 0 0.75rem;
    color: var(--cvg-text);
    border-bottom: 1px solid var(--cvg-border);
    padding-bottom: 0.4rem;
}

.cvg-report h3 {
    font-size: 1rem;
    font-weight: 600;
    margin: 1rem 0 0.5rem;
    color: var(--cvg-text);
}

/* ── Outcome table ──────────────────────────────────────────── */
.cvg-outcome-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0;
    font-size: 0.875rem;
}

.cvg-outcome-table th {
    background: var(--cvg-table-header-bg);
    font-weight: 600;
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 2px solid var(--cvg-border);
}

.cvg-outcome-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--cvg-border);
}

.cvg-outcome-table tr:nth-child(even) td {
    background: var(--cvg-table-alt-bg);
}

/* ── Milestone table ────────────────────────────────────────── */
.cvg-milestone-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0;
    font-size: 0.875rem;
}

.cvg-milestone-table th {
    background: var(--cvg-table-header-bg);
    font-weight: 600;
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 2px solid var(--cvg-border);
}

.cvg-milestone-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--cvg-border);
}

.cvg-milestone-table tr:nth-child(even) td {
    background: var(--cvg-table-alt-bg);
}

/* ── Warning callout ────────────────────────────────────────── */
.cvg-warning {
    background: var(--cvg-warning-bg);
    border: 1px solid var(--cvg-warning-border);
    border-left: 4px solid var(--cvg-warning-border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    color: var(--cvg-warning-text);
    font-size: 0.875rem;
    line-height: 1.5;
}

/* ── Badge ──────────────────────────────────────────────────── */
.cvg-badge {
    display: inline-block;
    background: var(--cvg-badge-bg);
    color: var(--cvg-badge-text);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Anchor analysis section ────────────────────────────────── */
.cvg-anchor-section {
    margin: 1rem 0;
}

.cvg-anchor-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0;
    font-size: 0.875rem;
}

.cvg-anchor-table th {
    background: var(--cvg-table-header-bg);
    font-weight: 600;
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 2px solid var(--cvg-border);
}

.cvg-anchor-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--cvg-border);
}

.cvg-anchor-table tr:nth-child(even) td {
    background: var(--cvg-table-alt-bg);
}

.cvg-anchor-metric {
    font-size: 0.9rem;
    color: var(--cvg-muted);
    margin: 0.5rem 0;
}

/* ── Note callout ───────────────────────────────────────────── */
.cvg-note {
    background: var(--cvg-bg);
    border: 1px solid var(--cvg-border);
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin: 0.35rem 0;
    color: var(--cvg-muted);
    font-size: 0.8rem;
    line-height: 1.4;
}
"""
