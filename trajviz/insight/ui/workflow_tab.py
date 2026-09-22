"""Workflow tab: filter chips, TOC, step cards, and client JS."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..presenters.workflow import (
    DETAIL_PLACEHOLDER,
    FILTER_CHIPS_DEFAULT,
    build_filtered_workflow_outputs,
    build_workflow_outputs,
)
from ..rendering import render_filter_chips
from ..session import LoadedSession
from .shared import SharedState

WORKFLOW_JS = """
                            /* Filter chip click handler (delegated, survives re-renders) */
                            window.__syncWorkflowFilters = function(bar) {
                                if (!bar) return;
                                var active = Array.from(bar.querySelectorAll('.filter-chip.chip-active'))
                                    .map(function(c) { return c.dataset.filter; });
                                var hiddenEl = document.querySelector(
                                    '#wf-filter-hidden textarea, #wf-filter-hidden input'
                                );
                                if (!hiddenEl) return;
                                var proto = hiddenEl.tagName === 'TEXTAREA'
                                    ? window.HTMLTextAreaElement.prototype
                                    : window.HTMLInputElement.prototype;
                                var descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
                                if (descriptor && descriptor.set) {
                                    descriptor.set.call(hiddenEl, active.join(','));
                                } else {
                                    hiddenEl.value = active.join(',');
                                }
                                hiddenEl.dispatchEvent(new InputEvent('input', {
                                    bubbles: true,
                                    composed: true,
                                    inputType: 'insertText',
                                    data: null
                                }));
                                hiddenEl.dispatchEvent(new Event('change', {
                                    bubbles: true,
                                    composed: true
                                }));
                            };
                            window.__setWorkflowChipActive = function(chip, active) {
                                if (!chip) return;
                                chip.classList.toggle('chip-active', active);
                                chip.setAttribute('aria-pressed', active ? 'true' : 'false');
                            };
                            window.__updateWorkflowFilterQuery = function(root) {
                                if (!root) return;
                                var roles = Array.from(root.querySelectorAll(
                                    '[data-filter-group="role"].chip-active'
                                )).map(function(c) { return c.textContent; });
                                var features = Array.from(root.querySelectorAll(
                                    '[data-filter-group="feature"].chip-active'
                                )).map(function(c) { return c.textContent; });
                                var query = root.querySelector('#wf-filter-query');
                                if (!query) return;
                                var featureText = features.indexOf('All') >= 0
                                    ? 'All'
                                    : features.join(' or ');
                                var parts = [
                                    'Role: ' + roles.join(' or '),
                                    'Step feature: ' + featureText
                                ];
                                var agents = Array.from(root.querySelectorAll(
                                    '[data-filter-group="agent"].chip-active'
                                ));
                                if (root.querySelector('[data-filter-group="agent"]')) {
                                    var agentAll = agents.some(function(c) {
                                        return c.dataset.filter === 'agent:All';
                                    });
                                    var agentText = agentAll
                                        ? 'All'
                                        : agents.map(function(c) { return c.textContent; }).join(' or ');
                                    parts.push('Agent: ' + agentText);
                                }
                                query.textContent = parts.join(' · ');
                            };
                            /* Pure chip state machine, unit-tested in
                               tests/test_workflow_filtering.py by executing this
                               exact source in Node. Keep it DOM-free. */
                            /* __WF_CHIP_STATE_BEGIN__ */
                            window.__wfComputeChipState = function(state, action) {
                                var cloneFlags = function(src) {
                                    var out = {};
                                    Object.keys(src).forEach(function(k) { out[k] = !!src[k]; });
                                    return out;
                                };
                                var roles = cloneFlags(state.roles);
                                var features = cloneFlags(state.features);
                                var agents = cloneFlags(state.agents);
                                var rejected = false;
                                var toggleExclusiveAll = function(group, allKey, name) {
                                    if (name === allKey) {
                                        Object.keys(group).forEach(function(k) { group[k] = (k === allKey); });
                                        return;
                                    }
                                    group[name] = !group[name];
                                    group[allKey] = false;
                                    var anySpecific = Object.keys(group).some(function(k) {
                                        return k !== allKey && group[k];
                                    });
                                    if (!anySpecific) {
                                        group[allKey] = true;
                                    }
                                };
                                if (action.type === 'reset') {
                                    Object.keys(roles).forEach(function(k) { roles[k] = true; });
                                    Object.keys(features).forEach(function(k) { features[k] = (k === 'All'); });
                                    Object.keys(agents).forEach(function(k) { agents[k] = (k === 'agent:All'); });
                                } else if (action.group === 'role') {
                                    var activeRoles = Object.keys(roles).filter(function(k) { return roles[k]; });
                                    if (roles[action.name] && activeRoles.length === 1) {
                                        /* Refuse to deselect the last active role. */
                                        rejected = true;
                                    } else {
                                        roles[action.name] = !roles[action.name];
                                    }
                                } else if (action.group === 'feature') {
                                    toggleExclusiveAll(features, 'All', action.name);
                                } else if (action.group === 'agent') {
                                    toggleExclusiveAll(agents, 'agent:All', action.name);
                                }
                                return {roles: roles, features: features, agents: agents, rejected: rejected};
                            };
                            /* __WF_CHIP_STATE_END__ */
                            window.__wfChipGroups = [
                                ['role', 'roles'],
                                ['feature', 'features'],
                                ['agent', 'agents']
                            ];
                            window.__wfReadChipState = function(bar) {
                                var state = {roles: {}, features: {}, agents: {}};
                                window.__wfChipGroups.forEach(function(pair) {
                                    var group = pair[0];
                                    var key = pair[1];
                                    bar.querySelectorAll('[data-filter-group="' + group + '"]').forEach(function(c) {
                                        state[key][c.dataset.filter] = c.classList.contains('chip-active');
                                    });
                                });
                                return state;
                            };
                            window.__wfApplyChipState = function(bar, state) {
                                window.__wfChipGroups.forEach(function(pair) {
                                    var group = pair[0];
                                    var key = pair[1];
                                    bar.querySelectorAll('[data-filter-group="' + group + '"]').forEach(function(c) {
                                        window.__setWorkflowChipActive(c, !!state[key][c.dataset.filter]);
                                    });
                                });
                            };
                            if (!window.__wfChipHandlerAttached) {
                                window.__wfChipHandlerAttached = true;
                                document.addEventListener('click', function(e) {
                                    var chip = e.target.closest('.filter-chip');
                                    var reset = e.target.closest('[data-wf-action="reset-filters"]');
                                    if (!chip && !reset) return;
                                    var root = (chip || reset).closest('#wf-filter-chips');
                                    if (!root) return;
                                    var bar = root.querySelector('#wf-filter-bar');
                                    if (!bar) return;
                                    var action = reset
                                        ? {type: 'reset'}
                                        : {type: 'toggle', group: chip.dataset.filterGroup, name: chip.dataset.filter};
                                    var next = window.__wfComputeChipState(
                                        window.__wfReadChipState(bar), action
                                    );
                                    if (next.rejected) {
                                        var roleGroup = chip.closest('.filter-group');
                                        if (roleGroup) {
                                            roleGroup.classList.remove('filter-group-attention');
                                            void roleGroup.offsetWidth;
                                            roleGroup.classList.add('filter-group-attention');
                                            window.setTimeout(function() {
                                                roleGroup.classList.remove('filter-group-attention');
                                            }, 900);
                                        }
                                        return;
                                    }
                                    window.__wfApplyChipState(bar, next);
                                    window.__updateWorkflowFilterQuery(root);
                                    window.__syncWorkflowFilters(bar);
                                });
                            }

                            /* Detail tab click handler (delegated, survives detail HTML replacement) */
                            if (!window.__dpTabHandlerAttached) {
                                window.__dpTabHandlerAttached = true;
                                document.addEventListener('click', function(e) {
                                    var tab = e.target.closest('.dp-tab');
                                    if (!tab) return;
                                    var panel = tab.closest('.dp-panel');
                                    if (!panel) return;
                                    panel.querySelectorAll('.dp-tab').forEach(function(x) {
                                        x.classList.remove('dp-tab-active');
                                    });
                                    panel.querySelectorAll('.dp-tab-content').forEach(function(x) {
                                        x.classList.remove('dp-tab-visible');
                                    });
                                    tab.classList.add('dp-tab-active');
                                    var content = panel.querySelector(
                                        '[data-tab-content="' + tab.dataset.tab + '"]'
                                    );
                                    if (content) content.classList.add('dp-tab-visible');
                                });
                            }

                            /* Card click handler */
                            function selectCard(card, opts) {
                                if (!card) return;
                                opts = opts || {};
                                var pushHistory = opts.pushHistory !== false;
                                document.querySelectorAll('.wf-card').forEach(function(c) {
                                    c.classList.remove('wf-active');
                                });
                                card.classList.add('wf-active');
                                var idx = card.dataset.stepIdx;
                                /* Push on user navigation so Back works; replace on load/restore.
                                   Always include pathname+search so a <base href> or reverse-proxy
                                   prefix cannot resolve '#step-N' to the site root. */
                                if (idx != null) {
                                    window.__wfSelectedStep = idx;
                                    var targetHash = '#step-' + idx;
                                    var histState = { tvTab: 'Workflow' };
                                    var nextUrl = window.tvWorkflowStepUrl(idx);
                                    if (pushHistory && window.location.hash !== targetHash) {
                                        history.pushState(histState, '', nextUrl);
                                    } else {
                                        history.replaceState(histState, '', nextUrl);
                                    }
                                }
                                var storeEl = document.querySelector('#wf-detail-store [data-b64]');
                                var target = document.getElementById('wf-detail-content');
                                if (!storeEl || !target) return;
                                try {
                                    var details = JSON.parse(atob(storeEl.dataset.b64));
                                    if (details[idx] != null) {
                                        target.innerHTML = details[idx];
                                        var detailPanel = target.closest('.detail-panel');
                                        if (detailPanel) detailPanel.scrollTop = 0;
                                    }
                                } catch(ex) { console.error('wf-click:', ex); }
                            }
                            window.tvSelectWorkflowCard = selectCard;
                            element.addEventListener('click', function(e) {
                                selectCard(e.target.closest('.wf-card'));
                            });

                            /* Keyboard navigation: j/k for next/prev step */
                            document.addEventListener('keydown', function(e) {
                                var tag = (e.target.tagName || '').toLowerCase();
                                if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
                                if (e.key !== 'j' && e.key !== 'k') return;
                                var cards = Array.from(document.querySelectorAll('.wf-card'));
                                if (!cards.length) return;
                                var activeIdx = cards.findIndex(function(c) { return c.classList.contains('wf-active'); });
                                var nextIdx;
                                if (e.key === 'j') {
                                    nextIdx = activeIdx < 0 ? 0 : Math.min(activeIdx + 1, cards.length - 1);
                                } else {
                                    nextIdx = activeIdx < 0 ? 0 : Math.max(activeIdx - 1, 0);
                                }
                                cards[nextIdx].scrollIntoView({behavior:'smooth', block:'center'});
                                selectCard(cards[nextIdx]);
                            });

                            /* Deep link: on load, wait for the hashed card
                               (Workflow HTML often arrives after js_on_load). */
                            (function waitForDeepLink() {
                                var hash = window.location.hash;
                                var m = hash && hash.match(/^#step-(\\d+)$/);
                                if (m) {
                                    window.tvGotoWorkflowStep(Number(m[1]), { restore: true });
                                }
                            })();

                            /* Hidden-selection watcher: when a re-render (filter,
                               search, or label upload) removes the selected step's
                               card, tell the user in the detail panel; when the
                               card comes back, restore its detail. */
                            if (!window.__wfHiddenStepObserverAttached) {
                                window.__wfHiddenStepObserverAttached = true;
                                var hiddenStepCheckPending = null;
                                var checkSelectedStepVisible = function() {
                                    hiddenStepCheckPending = null;
                                    var target = document.getElementById('wf-detail-content');
                                    if (!target) return;
                                    if (target.querySelector('[data-wf-detail-placeholder]')) {
                                        /* The app reset the detail panel (new trajectory
                                           or label upload): the old selection is gone. */
                                        window.__wfSelectedStep = null;
                                        return;
                                    }
                                    var idx = window.__wfSelectedStep;
                                    if (idx == null) return;
                                    var card = document.getElementById('wf-card-' + idx);
                                    if (card) {
                                        if (target.querySelector('[data-wf-hidden-msg]')) {
                                            selectCard(card, { pushHistory: false });
                                        }
                                        return;
                                    }
                                    if (!document.querySelector('.wf-card')) return;
                                    if (target.querySelector('[data-wf-hidden-msg]')) return;
                                    target.innerHTML =
                                        "<div data-wf-hidden-msg='1'" +
                                        " style='padding:2em 1em;text-align:center;color:var(--ov-muted);'>" +
                                        "<p style='font-size:15px;margin-bottom:0.5em;'>" +
                                        "Selected step is hidden by the current filters</p>" +
                                        "<p style='font-size:12px;'>Adjust the filters to show it again.</p>" +
                                        "</div>";
                                };
                                new MutationObserver(function() {
                                    if (hiddenStepCheckPending) return;
                                    hiddenStepCheckPending = window.setTimeout(checkSelectedStepVisible, 120);
                                }).observe(document.body, {childList: true, subtree: true});
                            }

                            /* Auto-select first assistant card if no hash link */
                            setTimeout(function() {
                                if (window.location.hash) return;
                                var cards = document.querySelectorAll('.wf-card');
                                for (var i = 0; i < cards.length; i++) {
                                    var badges = cards[i].querySelectorAll('.wf-badge');
                                    for (var j = 0; j < badges.length; j++) {
                                        if (badges[j].textContent.trim() === 'Assistant') {
                                            selectCard(cards[i], { pushHistory: false });
                                            return;
                                        }
                                    }
                                }
                                if (cards.length) selectCard(cards[0], { pushHistory: false });
                            }, 600);
                            """


@dataclass
class WorkflowRefs:
    wf_toc_toggle: gr.Button
    wf_filter_chips_html: gr.HTML
    wf_search: gr.Textbox
    wf_filter_hidden: gr.Textbox
    wf_count_html: gr.HTML
    toc_html: gr.HTML
    workflow_html: gr.HTML
    detail_html: gr.HTML
    detail_store: gr.HTML


def layout() -> WorkflowRefs:
    with gr.TabItem("Workflow"):
        with gr.Row(equal_height=True):
            wf_toc_toggle = gr.Button("TOC", variant="secondary", scale=0, min_width=50)
            wf_filter_chips_html = gr.HTML(
                render_filter_chips(),
                elem_id="wf-filter-chips",
            )
            wf_search = gr.Textbox(
                label="Search",
                placeholder="Filter by keyword...",
                scale=1,
            )
        # Hidden textbox that JS writes active filters into (comma-separated)
        wf_filter_hidden = gr.Textbox(
            value=",".join(FILTER_CHIPS_DEFAULT),
            # Keep the component mounted so the delegated chip handler
            # can update it and trigger Gradio's input event.  Gradio
            # omits visible=False components from the browser DOM.
            visible=True,
            elem_id="wf-filter-hidden",
        )
        wf_count_html = gr.HTML("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=0, min_width=150):
                toc_html = gr.HTML("", elem_id="wf-toc-container")
            with gr.Column(scale=3, min_width=400):
                workflow_html = gr.HTML(
                    "<div style='padding:3em;color:var(--ov-muted);text-align:center;"
                    "font-size:15px;'>Load a trajectory to see the step flow.</div>",
                    js_on_load=WORKFLOW_JS,
                )
            with gr.Column(scale=2, min_width=300, elem_classes=["detail-panel"]):
                detail_html = gr.HTML(
                    DETAIL_PLACEHOLDER,
                    elem_id="wf-detail-panel",
                )
        detail_store = gr.HTML("", elem_id="wf-detail-store")
    return WorkflowRefs(
        wf_toc_toggle=wf_toc_toggle,
        wf_filter_chips_html=wf_filter_chips_html,
        wf_search=wf_search,
        wf_filter_hidden=wf_filter_hidden,
        wf_count_html=wf_count_html,
        toc_html=toc_html,
        workflow_html=workflow_html,
        detail_html=detail_html,
        detail_store=detail_store,
    )


def load_slots(refs: WorkflowRefs) -> dict:
    """Named Gradio components filled by the load packer."""
    return {
        "wf_filter_chips_html": refs.wf_filter_chips_html,
        "wf_filter_hidden": refs.wf_filter_hidden,
        "wf_count_html": refs.wf_count_html,
        "toc_html": refs.toc_html,
        "workflow_html": refs.workflow_html,
        "detail_store": refs.detail_store,
        "detail_html": refs.detail_html,
    }


def pack_load(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict:
    """Named Workflow values for a load (empty when *session* is None)."""
    del dark, banner
    if session is None:
        return {
            "wf_filter_chips_html": render_filter_chips(),
            "wf_filter_hidden": ",".join(FILTER_CHIPS_DEFAULT),
            "wf_count_html": "",
            "toc_html": "",
            "workflow_html": "<div></div>",
            "detail_store": "",
            "detail_html": DETAIL_PLACEHOLDER,
        }
    wf = build_workflow_outputs(session.steps)
    return {
        "wf_filter_chips_html": wf["wf_chips"],
        "wf_filter_hidden": wf["wf_filter_val"],
        "wf_count_html": wf["wf_count"],
        "toc_html": wf["toc_html_val"],
        "workflow_html": wf["wf_html"],
        "detail_store": wf["detail_store_val"],
        "detail_html": DETAIL_PLACEHOLDER,
    }


def bind(refs: WorkflowRefs, shared: SharedState) -> None:
    state_steps = shared.state_steps
    wf_filter_hidden = refs.wf_filter_hidden
    wf_search = refs.wf_search
    toc_html = refs.toc_html
    workflow_html = refs.workflow_html
    wf_count_html = refs.wf_count_html

    def do_filter_workflow(steps, filter_csv, keyword, current_toc):
        """Re-render Workflow cards, count, and TOC with filters applied."""
        return build_filtered_workflow_outputs(steps, filter_csv, keyword, current_toc)

    def on_toc_toggle(current_toc):
        """Toggle TOC sidebar visibility via CSS class."""
        if not current_toc:
            return current_toc
        if "toc-hidden" in current_toc:
            return current_toc.replace("toc-hidden", "").strip()
        return current_toc.replace("wf-toc-sidebar", "wf-toc-sidebar toc-hidden")

    refs.wf_toc_toggle.click(
        fn=on_toc_toggle,
        inputs=[toc_html],
        outputs=[toc_html],
    )

    wf_filter_hidden.change(
        fn=do_filter_workflow,
        inputs=[state_steps, wf_filter_hidden, wf_search, toc_html],
        outputs=[workflow_html, wf_count_html, toc_html],
    )
    wf_search.change(
        fn=do_filter_workflow,
        inputs=[state_steps, wf_filter_hidden, wf_search, toc_html],
        outputs=[workflow_html, wf_count_html, toc_html],
    )
