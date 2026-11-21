//! `/model` picker modal: pick a (provider, model) pair and a thinking-effort
//! tier and apply both at once (#39).
//!
//! Two side-by-side panes — Models on the left, Thinking effort on the
//! right. Tab swaps focus, ↑/↓ moves within the focused pane, Enter applies
//! both and closes the modal, Esc cancels.
//!
//! The model pane is grouped into provider sections — DeepSeek first, then
//! Ollama. Ollama is split three ways:
//!   • **Local** — models pulled and resident on the local daemon.
//!   • **Cloud** — `:cloud`-suffixed models (Ollama Turbo / hosted).
//!     Discovered from `~/.ollama/config.json` + `~/.ollama/server.json`
//!     plus `[providers.ollama] cloud_models` in `~/.deepseek/config.toml`.
//!     There is no public catalog API; this is the best we can do.
//!   • **Library (pullable)** — scraped from `https://ollama.com/library`.
//!     Rows the user hasn't pulled get a `(pull first)` hint; selecting
//!     one still emits an apply event but chat will fail until they run
//!     `ollama pull <name>` themselves.
//!
//! Section headers are non-selectable; navigation skips over them. The
//! pane scrolls when rows exceed the visible height (PageUp/PageDown,
//! Home/End).
//!
//! The effort pane intentionally only exposes `Off / High / Max`. For
//! Ollama rows the effort pane is disabled — Ollama doesn't speak the
//! DeepSeek thinking-effort protocol, so we force `Off` and grey the pane.
//!
//! On apply we emit a [`ViewEvent::ModelPickerApplied`] with the resolved
//! model id, effort tier, and (when crossing providers) the target provider.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    prelude::Stylize,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::OnceLock;
use std::time::Duration;

use crate::config::{ApiProvider, Config};
use crate::palette;
use crate::tui::app::{App, ReasoningEffort};
use crate::tui::views::{ModalKind, ModalView, ViewAction, ViewEvent};

const DEEPSEEK_ROWS: &[(&str, &str)] = &[
    ("auto", "select per turn"),
    ("deepseek-v4-pro", "flagship"),
    ("deepseek-v4-flash", "fast / cheap"),
];

const PICKER_EFFORTS: &[ReasoningEffort] = &[
    ReasoningEffort::Auto,
    ReasoningEffort::Off,
    ReasoningEffort::High,
    ReasoningEffort::Max,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pane {
    Model,
    Effort,
}

#[derive(Debug, Clone)]
struct Row {
    label: String,
    hint: String,
    kind: RowKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RowKind {
    /// Non-selectable section header.
    Header,
    /// A DeepSeek model. Apply normalizes provider to Deepseek.
    Deepseek,
    /// An Ollama model resident on the local daemon.
    OllamaLocal,
    /// An Ollama Cloud / Turbo model (`:cloud`-suffixed). Same chat
    /// endpoint as local — the daemon transparently routes.
    OllamaCloud,
    /// A model in the Ollama public library. May or may not be pulled
    /// locally; if not, chat will fail until `ollama pull <name>` runs.
    OllamaLibrary,
    /// Sentinel row shown when Ollama is configured but unreachable.
    /// Selecting it is a no-op.
    OllamaUnavailable,
    /// Preserves the user's currently active model when it isn't in any
    /// known section.
    Custom,
}

impl RowKind {
    fn is_selectable(&self) -> bool {
        !matches!(self, RowKind::Header)
    }

    fn is_ollama(&self) -> bool {
        matches!(
            self,
            RowKind::OllamaLocal
                | RowKind::OllamaCloud
                | RowKind::OllamaLibrary
                | RowKind::OllamaUnavailable
        )
    }
}

/// All Ollama-flavoured input lists the picker needs.
#[derive(Debug, Default, Clone)]
pub struct OllamaSources {
    pub local: Vec<String>,
    pub cloud: Vec<String>,
    pub library: Vec<String>,
}

pub struct ModelPickerView {
    initial_model: String,
    initial_effort: ReasoningEffort,
    initial_provider: ApiProvider,
    rows: Vec<Row>,
    selected_row_idx: usize,
    selected_effort_idx: usize,
    focus: Pane,
    /// Top of the visible window in the model pane. Adjusted whenever
    /// the selection moves out of view.
    scroll_offset: usize,
    /// Captured during render so key handlers can scroll by visible page.
    /// Cell since render takes &self.
    visible_rows: Cell<usize>,
}

impl ModelPickerView {
    #[must_use]
    pub fn new(app: &App, sources: OllamaSources) -> Self {
        let initial_provider = app.api_provider;
        let initial_model = if app.auto_model {
            "auto".to_string()
        } else {
            app.model.clone()
        };
        let initial_effort = app.reasoning_effort;

        let rows = build_rows(&initial_model, initial_provider, &sources);
        let selected_row_idx = rows
            .iter()
            .position(|row| row_matches_current(row, initial_provider, &initial_model))
            .or_else(|| rows.iter().position(|row| row.kind.is_selectable()))
            .unwrap_or(0);

        let normalized = match initial_effort {
            ReasoningEffort::Low | ReasoningEffort::Medium => ReasoningEffort::High,
            other => other,
        };
        let selected_effort_idx = PICKER_EFFORTS
            .iter()
            .position(|e| *e == normalized)
            .unwrap_or(2);

        Self {
            initial_model,
            initial_effort,
            initial_provider,
            rows,
            selected_row_idx,
            selected_effort_idx,
            focus: Pane::Model,
            scroll_offset: 0,
            visible_rows: Cell::new(10),
        }
    }

    fn selected_row(&self) -> &Row {
        &self.rows[self.selected_row_idx]
    }

    fn resolved_model(&self) -> String {
        match self.selected_row().kind {
            RowKind::Header | RowKind::OllamaUnavailable => self.initial_model.clone(),
            RowKind::Custom => self.initial_model.clone(),
            RowKind::Deepseek
            | RowKind::OllamaLocal
            | RowKind::OllamaCloud
            | RowKind::OllamaLibrary => self.selected_row().label.clone(),
        }
    }

    fn resolved_provider(&self) -> Option<ApiProvider> {
        match self.selected_row().kind {
            RowKind::OllamaLocal | RowKind::OllamaCloud | RowKind::OllamaLibrary => {
                Some(ApiProvider::Ollama)
            }
            RowKind::Deepseek => Some(ApiProvider::Deepseek),
            _ => None,
        }
    }

    fn resolved_effort(&self) -> ReasoningEffort {
        if self.thinking_pane_disabled() {
            return ReasoningEffort::Off;
        }
        if self.resolved_model().trim().eq_ignore_ascii_case("auto") {
            return ReasoningEffort::Auto;
        }
        PICKER_EFFORTS[self.selected_effort_idx]
    }

    fn thinking_pane_disabled(&self) -> bool {
        self.selected_row().kind.is_ollama()
    }

    fn move_selection(&mut self, step: isize) {
        if self.rows.is_empty() {
            return;
        }
        let len = self.rows.len() as isize;
        let mut idx = self.selected_row_idx as isize;
        loop {
            idx += step;
            if idx < 0 || idx >= len {
                return;
            }
            if self.rows[idx as usize].kind.is_selectable() {
                self.selected_row_idx = idx as usize;
                self.ensure_selection_visible();
                return;
            }
        }
    }

    fn jump_to_first_selectable(&mut self) {
        if let Some(idx) = self.rows.iter().position(|r| r.kind.is_selectable()) {
            self.selected_row_idx = idx;
            self.ensure_selection_visible();
        }
    }

    fn jump_to_last_selectable(&mut self) {
        if let Some(idx) = self.rows.iter().rposition(|r| r.kind.is_selectable()) {
            self.selected_row_idx = idx;
            self.ensure_selection_visible();
        }
    }

    fn page(&mut self, delta: isize) {
        let visible = self.visible_rows.get().max(1) as isize;
        let len = self.rows.len() as isize;
        if len == 0 {
            return;
        }
        let mut idx = (self.selected_row_idx as isize + delta * visible)
            .clamp(0, len - 1);
        // Snap to a selectable row in the direction of travel.
        let dir: isize = if delta >= 0 { 1 } else { -1 };
        loop {
            if self.rows[idx as usize].kind.is_selectable() {
                self.selected_row_idx = idx as usize;
                self.ensure_selection_visible();
                return;
            }
            idx += dir;
            if idx < 0 || idx >= len {
                // Fallback: try the other direction.
                self.move_selection(-dir);
                return;
            }
        }
    }

    /// Clamp `scroll_offset` so the selected row is within the visible
    /// window. Called after every navigation event.
    fn ensure_selection_visible(&mut self) {
        let visible = self.visible_rows.get().max(1);
        if self.selected_row_idx < self.scroll_offset {
            self.scroll_offset = self.selected_row_idx;
        } else if self.selected_row_idx >= self.scroll_offset + visible {
            self.scroll_offset = self.selected_row_idx + 1 - visible;
        }
        let max_offset = self.rows.len().saturating_sub(visible);
        if self.scroll_offset > max_offset {
            self.scroll_offset = max_offset;
        }
    }

    fn move_up(&mut self) {
        match self.focus {
            Pane::Model => self.move_selection(-1),
            Pane::Effort => {
                if self.thinking_pane_disabled() {
                    return;
                }
                if self.selected_effort_idx > 0 {
                    self.selected_effort_idx -= 1;
                }
            }
        }
    }

    fn move_down(&mut self) {
        match self.focus {
            Pane::Model => self.move_selection(1),
            Pane::Effort => {
                if self.thinking_pane_disabled() {
                    return;
                }
                let max = PICKER_EFFORTS.len().saturating_sub(1);
                if self.selected_effort_idx < max {
                    self.selected_effort_idx += 1;
                }
            }
        }
    }

    fn toggle_focus(&mut self) {
        if self.focus == Pane::Model && self.thinking_pane_disabled() {
            return;
        }
        self.focus = match self.focus {
            Pane::Model => Pane::Effort,
            Pane::Effort => Pane::Model,
        };
    }

    fn build_event(&self) -> Option<ViewEvent> {
        if matches!(self.selected_row().kind, RowKind::OllamaUnavailable) {
            return None;
        }
        let model = self.resolved_model();
        let provider = self
            .resolved_provider()
            .filter(|p| *p != self.initial_provider);
        Some(ViewEvent::ModelPickerApplied {
            model,
            effort: self.resolved_effort(),
            previous_model: self.initial_model.clone(),
            previous_effort: self.initial_effort,
            provider,
        })
    }

    fn render_model_pane(&self, area: Rect, buf: &mut Buffer, focused: bool) {
        let border_style = if focused {
            Style::default().fg(palette::DEEPSEEK_SKY)
        } else {
            Style::default().fg(palette::BORDER_COLOR)
        };
        let block = Block::default()
            .title(Line::from(Span::styled(
                " Model ",
                Style::default().fg(palette::TEXT_PRIMARY).bold(),
            )))
            .borders(Borders::ALL)
            .border_style(border_style)
            .style(Style::default());
        let inner = block.inner(area);
        block.render(area, buf);

        let visible = inner.height as usize;
        self.visible_rows.set(visible.max(1));

        let max_offset = self.rows.len().saturating_sub(visible);
        let start = self.scroll_offset.min(max_offset);
        let end = (start + visible).min(self.rows.len());

        let mut lines = Vec::with_capacity(end.saturating_sub(start));
        for (idx, row) in self.rows[start..end].iter().enumerate() {
            let real_idx = start + idx;
            match row.kind {
                RowKind::Header => {
                    lines.push(Line::from(vec![
                        Span::raw(" "),
                        Span::styled(
                            row.label.clone(),
                            Style::default()
                                .fg(palette::TEXT_MUTED)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                }
                _ => {
                    let is_selected = real_idx == self.selected_row_idx;
                    let marker = if is_selected { "▸" } else { " " };
                    let label_style = if is_selected {
                        Style::default()
                            .fg(palette::SELECTION_TEXT)
                            .bg(palette::SELECTION_BG)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(palette::TEXT_PRIMARY)
                    };
                    let hint_style = if is_selected {
                        Style::default()
                            .fg(palette::SELECTION_TEXT)
                            .bg(palette::SELECTION_BG)
                    } else {
                        Style::default().fg(palette::TEXT_MUTED)
                    };
                    let mut spans = vec![
                        Span::raw("   "),
                        Span::styled(marker, label_style),
                        Span::raw(" "),
                        Span::styled(row.label.clone(), label_style),
                    ];
                    if !row.hint.is_empty() {
                        spans.push(Span::raw("  "));
                        spans.push(Span::styled(format!("({})", row.hint), hint_style));
                    }
                    lines.push(Line::from(spans));
                }
            }
        }
        // Scroll indicator when there's overflow above/below.
        if start > 0 || end < self.rows.len() {
            // Replace last line with a hint when more rows exist below.
            if end < self.rows.len()
                && let Some(last) = lines.last_mut()
            {
                let suffix = format!("  ↓ {} more", self.rows.len() - end);
                last.spans.push(Span::styled(
                    suffix,
                    Style::default().fg(palette::TEXT_MUTED),
                ));
            }
        }
        Paragraph::new(lines).render(inner, buf);
    }

    fn render_effort_pane(&self, area: Rect, buf: &mut Buffer, focused: bool) {
        let disabled = self.thinking_pane_disabled();
        let border_style = if focused && !disabled {
            Style::default().fg(palette::DEEPSEEK_SKY)
        } else {
            Style::default().fg(palette::BORDER_COLOR)
        };
        let title_style = if disabled {
            Style::default().fg(palette::TEXT_MUTED)
        } else {
            Style::default().fg(palette::TEXT_PRIMARY).bold()
        };
        let block = Block::default()
            .title(Line::from(Span::styled(" Thinking ", title_style)))
            .borders(Borders::ALL)
            .border_style(border_style)
            .style(Style::default());
        let inner = block.inner(area);
        block.render(area, buf);

        if disabled {
            let lines = vec![
                Line::from(""),
                Line::from(vec![
                    Span::raw(" "),
                    Span::styled(
                        "n/a for Ollama",
                        Style::default().fg(palette::TEXT_MUTED),
                    ),
                ]),
            ];
            Paragraph::new(lines).render(inner, buf);
            return;
        }

        let mut lines = Vec::with_capacity(PICKER_EFFORTS.len());
        for (idx, effort) in PICKER_EFFORTS.iter().enumerate() {
            let is_selected = idx == self.selected_effort_idx;
            let marker = if is_selected { "▸" } else { " " };
            let label_style = if is_selected {
                Style::default()
                    .fg(palette::SELECTION_TEXT)
                    .bg(palette::SELECTION_BG)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(palette::TEXT_PRIMARY)
            };
            let hint = match effort {
                ReasoningEffort::Auto => "auto-select per turn",
                ReasoningEffort::Off => "thinking disabled",
                ReasoningEffort::High => "thinking enabled (default)",
                ReasoningEffort::Max => "thinking enabled, max effort",
                _ => "",
            };
            let hint_style = if is_selected {
                Style::default()
                    .fg(palette::SELECTION_TEXT)
                    .bg(palette::SELECTION_BG)
            } else {
                Style::default().fg(palette::TEXT_MUTED)
            };
            let mut spans = vec![
                Span::raw(" "),
                Span::styled(marker, label_style),
                Span::raw(" "),
                Span::styled(effort.short_label().to_string(), label_style),
            ];
            if !hint.is_empty() {
                spans.push(Span::raw("  "));
                spans.push(Span::styled(format!("({hint})"), hint_style));
            }
            lines.push(Line::from(spans));
        }
        Paragraph::new(lines).render(inner, buf);
    }
}

fn row_matches_current(row: &Row, current_provider: ApiProvider, current_model: &str) -> bool {
    match row.kind {
        RowKind::Header | RowKind::OllamaUnavailable => false,
        RowKind::Custom => row.label == current_model,
        RowKind::Deepseek => {
            current_provider == ApiProvider::Deepseek && row.label == current_model
        }
        RowKind::OllamaLocal | RowKind::OllamaCloud | RowKind::OllamaLibrary => {
            current_provider == ApiProvider::Ollama && row.label == current_model
        }
    }
}

fn build_rows(
    initial_model: &str,
    initial_provider: ApiProvider,
    sources: &OllamaSources,
) -> Vec<Row> {
    let mut rows = Vec::new();

    rows.push(Row {
        label: "DeepSeek".to_string(),
        hint: String::new(),
        kind: RowKind::Header,
    });
    for (id, hint) in DEEPSEEK_ROWS {
        rows.push(Row {
            label: (*id).to_string(),
            hint: (*hint).to_string(),
            kind: RowKind::Deepseek,
        });
    }

    // Ollama Local — always render the header so users know it's an option.
    rows.push(Row {
        label: "Ollama (local)".to_string(),
        hint: String::new(),
        kind: RowKind::Header,
    });
    if sources.local.is_empty() && sources.cloud.is_empty() && sources.library.is_empty() {
        rows.push(Row {
            label: "(not running — start ollama and reopen)".to_string(),
            hint: String::new(),
            kind: RowKind::OllamaUnavailable,
        });
    } else if sources.local.is_empty() {
        rows.push(Row {
            label: "(no local models pulled)".to_string(),
            hint: String::new(),
            kind: RowKind::OllamaUnavailable,
        });
    } else {
        for model in &sources.local {
            rows.push(Row {
                label: model.clone(),
                hint: String::new(),
                kind: RowKind::OllamaLocal,
            });
        }
    }

    // Ollama Cloud — only render the section if we have something to show.
    let cloud_dedup: Vec<&String> = sources
        .cloud
        .iter()
        .filter(|m| !sources.local.contains(*m))
        .collect();
    if !cloud_dedup.is_empty() {
        rows.push(Row {
            label: "Ollama (cloud)".to_string(),
            hint: String::new(),
            kind: RowKind::Header,
        });
        for model in &cloud_dedup {
            rows.push(Row {
                label: (*model).clone(),
                hint: String::new(),
                kind: RowKind::OllamaCloud,
            });
        }
    }

    // Ollama Library — pullable catalog. Dedup against local + cloud so
    // the same model isn't listed in two sections.
    let library_dedup: Vec<&String> = sources
        .library
        .iter()
        .filter(|m| !sources.local.contains(*m) && !sources.cloud.contains(*m))
        .collect();
    if !library_dedup.is_empty() {
        rows.push(Row {
            label: "Ollama (library — pullable)".to_string(),
            hint: String::new(),
            kind: RowKind::Header,
        });
        for model in &library_dedup {
            rows.push(Row {
                label: (*model).clone(),
                hint: "pull first: ollama pull".to_string(),
                kind: RowKind::OllamaLibrary,
            });
        }
    }

    let current_in_known = rows
        .iter()
        .any(|row| row_matches_current(row, initial_provider, initial_model));
    if !current_in_known && !initial_model.trim().is_empty() {
        rows.push(Row {
            label: initial_model.to_string(),
            hint: "current (custom)".to_string(),
            kind: RowKind::Custom,
        });
    }

    rows
}

fn ollama_base_url() -> String {
    std::env::var("OLLAMA_BASE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| crate::config::DEFAULT_OLLAMA_BASE_URL.to_string())
}

/// Locally-installed Ollama model ids via the OpenAI-compatible
/// `/v1/models` endpoint. Short timeout — picker is opening, so we can't
/// block. Errors collapse to an empty Vec.
pub async fn fetch_ollama_local() -> Vec<String> {
    #[derive(serde::Deserialize)]
    struct ModelEntry {
        id: String,
    }
    #[derive(serde::Deserialize)]
    struct ListResponse {
        data: Vec<ModelEntry>,
    }

    let base = ollama_base_url();
    let trimmed = base.trim_end_matches('/');
    let url = format!("{trimmed}/models");
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(200))
        .build()
    {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let parsed: ListResponse = match resp.json().await {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };
    parsed.data.into_iter().map(|m| m.id).collect()
}

/// Cloud-hosted models the user has touched. Read from
/// `~/.ollama/config.json` (`integrations.*.models[]`) and
/// `~/.ollama/server.json` (`aliases.*.target`). No public API exists for
/// the full Ollama Cloud catalog, so this is the canonical source.
pub fn discover_user_cloud_models(config: &Config) -> Vec<String> {
    let mut found: BTreeSet<String> = BTreeSet::new();

    if let Some(home) = dirs::home_dir() {
        let cfg_path = home.join(".ollama").join("config.json");
        if let Ok(text) = std::fs::read_to_string(&cfg_path)
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(&text)
            && let Some(integrations) = value.get("integrations").and_then(|v| v.as_object())
        {
            for (_, integration) in integrations {
                if let Some(models) = integration.get("models").and_then(|v| v.as_array()) {
                    for m in models {
                        if let Some(s) = m.as_str() {
                            insert_if_modelish(&mut found, s);
                        }
                    }
                }
            }
        }

        let srv_path = home.join(".ollama").join("server.json");
        if let Ok(text) = std::fs::read_to_string(&srv_path)
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(&text)
            && let Some(aliases) = value.get("aliases").and_then(|v| v.as_array())
        {
            for alias in aliases {
                if let Some(target) = alias.get("target").and_then(|v| v.as_str()) {
                    insert_if_modelish(&mut found, target);
                }
            }
        }
    }

    if let Some(providers) = config.providers.as_ref()
        && let Some(extra) = &providers.ollama.cloud_models
    {
        for s in extra {
            insert_if_modelish(&mut found, s);
        }
    }

    found.into_iter().collect()
}

fn insert_if_modelish(set: &mut BTreeSet<String>, raw: &str) {
    let s = raw.trim();
    if s.is_empty() || s.contains(' ') {
        // Reject things like "Ollama Cloud" placeholder labels.
        return;
    }
    set.insert(s.to_string());
}

/// Scrape the Ollama public library catalog for pullable model names.
/// Cached for the rest of the process — re-opening the picker is instant
/// after the first fetch. 1.5s timeout since we hit ollama.com over WAN.
pub async fn fetch_ollama_library() -> Vec<String> {
    static CACHE: OnceLock<tokio::sync::Mutex<Option<Vec<String>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| tokio::sync::Mutex::new(None));
    {
        let guard = cache.lock().await;
        if let Some(cached) = guard.as_ref() {
            return cached.clone();
        }
    }

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(1500))
        .build()
    {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let resp = match client.get("https://ollama.com/library").send().await {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let text = match resp.text().await {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let mut set: BTreeSet<String> = BTreeSet::new();
    let re = match regex::Regex::new(r#"href="/library/([a-zA-Z0-9._-]+)""#) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    for cap in re.captures_iter(&text) {
        if let Some(m) = cap.get(1) {
            set.insert(m.as_str().to_string());
        }
    }
    let models: Vec<String> = set.into_iter().collect();

    let mut guard = cache.lock().await;
    *guard = Some(models.clone());
    models
}

impl ModalView for ModelPickerView {
    fn kind(&self) -> ModalKind {
        ModalKind::ModelPicker
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> ViewAction {
        match key.code {
            KeyCode::Esc => ViewAction::Close,
            KeyCode::Enter => match self.build_event() {
                Some(event) => ViewAction::EmitAndClose(event),
                None => ViewAction::None,
            },
            KeyCode::Up => {
                self.move_up();
                ViewAction::None
            }
            KeyCode::Down => {
                self.move_down();
                ViewAction::None
            }
            KeyCode::PageUp => {
                if self.focus == Pane::Model {
                    self.page(-1);
                }
                ViewAction::None
            }
            KeyCode::PageDown => {
                if self.focus == Pane::Model {
                    self.page(1);
                }
                ViewAction::None
            }
            KeyCode::Home => {
                if self.focus == Pane::Model {
                    self.jump_to_first_selectable();
                }
                ViewAction::None
            }
            KeyCode::End => {
                if self.focus == Pane::Model {
                    self.jump_to_last_selectable();
                }
                ViewAction::None
            }
            KeyCode::Tab | KeyCode::Right | KeyCode::Left | KeyCode::BackTab => {
                self.toggle_focus();
                ViewAction::None
            }
            _ => ViewAction::None,
        }
    }

    fn render(&self, area: Rect, buf: &mut Buffer) {
        let popup_width = 72.min(area.width.saturating_sub(4)).max(40);
        // Try to fit ~60% of available height; cap to row count + chrome.
        let preferred_height = (area.height.saturating_sub(4)).max(10);
        let popup_height = preferred_height.min((self.rows.len() as u16).saturating_add(4).max(10));
        let popup_area = Rect {
            x: area.x + (area.width.saturating_sub(popup_width)) / 2,
            y: area.y + (area.height.saturating_sub(popup_height)) / 2,
            width: popup_width,
            height: popup_height,
        };

        Clear.render(popup_area, buf);

        let outer = Block::default()
            .title(Line::from(Span::styled(
                " Model & thinking ",
                Style::default()
                    .fg(palette::DEEPSEEK_SKY)
                    .add_modifier(Modifier::BOLD),
            )))
            .title_bottom(Line::from(vec![
                Span::styled(" ↑↓ ", Style::default().fg(palette::TEXT_MUTED)),
                Span::raw("move "),
                Span::styled(" PgUp/PgDn ", Style::default().fg(palette::TEXT_MUTED)),
                Span::raw("page "),
                Span::styled(" Tab ", Style::default().fg(palette::TEXT_MUTED)),
                Span::raw("switch "),
                Span::styled(" Enter ", Style::default().fg(palette::TEXT_MUTED)),
                Span::raw("apply "),
                Span::styled(" Esc ", Style::default().fg(palette::TEXT_MUTED)),
                Span::raw("cancel "),
            ]))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(palette::BORDER_COLOR))
            .style(Style::default());
        let inner = outer.inner(popup_area);
        outer.render(popup_area, buf);

        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(inner);

        self.render_model_pane(columns[0], buf, self.focus == Pane::Model);
        self.render_effort_pane(columns[1], buf, self.focus == Pane::Effort);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::tui::app::{App, TuiOptions};
    use std::path::PathBuf;

    fn create_test_app() -> App {
        let options = TuiOptions {
            model: "deepseek-v4-pro".to_string(),
            workspace: PathBuf::from("."),
            config_path: None,
            config_profile: None,
            allow_shell: false,
            use_alt_screen: true,
            use_mouse_capture: false,
            use_bracketed_paste: true,
            max_subagents: 1,
            skills_dir: PathBuf::from("."),
            memory_path: PathBuf::from("memory.md"),
            notes_path: PathBuf::from("notes.txt"),
            mcp_config_path: PathBuf::from("mcp.json"),
            use_memory: false,
            start_in_agent_mode: true,
            skip_onboarding: true,
            yolo: false,
            resume_session_id: None,
            initial_input: None,
        };
        let mut app = App::new(options, &Config::default());
        app.model = "deepseek-v4-pro".to_string();
        app.reasoning_effort = ReasoningEffort::Max;
        app.api_provider = ApiProvider::Deepseek;
        app.auto_model = false;
        app
    }

    fn picker(app: &App, local: &[&str], cloud: &[&str], library: &[&str]) -> ModelPickerView {
        let sources = OllamaSources {
            local: local.iter().map(|s| s.to_string()).collect(),
            cloud: cloud.iter().map(|s| s.to_string()).collect(),
            library: library.iter().map(|s| s.to_string()).collect(),
        };
        ModelPickerView::new(app, sources)
    }

    #[test]
    fn picker_initial_selection_matches_app_state() {
        let mut app = create_test_app();
        app.model = "deepseek-v4-flash".to_string();
        let view = picker(&app, &[], &[], &[]);
        assert_eq!(view.resolved_model(), "deepseek-v4-flash");
        assert_eq!(view.resolved_effort(), ReasoningEffort::Max);
    }

    #[test]
    fn picker_initial_selection_matches_auto_state() {
        let mut app = create_test_app();
        app.model = "auto".to_string();
        app.auto_model = true;
        app.reasoning_effort = ReasoningEffort::Auto;
        let view = picker(&app, &[], &[], &[]);
        assert_eq!(view.resolved_model(), "auto");
        assert_eq!(view.resolved_effort(), ReasoningEffort::Auto);
    }

    #[test]
    fn picker_normalizes_low_medium_to_high() {
        let mut app = create_test_app();
        app.reasoning_effort = ReasoningEffort::Medium;
        let view = picker(&app, &[], &[], &[]);
        assert_eq!(view.resolved_effort(), ReasoningEffort::High);
    }

    #[test]
    fn picker_renders_three_ollama_sections_when_all_present() {
        let app = create_test_app();
        let view = picker(
            &app,
            &["qwen3.6:35b"],
            &["glm-5:cloud"],
            &["llama3.3", "deepseek-r1"],
        );
        let header_labels: Vec<&str> = view
            .rows
            .iter()
            .filter(|r| r.kind == RowKind::Header)
            .map(|r| r.label.as_str())
            .collect();
        assert_eq!(
            header_labels,
            vec![
                "DeepSeek",
                "Ollama (local)",
                "Ollama (cloud)",
                "Ollama (library — pullable)",
            ]
        );
    }

    #[test]
    fn cloud_section_omitted_when_no_cloud_models() {
        let app = create_test_app();
        let view = picker(&app, &["qwen3.6:35b"], &[], &["llama3.3"]);
        assert!(
            !view
                .rows
                .iter()
                .any(|r| r.kind == RowKind::Header && r.label == "Ollama (cloud)"),
            "cloud header must not render when cloud list is empty"
        );
    }

    #[test]
    fn library_dedupes_against_local_and_cloud() {
        let app = create_test_app();
        let view = picker(
            &app,
            &["llama3.3"],
            &["glm-5:cloud"],
            &["llama3.3", "glm-5:cloud", "qwen3", "phi4"],
        );
        let library_models: Vec<&str> = view
            .rows
            .iter()
            .filter(|r| r.kind == RowKind::OllamaLibrary)
            .map(|r| r.label.as_str())
            .collect();
        assert_eq!(library_models, vec!["qwen3", "phi4"]);
    }

    #[test]
    fn cloud_row_apply_emits_provider_switch_and_off_effort() {
        let app = create_test_app();
        let mut view = picker(&app, &[], &["glm-5:cloud"], &[]);
        let idx = view
            .rows
            .iter()
            .position(|r| r.kind == RowKind::OllamaCloud)
            .expect("cloud row exists");
        view.selected_row_idx = idx;
        match view.handle_key(KeyEvent::new(
            KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        )) {
            ViewAction::EmitAndClose(ViewEvent::ModelPickerApplied {
                model,
                effort,
                provider,
                ..
            }) => {
                assert_eq!(model, "glm-5:cloud");
                assert_eq!(effort, ReasoningEffort::Off);
                assert_eq!(provider, Some(ApiProvider::Ollama));
            }
            other => panic!("expected cloud apply, got {other:?}"),
        }
    }

    #[test]
    fn library_row_apply_emits_ollama_provider() {
        let app = create_test_app();
        let mut view = picker(&app, &[], &[], &["llama3.3"]);
        let idx = view
            .rows
            .iter()
            .position(|r| r.kind == RowKind::OllamaLibrary)
            .expect("library row exists");
        view.selected_row_idx = idx;
        match view.handle_key(KeyEvent::new(
            KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        )) {
            ViewAction::EmitAndClose(ViewEvent::ModelPickerApplied {
                model, provider, ..
            }) => {
                assert_eq!(model, "llama3.3");
                assert_eq!(provider, Some(ApiProvider::Ollama));
            }
            other => panic!("expected library apply, got {other:?}"),
        }
    }

    #[test]
    fn arrow_keys_skip_section_headers_across_all_sections() {
        let app = create_test_app();
        let mut view = picker(
            &app,
            &["qwen3.6:35b"],
            &["glm-5:cloud", "kimi:cloud"],
            &["llama3.3"],
        );
        for _ in 0..30 {
            view.handle_key(KeyEvent::new(
                KeyCode::Down,
                crossterm::event::KeyModifiers::NONE,
            ));
            assert!(
                view.selected_row().kind.is_selectable(),
                "navigation landed on a header row at idx {}",
                view.selected_row_idx
            );
        }
    }

    #[test]
    fn page_down_advances_by_visible_window() {
        let app = create_test_app();
        let library: Vec<String> = (0..50).map(|i| format!("model-{i:02}")).collect();
        let lib_refs: Vec<&str> = library.iter().map(|s| s.as_str()).collect();
        let mut view = picker(&app, &[], &[], &lib_refs);
        view.visible_rows.set(8);
        let before = view.selected_row_idx;
        view.handle_key(KeyEvent::new(
            KeyCode::PageDown,
            crossterm::event::KeyModifiers::NONE,
        ));
        assert!(
            view.selected_row_idx > before + 4,
            "PgDn should jump by visible_rows, went {} → {}",
            before,
            view.selected_row_idx
        );
    }

    #[test]
    fn end_jumps_to_last_selectable() {
        let app = create_test_app();
        let mut view = picker(&app, &["a"], &["b:cloud"], &["c"]);
        view.handle_key(KeyEvent::new(
            KeyCode::End,
            crossterm::event::KeyModifiers::NONE,
        ));
        let last = view
            .rows
            .iter()
            .rposition(|r| r.kind.is_selectable())
            .unwrap();
        assert_eq!(view.selected_row_idx, last);
    }

    #[test]
    fn ollama_unavailable_stub_is_inert_on_enter() {
        let app = create_test_app();
        let mut view = picker(&app, &[], &[], &[]);
        let stub_idx = view
            .rows
            .iter()
            .position(|r| r.kind == RowKind::OllamaUnavailable)
            .expect("stub row exists when all ollama lists are empty");
        view.selected_row_idx = stub_idx;
        let action = view.handle_key(KeyEvent::new(
            KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));
        assert!(matches!(action, ViewAction::None));
    }

    #[test]
    fn thinking_pane_disabled_for_all_ollama_kinds() {
        let app = create_test_app();
        let mut view = picker(
            &app,
            &["local-1"],
            &["cloud-1:cloud"],
            &["library-1"],
        );
        for kind in [RowKind::OllamaLocal, RowKind::OllamaCloud, RowKind::OllamaLibrary] {
            let idx = view.rows.iter().position(|r| r.kind == kind).unwrap();
            view.selected_row_idx = idx;
            assert!(
                view.thinking_pane_disabled(),
                "thinking pane should be disabled for {kind:?}"
            );
            assert_eq!(view.resolved_effort(), ReasoningEffort::Off);
        }
    }

    #[test]
    fn deepseek_row_apply_emits_no_provider_switch_when_already_deepseek() {
        let app = create_test_app();
        let mut view = picker(&app, &[], &[], &[]);
        let idx = view
            .rows
            .iter()
            .position(|r| r.kind == RowKind::Deepseek && r.label == "deepseek-v4-flash")
            .unwrap();
        view.selected_row_idx = idx;
        match view.handle_key(KeyEvent::new(
            KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        )) {
            ViewAction::EmitAndClose(ViewEvent::ModelPickerApplied { provider, .. }) => {
                assert_eq!(provider, None);
            }
            other => panic!("expected DeepSeek apply, got {other:?}"),
        }
    }

    #[test]
    fn esc_closes_without_emitting() {
        let app = create_test_app();
        let mut view = picker(&app, &[], &[], &[]);
        let action = view.handle_key(KeyEvent::new(
            KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        ));
        assert!(matches!(action, ViewAction::Close));
    }

    #[test]
    fn picker_only_exposes_auto_off_high_max() {
        let labels: Vec<&str> = PICKER_EFFORTS
            .iter()
            .map(|effort| effort.short_label())
            .collect();
        assert_eq!(labels, vec!["auto", "off", "high", "max"]);
    }

    #[test]
    fn discover_user_cloud_models_filters_garbage_labels() {
        let mut set = BTreeSet::new();
        insert_if_modelish(&mut set, "Ollama Cloud");
        insert_if_modelish(&mut set, "");
        insert_if_modelish(&mut set, "  ");
        insert_if_modelish(&mut set, "glm-5:cloud");
        insert_if_modelish(&mut set, " kimi-k2.6:cloud ");
        let collected: Vec<String> = set.into_iter().collect();
        assert_eq!(collected, vec!["glm-5:cloud", "kimi-k2.6:cloud"]);
    }
}
