---
type: module
title: "Global Stylesheets & Design System"
description: "Modular CSS stylesheets defining color tokens, typography, radiant buttons, high-contrast dark theme, and responsive layouts."
tags: [css, design-system, tokens, responsive, buttons]
timestamp: 2026-11-10T08:43:02Z
---

# Global Stylesheets & Design System

Modular CSS stylesheets defining color tokens, typography, radiant buttons, high-contrast dark theme, and responsive layouts.

## Directory Role & Agent Guidelines

- `buttons.css` defines canonical button formats (`.btn`, `.btn-commission`, `.btn-linkedin`, `.btn-gold`, `.btn-lg`). Buttons must never have underlined text.
- Hyperlinks default to gold (`var(--gold, #fbbf24)`) transitioning to gold-dim (`#d97706`) on hover.
- Code windows must have sharp dark boundaries (`#040711` or `#060913`) and crisp borders.
- Responsive design relies on media queries ensuring zero horizontal scroll blowout across mobile viewports.

## Subdirectories

None (Leaf directory).

## File Inventory

- [`article.css`](article.css)
- [`background.css`](background.css)
- [`business-services.css`](business-services.css)
- [`buttons.css`](buttons.css)
- [`claude.css`](claude.css)
- [`cta.css`](cta.css)
- [`faq.css`](faq.css)
- [`fonts.css`](fonts.css)
- [`footer.css`](footer.css)
- [`gemini.css`](gemini.css)
- [`navbar.css`](navbar.css)
- [`personal-services.css`](personal-services.css)
- [`pricing.css`](pricing.css)
- [`responsive.css`](responsive.css)
- [`service.css`](service.css)
- [`styles.css`](styles.css)

## Context & Cross-References

- Root Directory: [`/OKF.md`](../OKF.md)
- Architectural Specification: [`GEMINI.md`](/GEMINI.md)
- Repository Telemetry: [`METRICS.md`](/METRICS.md)
