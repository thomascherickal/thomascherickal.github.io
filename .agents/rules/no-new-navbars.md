# Rule: No New Navbars for New Content

- **Strict Ban on Secondary Navbars**: Do not create or add new `<nav>` elements, sub-navigation bars, in-page guide strips, or topic ribbons to new content or pages.
- **Centralized Navigation Only**: All site navigation must rely exclusively on the global navigation header defined in `header.html` and inlined by `python3 python/build_components.py`.
- **Linking New Content**: Links to new pages or content must be added to the appropriate dropdown menus in `header.html`, or linked contextually via cards, buttons, or in-text hyperlinks within the page body.
