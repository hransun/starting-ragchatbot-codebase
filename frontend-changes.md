# Frontend Changes: Dark/Light Theme Toggle

## Overview
Added a toggle button that allows users to switch between dark and light themes with smooth transitions and localStorage persistence.

## Files Modified

### 1. `frontend/index.html`
- Added a theme toggle button inside the container with sun/moon SVG icons
- Button is positioned in the top-right corner
- Includes proper accessibility attributes (`aria-label`, `title`)

### 2. `frontend/style.css`

#### New CSS Variables (`:root`)
Added new CSS variables for scrollbar and code block theming:
- `--code-bg`: Background color for code blocks
- `--scrollbar-track`: Scrollbar track color
- `--scrollbar-thumb`: Scrollbar thumb color
- `--scrollbar-thumb-hover`: Scrollbar thumb hover color

#### Light Theme (`[data-theme="light"]`)
Complete light theme color scheme:
- `--background: #f8fafc` - Light gray background
- `--surface: #ffffff` - White surface color
- `--surface-hover: #f1f5f9` - Light hover state
- `--text-primary: #1e293b` - Dark text for contrast
- `--text-secondary: #64748b` - Muted secondary text
- `--border-color: #e2e8f0` - Subtle borders
- `--assistant-message: #f1f5f9` - Light assistant message background
- `--shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1)` - Softer shadows
- `--welcome-bg: #eff6ff` - Light blue welcome background
- `--code-bg: rgba(0, 0, 0, 0.05)` - Subtle code background
- Scrollbar colors adjusted for light theme

#### Theme Transition
Added smooth transitions for theme switching on:
- `body`, `.container`, `.sidebar`, `.chat-main`
- `.chat-container`, `.chat-messages`, `.message-content`
- `.chat-input-container`, `#chatInput`
- `.stat-item`, `.suggested-item`
- `.new-chat-button`, `.theme-toggle`

#### Theme Toggle Button Styles
- `.theme-toggle`: Fixed position (top-right), circular button with shadow
- Hover effects: scale, color change, border highlight
- Focus state with focus ring for accessibility
- Active state with slight scale down
- Sun/moon icon visibility based on current theme

#### Updated Existing Styles
- Code blocks now use `var(--code-bg)` instead of hardcoded colors
- Scrollbars use CSS variables for consistent theming
- Container has `position: relative` for toggle button positioning

### 3. `frontend/script.js`

#### New Theme Management Functions
```javascript
function initTheme()    // Loads saved theme or defaults to dark
function applyTheme()   // Sets data-theme attribute and saves to localStorage
function toggleTheme()  // Switches between dark and light themes
```

#### Changes
- Added `themeToggle` to DOM element references
- Added event listener for theme toggle button click
- Theme initializes before DOMContentLoaded to prevent flash of wrong theme
- Theme preference persists across sessions via localStorage

## How It Works

1. **Initial Load**: `initTheme()` runs immediately (before DOM loads) to check localStorage for saved theme preference. Defaults to dark theme if none saved.

2. **Toggle Action**: Clicking the toggle button calls `toggleTheme()` which:
   - Gets current theme from `data-theme` attribute
   - Switches to opposite theme
   - Updates `data-theme` on `<html>` element
   - Saves preference to localStorage

3. **Visual Feedback**:
   - Sun icon shows in dark mode (click to switch to light)
   - Moon icon shows in light mode (click to switch to dark)
   - Smooth 0.3s transitions on all themed elements

## Accessibility Features
- Keyboard navigable (focusable button)
- `aria-label` for screen readers
- `title` attribute for tooltip
- Visible focus ring
- Sufficient color contrast in both themes
