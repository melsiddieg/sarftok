# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Interactive Arud Explorer is a React-based web application that visualizes Arabic poetry meters (Arud) based on Al-Khalil ibn Ahmad al-Farahidi's classical system. The app displays an interactive circular representation where users can rotate through different poetic meters to explore their patterns and structures.

## Development Commands

**Local Development:**
- `npm install` - Install dependencies
- `npm run dev` - Start development server (Vite)
- `npm run build` - Build for production
- `npm run preview` - Preview production build

**Prerequisites:** Node.js is required. Set `GEMINI_API_KEY` in `.env.local` (though the current codebase doesn't appear to actively use Gemini API).

## Design Specifications

### Theme & Visual Design
- **Color Scheme**: Deep gray/black background (#111827) with amber/gold accents (#FBBF24, #F59E0B)
- **Typography**: 
  - Arabic text: 'Amiri' Google Font (classic serif feel)
  - English UI text: 'Inter' font (modern clarity)
- **Directionality**: Entire application rendered in RTL mode (dir="rtl" on <html> tag)
- **Layout**: High-contrast, elegant dark theme design

### Critical RTL Navigation Requirements
- **Next Button**: Must be on the LEFT side with LEFT-pointing chevron (moves sequence forward)
- **Previous Button**: Must be on the RIGHT side with RIGHT-pointing chevron
- This ensures intuitive navigation for RTL interface users

## Architecture & Core Concepts

### Arud Theory Implementation
The application is built around Al-Khalil's circular theory of Arabic prosody:

- **ATOMIC_SEQUENCE**: A repeating 10-unit pattern `['//0', '/0', '//0', '/0', '/0', '//0', '/0', '//0', '/0', '/0']` representing the fundamental syllabic units
  - `'//0'`: watid majmū' (long-short-long)
  - `'/0'`: sabab khafif (short-long)
- **Poetic Feet (Tafila)**: Groups of atomic units that form complete metrical patterns
- **Meters**: Different starting positions and parsing instructions that create distinct poetic meters

### Data Model Requirements
- **Meter Objects**: Each must contain:
  - `name`: Arabic name (e.g., 'البحر الطويل')
  - `description`: Brief English description
  - `startOffset`: Starting index on ATOMIC_SEQUENCE
  - `parsingInstructions`: Array dictating atomic unit grouping (e.g., [2, 3, 2, 3])
  - `patternTransliteration`: English transliteration of full pattern
- **Accuracy**: All meters must be in complete, theoretical forms (not shortened majzū' versions)

### Key Data Structures
- `Tafila`: Represents a single poetic foot with both unmerged and merged forms
- `Meter`: Defines a complete meter with ID, name, description, start offset, parsing instructions, and transliteration
- `TAFILA_MAP`: Maps atomic unit sequences to their Arabic names (e.g., '//0,/0' → 'فعولن')

### Component Architecture
- **App.tsx**: Main component managing meter navigation state
- **ArudCircle.tsx**: The critical "roulette" component - implements smooth sliding animation
- **MeterDisplay.tsx**: Shows meter details with fade-in animations
- **Controls.tsx**: RTL-compliant navigation controls
- **constants.ts**: Contains the atomic sequence, tafila mappings, and meter definitions
- **types.ts**: TypeScript interfaces for core data structures

### Interactive Banner ("Roulette") Specifications
**Critical Animation Requirements:**
- Render a continuous strip of ATOMIC_SEQUENCE (repeated multiple times)
- Smooth sliding animation right-to-left for "Next" navigation
- NO fading or abrupt re-rendering - only sliding motion
- Creates illusion of moving viewing frame along infinite circular tape

**Visual Hierarchy:**
- **Top Layer**: Raw atomic units (//0, /0) with amber highlighting on grouped units
- **Bottom Layer**: Resulting tafila names (e.g., فعولن)
- **Connector**: Upward-pointing curly brace SVG between layers (no "hats" above)
- **Animation**: Tafila text fades in with delay after atomic group slides into place

### Parsing Algorithm
The `parseMeterPattern()` function in constants.ts:71 is central to the app. It takes a meter's start offset and parsing instructions to extract the correct atomic units from the circular sequence and map them to their corresponding tafila.

### Styling Approach
- Uses Tailwind CSS for styling
- Arabic text uses `font-amiri` class (Amiri Google Font)
- English text uses `font-inter` class (Inter font)
- Custom CSS animations for fade-in effects
- RTL (right-to-left) text direction support with `dir="rtl"`
- Amber color scheme (`text-amber-400`, `bg-amber-500/10`) for highlighting
- Deep gray background (#111827) for dark theme

## Key Files for Modifications

**Adding New Meters:**
- Modify `constants.ts:37` - Add new meter objects to `METERS` array
- Ensure proper `startOffset` and `parsingInstructions` values
- Include complete theoretical forms, not shortened versions

**Adding New Tafila:**
- Update `TAFILA_MAP` in `constants.ts:9` with new atomic unit combinations

**UI Changes:**
- `ArudCircle.tsx` for the main "roulette" visualization (critical component)
- `MeterDisplay.tsx` for meter information display with animations
- `Controls.tsx` for RTL-compliant navigation
- Tailwind classes are used throughout for responsive design

## Technical Notes

- Built with Vite + React 19 + TypeScript
- No external state management (uses React's built-in useState)
- Uses SVG for the curved brace graphics (upward-pointing between layers)
- Supports both Arabic and English text with proper RTL handling
- The `unitWidth` constant (90px) in ArudCircle.tsx:12 controls the visual spacing of atomic units
- Animation timing and easing critical for smooth "roulette" effect