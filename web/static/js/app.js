/**
 * LangRAG Web Console - Main Entry Point
 *
 * This file loads all modules in the correct order.
 * Using script tags instead of ES modules for simplicity (no build step).
 */

// Modules are loaded via script tags in index.html in this order:
// 1. core/api.js     - API client
// 2. core/utils.js   - Toast and utilities
// 3. core/stores.js  - Alpine.js stores
// 4. pages/kb-list.js
// 5. pages/kb-detail.js
// 6. pages/chat.js
// 7. pages/models.js
// 8. Alpine.js (via CDN)

console.log('LangRAG Web Console loaded');
