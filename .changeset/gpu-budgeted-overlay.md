---
"@tissuumaps/render": minor
"@tissuumaps/react": patch
---

Pace the WebGL overlay redraws by the GPU, so that panning stays smooth: the overlay takes at most half of the GPU, and between redraws a viewport change only moves the drawn pixels with a CSS transform. Adds `WebGLFrameScheduler` to `@tissuumaps/render`; `useWebGL` hands all redraws to it.
