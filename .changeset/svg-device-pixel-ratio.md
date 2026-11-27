---
"@tissuumaps/render": minor
"@tissuumaps/react": patch
---

Shapes drawn in the viewer end up where they were drawn, also when `devicePixelRatio` is not 1 (#323). `SVGController` maps world to screen through the `<svg>` `viewBox`; `resizeContainer` and `transformNode` are removed.
