---
"@tissuumaps/render": minor
"@tissuumaps/react": patch
---

Points, shapes, labels and images are each drawn as soon as their own data is ready, so a slow or failing object no longer holds back the other objects of its kind (#269). The WebGL renderers' `synchronize` takes an `onChange` callback, which is called whenever a rendered object has been added, updated or dropped.
