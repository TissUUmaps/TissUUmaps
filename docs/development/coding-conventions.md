---
sidebar_position: 3
---

# Coding conventions

## Principles

TissUUmaps is research software that is meant to be maintained over many years, by many people. When writing code, we therefore value:

- **Correctness**, scientific and otherwise: results must be right, and the code must make it easy to verify that they are.
- **Readability and simplicity**: code is read far more often than it is written; prefer the plain, obvious solution.
- **Consistency**: follow the patterns already established in the code base, even if you would personally have chosen differently.
- **Transparency**: make behavior explicit, in particular around error handling; fail loudly rather than silently swallowing or masking problems.
- **Resource efficiency**: the application handles large data sets in the browser; be deliberate about memory, network traffic and rendering cost.
- **Maintainability and clean architecture**: respect module boundaries and keep responsibilities well separated (see [Code architecture](./code-architecture.md)).

Conversely, we do not like shortcuts and "quick solutions", premature optimization, and unnecessary over-engineering. Abstractions must earn their place: new named types, generics, conditional types and similar constructs should only be introduced when there is a real need for them. If a single feature ships several of these without such a reason, it is over-engineered and should be simplified.

## Modules

- Exports are named; there are no default exports anywhere.
- Functions are declared as named function declarations (`export function myFunction(...)`), not as arrow functions assigned to constants.
- Methods that implement or replace a base class member are marked `override`.
- Enum-like values are declared as a `const` object with a same-name type derived from it, never as a TypeScript `enum`:

  ```ts
  export const Shape = { Circle: 0, Square: 1, Triangle: 2 } as const;
  export type Shape = (typeof Shape)[keyof typeof Shape];
  ```

- Constants are camelCase module-level `const` declarations with a TSDoc comment, not `SCREAMING_CASE`.
- Generic type parameters are prefixed with `T` (`TData`, `TValue`), not single letters beyond the trivial `T`.

## Classes

- There is primarily one class per file, and the file is named after the class (`class MyClass` in `MyClass.ts`).
- Abstract base classes are suffixed `Base` (`MyClassBase`); classes bundling static helper functions are suffixed `Utils` (`MyUtils`).

## Functions

- Functions that start a long-lived process are named `startX()` and return a teardown callback that undoes everything they set up.
- Functions that call `await` must be declared `async`. Functions that merely return a `Promise` without awaiting anything must not be declared `async`, so that they can throw synchronously or return early synchronously. Return `Promise.resolve(...)`/`Promise.reject(...)` where necessary in those cases.
- Every function executing asynchronous code takes an `options?: { signal?: AbortSignal }` parameter. Exceptions are permitted when passing asynchronous functions to third-party APIs (e.g. asynchronous event handlers) and in functions solely responsible for object destruction/cleanup. Other deviations should be documented in a code comment.
- Functions taking an `AbortSignal` call `signal?.throwIfAborted()` right after destructuring the signal from `options` ("early pre-check"). Additional `throwIfAborted()` calls should only be made after awaiting asynchronous functions that do not receive the signal (e.g. third-party APIs), in which case this should be documented in a code comment.

```ts
async function load(
  id: string,
  options?: { signal?: AbortSignal },
): Promise<Result> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const data = await fetchData(id, { signal });
  return parseData(data);
}
```

## Comments

We are old-school and prefer readable and well-organized code over excessive inline comments. Variable and function names, types, and architecture should be self-explanatory; `//` comments should only be placed where essential. Non-obvious code gets a `//` comment explaining _why_ it is written the way it is (not what it does), ideally linking to the issue, specification or upstream bug that motivates it. Deliberate deviations from the conventions on this page are documented the same way.

Every export carries a TSDoc comment, from which API documentation is generated (see [Development workflow](./development-workflow.md#documentation)). `@param`, `@returns` and `@throws` are mandatory where applicable; `@typeParam` and `@defaultValue` should be used where they add information.

## Error handling

- Errors are thrown as plain `Error` instances (`throw new Error("...")`), optionally with a `cause`. There are no custom error classes and no `Result` types.
- Code paths that must not fail (e.g. plugin setup, event handlers) catch, report via `console.error`, and continue.
- Only `console.error` and `console.warn` are used, for errors and warnings respectively. Any `console.log` is a finding.

## React components

- Components are named function declarations taking a destructured props object: `export function MyComponent({ className }: MyComponentProps)`.
- The props type is always named `<ComponentName>Props` and declared with `type`, not `interface`.
- Callback props are named `onX` for events (`onClose`) and `onXChange` for controlled values (`onValueChange`).
- Every component that renders an element of its own accepts `className?: string`. In the application, it is merged into the root element via `cn()`; components in publishable packages are styling-agnostic and forward it verbatim.
- In the application, styling uses Tailwind (v4) utility classes and `cva` variants; there are no CSS modules. Inline styles are limited to values that cannot be expressed as classes (e.g. measured pixel sizes, CSS custom properties).
- Icons come from `lucide-react`.
- There is no internationalization; user-facing strings are inline English literals.

## Tests

- Tests are colocated with the unit they test, as `X.test.ts` next to `X.ts`.
- Pure utilities, resolvers, caches, workers, data providers and similar logic-bearing units are expected to have tests.
- GUI components, stores and similar glue code do not require tests.
