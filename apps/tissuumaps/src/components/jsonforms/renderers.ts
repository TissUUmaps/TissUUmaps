import {
  and,
  isControl,
  isPrimitiveArrayControl,
  or,
  rankWith,
  uiTypeIs,
} from "@jsonforms/core";

import { ArrayControl } from "./controls/ArrayControl";
import { InputControl } from "./controls/InputControl";
import { LabelControl } from "./controls/LabelControl";
import { ListArrayControl } from "./controls/ListArrayControl";
import { ListControl } from "./controls/ListControl";
import { GroupLayout } from "./layouts/GroupLayout";
import { HorizontalLayout } from "./layouts/HorizontalLayout";
import { ListLayout } from "./layouts/ListLayout";
import { VerticalLayout } from "./layouts/VerticalLayout";

export const renderers = [
  { tester: rankWith(1, uiTypeIs("Group")), renderer: GroupLayout },
  { tester: rankWith(1, uiTypeIs("VerticalLayout")), renderer: VerticalLayout },
  {
    tester: rankWith(1, uiTypeIs("HorizontalLayout")),
    renderer: HorizontalLayout,
  },
  { tester: rankWith(1, uiTypeIs("Label")), renderer: LabelControl },
  { tester: rankWith(1, isControl), renderer: InputControl },
  {
    tester: rankWith(2, and(isControl, isPrimitiveArrayControl)),
    renderer: ArrayControl,
  },
];

/**
 * Renderers that show a form's fields as rows of a two-column list, without
 * labels, for a read-only form inside a `<dl>`
 */
export const listRenderers = [
  {
    tester: rankWith(
      1,
      or(
        uiTypeIs("Group"),
        uiTypeIs("VerticalLayout"),
        uiTypeIs("HorizontalLayout"),
      ),
    ),
    renderer: ListLayout,
  },
  { tester: rankWith(1, uiTypeIs("Label")), renderer: () => null },
  { tester: rankWith(1, isControl), renderer: ListControl },
  {
    tester: rankWith(2, and(isControl, isPrimitiveArrayControl)),
    renderer: ListArrayControl,
  },
];
