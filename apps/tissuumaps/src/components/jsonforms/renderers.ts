import {
  and,
  isControl,
  isPrimitiveArrayControl,
  rankWith,
  uiTypeIs,
} from "@jsonforms/core";

import { ArrayControl } from "./controls/ArrayControl";
import { InputControl } from "./controls/InputControl";
import { LabelControl } from "./controls/LabelControl";
import { GroupLayout } from "./layouts/GroupLayout";
import { HorizontalLayout } from "./layouts/HorizontalLayout";
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
