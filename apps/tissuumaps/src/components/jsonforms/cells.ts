import {
  and,
  isBooleanControl,
  isEnumControl,
  isIntegerControl,
  isNumberControl,
  isStringControl,
  rankWith,
  scopeEndIs,
} from "@jsonforms/core";

import { BooleanCell } from "./cells/BooleanCell";
import { IntegerCell } from "./cells/IntegerCell";
import { NumberCell } from "./cells/NumberCell";
import { StringEnumCell } from "./cells/StringEnumCell";
import { TableEnumCell } from "./cells/TableEnumCell";
import { TextCell } from "./cells/TextCell";

export const cells = [
  { tester: rankWith(1, isBooleanControl), cell: BooleanCell },
  { tester: rankWith(1, isIntegerControl), cell: IntegerCell },
  { tester: rankWith(1, isNumberControl), cell: NumberCell },
  { tester: rankWith(1, isStringControl), cell: TextCell },
  {
    tester: rankWith(2, and(isStringControl, isEnumControl)),
    cell: StringEnumCell,
  },
  {
    tester: rankWith(3, and(isStringControl, scopeEndIs("table"))),
    cell: TableEnumCell,
  },
];
