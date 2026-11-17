import { lengthUnits } from "@/lib/lengthUnits";

/** Formats the value of a scale bar length, hiding floating-point noise */
const lengthValueFormat = new Intl.NumberFormat(undefined, {
  maximumSignificantDigits: 3,
});

/**
 * Formats a scale bar length in world units, or in meters converted to the
 * largest unit that keeps its value at least 1, from picometers to kilometers
 *
 * @param length - The length, in `unit`
 * @param unit - The unit of `length`: meters (m) or world units (units)
 * @returns The formatted length, e.g. "500 nm", "2 km" or "50 units"
 */
export function formatScaleBarLength(
  length: number,
  unit: "m" | "units",
): string {
  if (unit === "units") {
    return `${lengthValueFormat.format(length)} units`;
  }
  const lengthUnit =
    Object.values(lengthUnits).findLast(
      (lengthUnit) => lengthUnit.size <= length,
    ) ?? lengthUnits.picometer;
  return `${lengthValueFormat.format(length / lengthUnit.size)} ${lengthUnit.symbol}`;
}
