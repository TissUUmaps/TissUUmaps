import { lengthUnits } from "@/lib/lengthUnits";

/** Formats the value of a scale bar length, hiding floating-point noise */
const lengthValueFormat = new Intl.NumberFormat(undefined, {
  maximumSignificantDigits: 3,
});

/**
 * Formats a scale bar length in pixels, or in meters converted to the
 * largest unit that keeps its value at least 1, from nanometers to kilometers
 *
 * @param length - The length, in `lengthUnit`
 * @param lengthUnit - The unit of `length`: meters (m) or pixels (px)
 * @returns The formatted length, e.g. "500 nm", "2 km" or "50 px"
 */
export function formatScaleBarLength(
  length: number,
  lengthUnit: "m" | "px",
): string {
  if (lengthUnit === "px") {
    return `${lengthValueFormat.format(length)} px`;
  }
  const unit =
    Object.values(lengthUnits).findLast((unit) => unit.size <= length) ??
    lengthUnits.nanometer;
  return `${lengthValueFormat.format(length / unit.size)} ${unit.symbol}`;
}
