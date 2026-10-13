/** Units of a scale bar length, with their size in micrometers, small first */
const lengthUnits = [
  { symbol: "nm", size: 1e-3 },
  { symbol: "µm", size: 1 },
  { symbol: "mm", size: 1e3 },
  { symbol: "m", size: 1e6 },
] as const;

/** Formats the value of a scale bar length, hiding floating-point noise */
const lengthValueFormat = new Intl.NumberFormat(undefined, {
  maximumSignificantDigits: 3,
});

/**
 * Formats a scale bar length in pixels, or in micrometers converted to the
 * largest unit that keeps its value at least 1, from nanometers to meters
 *
 * @param length - The length, in `lengthUnit`
 * @param lengthUnit - The unit of `length`: micrometers (µm) or pixels (px)
 * @returns The formatted length, e.g. "500 nm", "2 mm" or "50 px"
 */
export function formatScaleBarLength(
  length: number,
  lengthUnit: "µm" | "px",
): string {
  if (lengthUnit === "px") {
    return `${lengthValueFormat.format(length)} px`;
  }
  const unit =
    lengthUnits.findLast((unit) => unit.size <= length) ?? lengthUnits[0];
  return `${lengthValueFormat.format(length / unit.size)} ${unit.symbol}`;
}
