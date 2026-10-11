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
 * Formats a scale bar length in the largest unit that keeps its value at
 * least 1, from nanometers to meters
 *
 * @param length - The length, in micrometers (µm)
 * @returns The formatted length, e.g. "500 nm" or "2 mm"
 */
export function formatScaleBarLength(length: number): string {
  const unit =
    lengthUnits.findLast((unit) => unit.size <= length) ?? lengthUnits[0];
  return `${lengthValueFormat.format(length / unit.size)} ${unit.symbol}`;
}
