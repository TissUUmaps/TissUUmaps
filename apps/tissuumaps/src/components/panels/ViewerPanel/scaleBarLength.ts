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
 * Rounds a scale bar length down to 1, 2 or 5 times a power of ten
 *
 * @param maxLength - The longest allowed length, which must be positive
 * @returns The rounded length, in the unit of `maxLength`
 */
export function getScaleBarLength(maxLength: number): number {
  const power = 10 ** Math.floor(Math.log10(maxLength));
  const significand = maxLength / power;
  return (significand >= 5 ? 5 : significand >= 2 ? 2 : 1) * power;
}

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
