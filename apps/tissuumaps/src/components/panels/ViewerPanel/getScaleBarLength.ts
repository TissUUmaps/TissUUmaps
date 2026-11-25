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
