import { PhysicalUnit, physicalUnits } from "@tissuumaps/core";

/** Formats the value of a scalebar length, hiding floating-point noise */
const lengthValueFormat = new Intl.NumberFormat(undefined, {
  maximumSignificantDigits: 3,
});

/**
 * Formats a scalebar length, converting meters to the largest physical unit
 * that keeps the value at least 1, from picometers to kilometers
 *
 * @param length - The length, in `unit`
 * @param unit - The unit of `length`; any unit other than meters is shown as is
 * @returns The formatted length, e.g. "500 nm", "2 km" or "50 units"
 */
export function formatScalebarLength(length: number, unit: string): string {
  if (unit !== PhysicalUnit.meter) {
    return `${lengthValueFormat.format(length)} ${unit}`;
  }
  const physicalUnit =
    Object.values(physicalUnits).findLast(
      (physicalUnit) => physicalUnit.size <= length,
    ) ?? physicalUnits.picometer;
  return `${lengthValueFormat.format(length / physicalUnit.size)} ${physicalUnit.symbol}`;
}
