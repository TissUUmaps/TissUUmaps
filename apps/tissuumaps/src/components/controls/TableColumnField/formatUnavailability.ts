/**
 * Why a column cannot be read, in the words of the user interface, by the
 * reasons the built-in data providers give
 */
const unavailabilityMessages = new Map([
  ["row-major-sparse", "stored by rows; cannot be read by column"],
  ["length-mismatch", "not one value per row"],
  ["unsupported-type", "data type not supported"],
]);

/**
 * Formats why a column cannot be read
 *
 * @param unavailability - The reason the table data gives, in its own words
 * @returns The reason to show; a generic one for reasons the user interface
 * does not know
 */
export function formatUnavailability(unavailability: string): string {
  return unavailabilityMessages.get(unavailability) ?? "Can't be read";
}
