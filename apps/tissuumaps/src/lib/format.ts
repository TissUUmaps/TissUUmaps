/** Formats a fraction as a whole percentage, such as 0.5 as "50%" */
export const percentFormat = new Intl.NumberFormat(undefined, {
  style: "percent",
  maximumFractionDigits: 0,
});

/** Formats a number with up to six significant digits, such as 0.1234567 as "0.123457" */
export const numberFormat = new Intl.NumberFormat(undefined, {
  maximumSignificantDigits: 6,
});
