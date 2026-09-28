/** Formats a fraction as a whole percentage, such as 0.5 as "50%" */
export const percentFormat = new Intl.NumberFormat(undefined, {
  style: "percent",
  maximumFractionDigits: 0,
});
