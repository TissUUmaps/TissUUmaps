import { LengthUnit } from "@tissuumaps/core";

/** Symbol and size, in meters, of every length unit, small first */
export const lengthUnits: Record<LengthUnit, { symbol: string; size: number }> =
  {
    [LengthUnit.nanometer]: { symbol: "nm", size: 1e-9 },
    [LengthUnit.micrometer]: { symbol: "µm", size: 1e-6 },
    [LengthUnit.millimeter]: { symbol: "mm", size: 1e-3 },
    [LengthUnit.centimeter]: { symbol: "cm", size: 1e-2 },
    [LengthUnit.meter]: { symbol: "m", size: 1 },
    [LengthUnit.kilometer]: { symbol: "km", size: 1e3 },
  };
