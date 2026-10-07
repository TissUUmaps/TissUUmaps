import { createAjv } from "@jsonforms/core";

/**
 * The Ajv instance data sources are validated with, configured like the one
 * JSON Forms uses; shared, so that it compiles each schema only once
 */
export const ajv: ReturnType<typeof createAjv> = createAjv();
