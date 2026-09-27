// workaround for @microsoft/api-extractor not yet supporting ES2025
type Float16Array = typeof globalThis extends {
  Float16Array: { prototype: infer TFloat16ArrayPrototype };
}
  ? TFloat16ArrayPrototype
  : never;

/** Union of all natively supported signed integer typed arrays */
export type IntArray = Int8Array | Int16Array | Int32Array;

/** Union of all natively supported unsigned integer typed arrays */
export type UintArray =
  Uint8Array | Uint8ClampedArray | Uint16Array | Uint32Array;

/** Union of all natively supported integer typed arrays */
export type IntOrUintArray = IntArray | UintArray;

/** Union of all natively supported floating-point typed arrays */
export type FloatArray = Float16Array | Float32Array | Float64Array;

/** Union of all natively supported numeric typed arrays */
export type TypedArray = IntOrUintArray | FloatArray;

/**
 * The array types that can hold ID values
 *
 * Integer IDs are held by an integer typed array or by 64-bit floats (safe
 * integers only), string IDs by a plain array.
 */
export type IDArray = IntOrUintArray | Float64Array | string[];

/**
 * The array types that can hold values of type `T`
 *
 * Numbers are held by numeric typed arrays only, anything else by a plain
 * array. For an unconstrained `T`, both are possible.
 */
export type TypedArrayOrArray<T> = [T] extends [number]
  ? TypedArray
  : unknown extends T
    ? TypedArray | T[]
    : T[];
