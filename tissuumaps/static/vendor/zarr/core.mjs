const registry = new Map();
function addCodec(id, importFn) {
    registry.set(id, importFn);
}
async function getCodec(config) {
    if (!registry.has(config.id)) {
        throw new Error(`Compression codec ${config.id} is not supported by Zarr.js yet.`);
    }
    /* eslint-disable @typescript-eslint/no-non-null-assertion */
    const codec = await registry.get(config.id)();
    return codec.fromConfig(config);
}

function createProxy(mapping) {
    return new Proxy(mapping, {
        set(target, key, value, _receiver) {
            return target.setItem(key, value);
        },
        get(target, key, _receiver) {
            return target.getItem(key);
        },
        deleteProperty(target, key) {
            return target.deleteItem(key);
        },
        has(target, key) {
            return target.containsItem(key);
        }
    });
}

function isZarrError(err) {
    return typeof err === 'object' && err !== null && '__zarr__' in err;
}
function isKeyError(o) {
    return isZarrError(o) && o.__zarr__ === 'KeyError';
}
// Custom error messages, note we have to patch the prototype of the
// errors to fix `instanceof` calls, see:
// https://github.com/Microsoft/TypeScript/wiki/Breaking-Changes#extending-built-ins-like-error-array-and-map-may-no-longer-work
class ContainsArrayError extends Error {
    constructor(path) {
        super(`path ${path} contains an array`);
        this.__zarr__ = 'ContainsArrayError';
        Object.setPrototypeOf(this, ContainsArrayError.prototype);
    }
}
class ContainsGroupError extends Error {
    constructor(path) {
        super(`path ${path} contains a group`);
        this.__zarr__ = 'ContainsGroupError';
        Object.setPrototypeOf(this, ContainsGroupError.prototype);
    }
}
class ArrayNotFoundError extends Error {
    constructor(path) {
        super(`array not found at path ${path}`);
        this.__zarr__ = 'ArrayNotFoundError';
        Object.setPrototypeOf(this, ArrayNotFoundError.prototype);
    }
}
class GroupNotFoundError extends Error {
    constructor(path) {
        super(`ground not found at path ${path}`);
        this.__zarr__ = 'GroupNotFoundError';
        Object.setPrototypeOf(this, GroupNotFoundError.prototype);
    }
}
class PathNotFoundError extends Error {
    constructor(path) {
        super(`nothing not found at path ${path}`);
        this.__zarr__ = 'PathNotFoundError';
        Object.setPrototypeOf(this, PathNotFoundError.prototype);
    }
}
class PermissionError extends Error {
    constructor(message) {
        super(message);
        this.__zarr__ = 'PermissionError';
        Object.setPrototypeOf(this, PermissionError.prototype);
    }
}
class KeyError extends Error {
    constructor(key) {
        super(`key ${key} not present`);
        this.__zarr__ = 'KeyError';
        Object.setPrototypeOf(this, KeyError.prototype);
    }
}
class TooManyIndicesError extends RangeError {
    constructor(selection, shape) {
        super(`too many indices for array; expected ${shape.length}, got ${selection.length}`);
        this.__zarr__ = 'TooManyIndicesError';
        Object.setPrototypeOf(this, TooManyIndicesError.prototype);
    }
}
class BoundsCheckError extends RangeError {
    constructor(message) {
        super(message);
        this.__zarr__ = 'BoundsCheckError';
        Object.setPrototypeOf(this, BoundsCheckError.prototype);
    }
}
class InvalidSliceError extends RangeError {
    constructor(from, to, stepSize, reason) {
        super(`slice arguments slice(${from}, ${to}, ${stepSize}) invalid: ${reason}`);
        this.__zarr__ = 'InvalidSliceError';
        Object.setPrototypeOf(this, InvalidSliceError.prototype);
    }
}
class NegativeStepError extends Error {
    constructor() {
        super(`Negative step size is not supported when indexing.`);
        this.__zarr__ = 'NegativeStepError';
        Object.setPrototypeOf(this, NegativeStepError.prototype);
    }
}
class ValueError extends Error {
    constructor(message) {
        super(message);
        this.__zarr__ = 'ValueError';
        Object.setPrototypeOf(this, ValueError.prototype);
    }
}
class HTTPError extends Error {
    constructor(code) {
        super(code);
        this.__zarr__ = 'HTTPError';
        Object.setPrototypeOf(this, HTTPError.prototype);
    }
}

function slice(start, stop = undefined, step = null) {
    // tslint:disable-next-line: strict-type-predicates
    if (start === undefined) { // Not possible in typescript
        throw new InvalidSliceError(start, stop, step, "The first argument must not be undefined");
    }
    if ((typeof start === "string" && start !== ":") || (typeof stop === "string" && stop !== ":")) { // Note in typescript this will never happen with type checking.
        throw new InvalidSliceError(start, stop, step, "Arguments can only be integers, \":\" or null");
    }
    // slice(5) === slice(null, 5)
    if (stop === undefined) {
        stop = start;
        start = null;
    }
    // if (start !== null && stop !== null && start > stop) {
    //     throw new InvalidSliceError(start, stop, step, "to is higher than from");
    // }
    return {
        start: start === ":" ? null : start,
        stop: stop === ":" ? null : stop,
        step,
        _slice: true,
    };
}
/**
 * Port of adjustIndices
 * https://github.com/python/cpython/blob/master/Objects/sliceobject.c#L243
 */
function adjustIndices(start, stop, step, length) {
    if (start < 0) {
        start += length;
        if (start < 0) {
            start = (step < 0) ? -1 : 0;
        }
    }
    else if (start >= length) {
        start = (step < 0) ? length - 1 : length;
    }
    if (stop < 0) {
        stop += length;
        if (stop < 0) {
            stop = (step < 0) ? -1 : 0;
        }
    }
    else if (stop >= length) {
        stop = (step < 0) ? length - 1 : length;
    }
    if (step < 0) {
        if (stop < start) {
            const length = Math.floor((start - stop - 1) / (-step) + 1);
            return [start, stop, step, length];
        }
    }
    else {
        if (start < stop) {
            const length = Math.floor((stop - start - 1) / step + 1);
            return [start, stop, step, length];
        }
    }
    return [start, stop, step, 0];
}
/**
 * Port of slice.indices(n) and PySlice_Unpack
 * https://github.com/python/cpython/blob/master/Objects/sliceobject.c#L166
 *  https://github.com/python/cpython/blob/master/Objects/sliceobject.c#L198
 *
 * Behaviour might be slightly different as it's a weird hybrid implementation.
 */
function sliceIndices(slice, length) {
    let start;
    let stop;
    let step;
    if (slice.step === null) {
        step = 1;
    }
    else {
        step = slice.step;
    }
    if (slice.start === null) {
        start = step < 0 ? Number.MAX_SAFE_INTEGER : 0;
    }
    else {
        start = slice.start;
        if (start < 0) {
            start += length;
        }
    }
    if (slice.stop === null) {
        stop = step < 0 ? -Number.MAX_SAFE_INTEGER : Number.MAX_SAFE_INTEGER;
    }
    else {
        stop = slice.stop;
        if (stop < 0) {
            stop += length;
        }
    }
    // This clips out of bounds slices
    const s = adjustIndices(start, stop, step, length);
    start = s[0];
    stop = s[1];
    step = s[2];
    // The output length
    length = s[3];
    // With out of bounds slicing these two assertions are not useful.
    // if (stop > length) throw new Error("Stop greater than length");
    // if (start >= length) throw new Error("Start greater than or equal to length");
    if (step === 0)
        throw new Error("Step size 0 is invalid");
    return [start, stop, step, length];
}

function ensureArray(selection) {
    if (!Array.isArray(selection)) {
        return [selection];
    }
    return selection;
}
function checkSelectionLength(selection, shape) {
    if (selection.length > shape.length) {
        throw new TooManyIndicesError(selection, shape);
    }
}
/**
 * Returns both the sliceIndices per dimension and the output shape after slicing.
 */
function selectionToSliceIndices(selection, shape) {
    const sliceIndicesResult = [];
    const outShape = [];
    for (let i = 0; i < selection.length; i++) {
        const s = selection[i];
        if (typeof s === "number") {
            sliceIndicesResult.push(s);
        }
        else {
            const x = sliceIndices(s, shape[i]);
            const dimLength = x[3];
            outShape.push(dimLength);
            sliceIndicesResult.push(x);
        }
    }
    return [sliceIndicesResult, outShape];
}
/**
 * This translates "...", ":", null into a list of slices or non-negative integer selections of length shape
 */
function normalizeArraySelection(selection, shape, convertIntegerSelectionToSlices = false) {
    selection = replaceEllipsis(selection, shape);
    for (let i = 0; i < selection.length; i++) {
        const dimSelection = selection[i];
        if (typeof dimSelection === "number") {
            if (convertIntegerSelectionToSlices) {
                selection[i] = slice(dimSelection, dimSelection + 1, 1);
            }
            else {
                selection[i] = normalizeIntegerSelection(dimSelection, shape[i]);
            }
        }
        else if (isIntegerArray(dimSelection)) {
            throw new TypeError("Integer array selections are not supported (yet)");
        }
        else if (dimSelection === ":" || dimSelection === null) {
            selection[i] = slice(null, null, 1);
        }
    }
    return selection;
}
function replaceEllipsis(selection, shape) {
    selection = ensureArray(selection);
    let ellipsisIndex = -1;
    let numEllipsis = 0;
    for (let i = 0; i < selection.length; i++) {
        if (selection[i] === "...") {
            ellipsisIndex = i;
            numEllipsis += 1;
        }
    }
    if (numEllipsis > 1) {
        throw new RangeError("an index can only have a single ellipsis ('...')");
    }
    if (numEllipsis === 1) {
        // count how many items to left and right of ellipsis
        const numItemsLeft = ellipsisIndex;
        const numItemsRight = selection.length - (numItemsLeft + 1);
        const numItems = selection.length - 1; // All non-ellipsis items
        if (numItems >= shape.length) {
            // Ellipsis does nothing, just remove it
            selection = selection.filter((x) => x !== "...");
        }
        else {
            // Replace ellipsis with as many slices are needed for number of dims
            const numNewItems = shape.length - numItems;
            let newItem = selection.slice(0, numItemsLeft).concat(new Array(numNewItems).fill(null));
            if (numItemsRight > 0) {
                newItem = newItem.concat(selection.slice(selection.length - numItemsRight));
            }
            selection = newItem;
        }
    }
    // Fill out selection if not completely specified
    if (selection.length < shape.length) {
        const numMissing = shape.length - selection.length;
        selection = selection.concat(new Array(numMissing).fill(null));
    }
    checkSelectionLength(selection, shape);
    return selection;
}
function normalizeIntegerSelection(dimSelection, dimLength) {
    // Note: Maybe we should convert to integer or warn if dimSelection is not an integer
    // handle wraparound
    if (dimSelection < 0) {
        dimSelection = dimLength + dimSelection;
    }
    // handle out of bounds
    if (dimSelection >= dimLength || dimSelection < 0) {
        throw new BoundsCheckError(`index out of bounds for dimension with length ${dimLength}`);
    }
    return dimSelection;
}
function isInteger(s) {
    return typeof s === "number";
}
function isIntegerArray(s) {
    if (!Array.isArray(s)) {
        return false;
    }
    for (const e of s) {
        if (typeof e !== "number") {
            return false;
        }
    }
    return true;
}
function isSlice(s) {
    if (s !== null && s["_slice"] === true) {
        return true;
    }
    return false;
}
function isContiguousSlice(s) {
    return isSlice(s) && (s.step === null || s.step === 1);
}
function isContiguousSelection(selection) {
    selection = ensureArray(selection);
    for (let i = 0; i < selection.length; i++) {
        const s = selection[i];
        if (!(isIntegerArray(s) || isContiguousSlice(s) || s === "...")) {
            return false;
        }
    }
    return true;
}
function* product(...iterables) {
    if (iterables.length === 0) {
        return;
    }
    // make a list of iterators from the iterables
    const iterators = iterables.map(it => it());
    const results = iterators.map(it => it.next());
    // Disabled to allow empty inputs
    // if (results.some(r => r.done)) {
    //     throw new Error("Input contains an empty iterator.");
    // }
    for (let i = 0;;) {
        if (results[i].done) {
            // reset the current iterator
            iterators[i] = iterables[i]();
            results[i] = iterators[i].next();
            // advance, and exit if we've reached the end
            if (++i >= iterators.length) {
                return;
            }
        }
        else {
            yield results.map(({ value }) => value);
            i = 0;
        }
        results[i] = iterators[i].next();
    }
}
class BasicIndexer {
    constructor(selection, array) {
        selection = normalizeArraySelection(selection, array.shape);
        // Setup per-dimension indexers
        this.dimIndexers = [];
        const arrayShape = array.shape;
        for (let i = 0; i < arrayShape.length; i++) {
            let dimSelection = selection[i];
            const dimLength = arrayShape[i];
            const dimChunkLength = array.chunks[i];
            if (dimSelection === null) {
                dimSelection = slice(null);
            }
            if (isInteger(dimSelection)) {
                this.dimIndexers.push(new IntDimIndexer(dimSelection, dimLength, dimChunkLength));
            }
            else if (isSlice(dimSelection)) {
                this.dimIndexers.push(new SliceDimIndexer(dimSelection, dimLength, dimChunkLength));
            }
            else {
                throw new RangeError(`Unspported selection item for basic indexing; expected integer or slice, got ${dimSelection}`);
            }
        }
        this.shape = [];
        for (const d of this.dimIndexers) {
            if (d instanceof SliceDimIndexer) {
                this.shape.push(d.numItems);
            }
        }
        this.dropAxes = null;
    }
    *iter() {
        const dimIndexerIterables = this.dimIndexers.map(x => (() => x.iter()));
        const dimIndexerProduct = product(...dimIndexerIterables);
        for (const dimProjections of dimIndexerProduct) {
            // TODO fix this, I think the product outputs too many combinations
            const chunkCoords = [];
            const chunkSelection = [];
            const outSelection = [];
            for (const p of dimProjections) {
                chunkCoords.push((p).dimChunkIndex);
                chunkSelection.push((p).dimChunkSelection);
                if ((p).dimOutSelection !== null) {
                    outSelection.push((p).dimOutSelection);
                }
            }
            yield {
                chunkCoords,
                chunkSelection,
                outSelection,
            };
        }
    }
}
class IntDimIndexer {
    constructor(dimSelection, dimLength, dimChunkLength) {
        dimSelection = normalizeIntegerSelection(dimSelection, dimLength);
        this.dimSelection = dimSelection;
        this.dimLength = dimLength;
        this.dimChunkLength = dimChunkLength;
        this.numItems = 1;
    }
    *iter() {
        const dimChunkIndex = Math.floor(this.dimSelection / this.dimChunkLength);
        const dimOffset = dimChunkIndex * this.dimChunkLength;
        const dimChunkSelection = this.dimSelection - dimOffset;
        const dimOutSelection = null;
        yield {
            dimChunkIndex,
            dimChunkSelection,
            dimOutSelection,
        };
    }
}
class SliceDimIndexer {
    constructor(dimSelection, dimLength, dimChunkLength) {
        // Normalize
        const [start, stop, step] = sliceIndices(dimSelection, dimLength);
        this.start = start;
        this.stop = stop;
        this.step = step;
        if (this.step < 1) {
            throw new NegativeStepError();
        }
        this.dimLength = dimLength;
        this.dimChunkLength = dimChunkLength;
        this.numItems = Math.max(0, Math.ceil((this.stop - this.start) / this.step));
        this.numChunks = Math.ceil(this.dimLength / this.dimChunkLength);
    }
    *iter() {
        const dimChunkIndexFrom = Math.floor(this.start / this.dimChunkLength);
        const dimChunkIndexTo = Math.ceil(this.stop / this.dimChunkLength);
        // Iterate over chunks in range
        for (let dimChunkIndex = dimChunkIndexFrom; dimChunkIndex < dimChunkIndexTo; dimChunkIndex++) {
            // Compute offsets for chunk within overall array
            const dimOffset = dimChunkIndex * this.dimChunkLength;
            const dimLimit = Math.min(this.dimLength, (dimChunkIndex + 1) * this.dimChunkLength);
            // Determine chunk length, accounting for trailing chunk
            const dimChunkLength = dimLimit - dimOffset;
            let dimChunkSelStart;
            let dimChunkSelStop;
            let dimOutOffset;
            if (this.start < dimOffset) {
                // Selection starts before current chunk
                dimChunkSelStart = 0;
                const remainder = (dimOffset - this.start) % this.step;
                if (remainder > 0) {
                    dimChunkSelStart += this.step - remainder;
                }
                // Compute number of previous items, provides offset into output array
                dimOutOffset = Math.ceil((dimOffset - this.start) / this.step);
            }
            else {
                // Selection starts within current chunk
                dimChunkSelStart = this.start - dimOffset;
                dimOutOffset = 0;
            }
            if (this.stop > dimLimit) {
                // Selection ends after current chunk
                dimChunkSelStop = dimChunkLength;
            }
            else {
                // Selection ends within current chunk
                dimChunkSelStop = this.stop - dimOffset;
            }
            const dimChunkSelection = slice(dimChunkSelStart, dimChunkSelStop, this.step);
            const dimChunkNumItems = Math.ceil((dimChunkSelStop - dimChunkSelStart) / this.step);
            const dimOutSelection = slice(dimOutOffset, dimOutOffset + dimChunkNumItems);
            yield {
                dimChunkIndex,
                dimChunkSelection,
                dimOutSelection,
            };
        }
    }
}

/**
 * This should be true only if this javascript is getting executed in Node.
 */
const IS_NODE = typeof process !== "undefined" && process.versions && process.versions.node;
// eslint-disable-next-line @typescript-eslint/no-empty-function
function noop() { }
// eslint-disable-next-line @typescript-eslint/ban-types
function normalizeStoragePath(path) {
    if (path === null) {
        return "";
    }
    if (path instanceof String) {
        path = path.valueOf();
    }
    // convert backslash to forward slash
    path = path.replace(/\\/g, "/");
    // ensure no leading slash
    while (path.length > 0 && path[0] === '/') {
        path = path.slice(1);
    }
    // ensure no trailing slash
    while (path.length > 0 && path[path.length - 1] === '/') {
        path = path.slice(0, path.length - 1);
    }
    // collapse any repeated slashes
    path = path.replace(/\/\/+/g, "/");
    // don't allow path segments with just '.' or '..'
    const segments = path.split('/');
    for (const s of segments) {
        if (s === "." || s === "..") {
            throw Error("path containing '.' or '..' segment not allowed");
        }
    }
    return path;
}
function normalizeShape(shape) {
    if (typeof shape === "number") {
        shape = [shape];
    }
    return shape.map(x => Math.floor(x));
}
function normalizeChunks(chunks, shape) {
    // Assume shape is already normalized
    if (chunks === null || chunks === true) {
        throw new Error("Chunk guessing is not supported yet");
    }
    if (chunks === false) {
        return shape;
    }
    if (typeof chunks === "number") {
        chunks = [chunks];
    }
    // handle underspecified chunks
    if (chunks.length < shape.length) {
        // assume chunks across remaining dimensions
        chunks = chunks.concat(shape.slice(chunks.length));
    }
    return chunks.map((x, idx) => {
        // handle null or -1 in chunks
        if (x === -1 || x === null) {
            return shape[idx];
        }
        else {
            return Math.floor(x);
        }
    });
}
function normalizeOrder(order) {
    order = order.toUpperCase();
    return order;
}
function normalizeDtype(dtype) {
    return dtype;
}
function normalizeFillValue(fillValue) {
    return fillValue;
}
/**
 * Determine whether `item` specifies a complete slice of array with the
 *  given `shape`. Used to optimize __setitem__ operations on chunks
 * @param item
 * @param shape
 */
function isTotalSlice(item, shape) {
    if (item === null) {
        return true;
    }
    if (!Array.isArray(item)) {
        item = [item];
    }
    for (let i = 0; i < Math.min(item.length, shape.length); i++) {
        const it = item[i];
        if (it === null)
            continue;
        if (isSlice(it)) {
            const s = it;
            const isStepOne = s.step === 1 || s.step === null;
            if (s.start === null && s.stop === null && isStepOne) {
                continue;
            }
            if ((s.stop - s.start) === shape[i] && isStepOne) {
                continue;
            }
            return false;
        }
        return false;
        // } else {
        //     console.error(`isTotalSlice unexpected non-slice, got ${it}`);
        //     return false;
        // }
    }
    return true;
}
/**
 * Checks for === equality of all elements.
 */
function arrayEquals1D(a, b) {
    if (a.length !== b.length) {
        return false;
    }
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) {
            return false;
        }
    }
    return true;
}
/*
 * Determines "C" order strides for a given shape array.
 * Strides provide integer steps in each dimention to traverse an ndarray.
 *
 * NOTE: - These strides here are distinct from numpy.ndarray.strides, which describe actual byte steps.
 *       - Strides are assumed to be contiguous, so initial step is 1. Thus, output will always be [XX, XX, 1].
 */
function getStrides(shape) {
    // adapted from https://github.com/scijs/ndarray/blob/master/ndarray.js#L326-L330
    const ndim = shape.length;
    const strides = Array(ndim);
    let step = 1; // init step
    for (let i = ndim - 1; i >= 0; i--) {
        strides[i] = step;
        step *= shape[i];
    }
    return strides;
}
function resolveUrl(root, path) {
    const base = typeof root === 'string' ? new URL(root) : root;
    if (!base.pathname.endsWith('/')) {
        // ensure trailing slash so that base is resolved as _directory_
        base.pathname += '/';
    }
    const resolved = new URL(path, base);
    // copy search params to new URL
    resolved.search = base.search;
    return resolved.href;
}
/**
 * Swaps byte order in-place for a given TypedArray.
 * Used to flip endian-ness when getting/setting chunks from/to zarr store.
 * @param src TypedArray
 */
function byteSwapInplace(src) {
    const b = src.BYTES_PER_ELEMENT;
    if (b === 1)
        return; // no swapping needed
    if (IS_NODE) {
        // Use builtin methods for swapping if in Node environment
        const bytes = Buffer.from(src.buffer, src.byteOffset, src.length * b);
        if (b === 2)
            bytes.swap16();
        if (b === 4)
            bytes.swap32();
        if (b === 8)
            bytes.swap64();
        return;
    }
    // In browser, need to flip manually
    // Adapted from https://github.com/zbjornson/node-bswap/blob/master/bswap.js
    const flipper = new Uint8Array(src.buffer, src.byteOffset, src.length * b);
    const numFlips = b / 2;
    const endByteIndex = b - 1;
    let t;
    for (let i = 0; i < flipper.length; i += b) {
        for (let j = 0; j < numFlips; j++) {
            t = flipper[i + j];
            flipper[i + j] = flipper[i + endByteIndex - j];
            flipper[i + endByteIndex - j] = t;
        }
    }
}
/**
 * Creates a copy of a TypedArray and swaps bytes.
 * Used to flip endian-ness when getting/setting chunks from/to zarr store.
 * @param src TypedArray
 */
function byteSwap(src) {
    const copy = src.slice();
    byteSwapInplace(copy);
    return copy;
}
function convertColMajorToRowMajor2D(src, out, shape) {
    let idx = 0;
    const shape0 = shape[0];
    const shape1 = shape[1];
    const stride0 = shape1;
    for (let i1 = 0; i1 < shape1; i1++) {
        for (let i0 = 0; i0 < shape0; i0++) {
            out[i0 * stride0 + i1] = src[idx++];
        }
    }
}
function convertColMajorToRowMajor3D(src, out, shape) {
    let idx = 0;
    const shape0 = shape[0];
    const shape1 = shape[1];
    const shape2 = shape[2];
    const stride0 = shape2 * shape1;
    const stride1 = shape2;
    for (let i2 = 0; i2 < shape2; i2++) {
        for (let i1 = 0; i1 < shape1; i1++) {
            for (let i0 = 0; i0 < shape0; i0++) {
                out[i0 * stride0 + i1 * stride1 + i2] = src[idx++];
            }
        }
    }
}
function convertColMajorToRowMajor4D(src, out, shape) {
    let idx = 0;
    const shape0 = shape[0];
    const shape1 = shape[1];
    const shape2 = shape[2];
    const shape3 = shape[3];
    const stride0 = shape3 * shape2 * shape1;
    const stride1 = shape3 * shape2;
    const stride2 = shape3;
    for (let i3 = 0; i3 < shape3; i3++) {
        for (let i2 = 0; i2 < shape2; i2++) {
            for (let i1 = 0; i1 < shape1; i1++) {
                for (let i0 = 0; i0 < shape0; i0++) {
                    out[i0 * stride0 + i1 * stride1 + i2 * stride2 + i3] = src[idx++];
                }
            }
        }
    }
}
function convertColMajorToRowMajorGeneric(src, out, shape) {
    const nDims = shape.length;
    const size = shape.reduce((r, a) => r * a);
    const rowMajorStrides = shape.map((_, i) => i + 1 === nDims ? 1 : shape.slice(i + 1).reduce((r, a) => r * a, 1));
    const index = Array(nDims).fill(0);
    for (let colMajorIdx = 0; colMajorIdx < size; colMajorIdx++) {
        let rowMajorIdx = 0;
        for (let dim = 0; dim < nDims; dim++) {
            rowMajorIdx += index[dim] * rowMajorStrides[dim];
        }
        out[rowMajorIdx] = src[colMajorIdx];
        index[0] += 1;
        // Handle carry-over
        for (let dim = 0; dim < nDims; dim++) {
            if (index[dim] === shape[dim]) {
                if (dim + 1 === nDims) {
                    return;
                }
                index[dim] = 0;
                index[dim + 1] += 1;
            }
        }
    }
}
const colMajorToRowMajorConverters = {
    [0]: noop,
    [1]: noop,
    [2]: convertColMajorToRowMajor2D,
    [3]: convertColMajorToRowMajor3D,
    [4]: convertColMajorToRowMajor4D,
};
/**
 * Rewrites a copy of a TypedArray while converting it from column-major (F-order) to row-major (C-order).
 * @param src TypedArray
 * @param out TypedArray
 * @param shape number[]
 */
function convertColMajorToRowMajor(src, out, shape) {
    return (colMajorToRowMajorConverters[shape.length] || convertColMajorToRowMajorGeneric)(src, out, shape);
}
function isArrayBufferLike(obj) {
    if (obj === null) {
        return false;
    }
    if (obj instanceof ArrayBuffer) {
        return true;
    }
    if (typeof SharedArrayBuffer === "function" && obj instanceof SharedArrayBuffer) {
        return true;
    }
    if (IS_NODE) { // Necessary for Node.js for some reason..
        return obj.toString().startsWith("[object ArrayBuffer]")
            || obj.toString().startsWith("[object SharedArrayBuffer]");
    }
    return false;
}

const ARRAY_META_KEY = ".zarray";
const GROUP_META_KEY = ".zgroup";
const ATTRS_META_KEY = ".zattrs";

/**
 * Return true if the store contains an array at the given logical path.
 */
async function containsArray(store, path = null) {
    path = normalizeStoragePath(path);
    const prefix = pathToPrefix(path);
    const key = prefix + ARRAY_META_KEY;
    console.log("containsArray",store);
    return store.containsItem(key);
}
/**
 * Return true if the store contains a group at the given logical path.
 */
async function containsGroup(store, path = null) {
    path = normalizeStoragePath(path);
    const prefix = pathToPrefix(path);
    const key = prefix + GROUP_META_KEY;
    console.log("containsGroup",store);
    return store.containsItem(key);
}
function pathToPrefix(path) {
    // assume path already normalized
    if (path.length > 0) {
        return path + '/';
    }
    return '';
}
async function requireParentGroup(store, path, chunkStore, overwrite) {
    // Assume path is normalized
    if (path.length === 0) {
        return;
    }
    const segments = path.split("/");
    let p = "";
    for (const s of segments.slice(0, segments.length - 1)) {
        p += s;
        if (await containsArray(store, p)) {
            await initGroupMetadata(store, p, overwrite);
        }
        else if (!await containsGroup(store, p)) {
            await initGroupMetadata(store, p);
        }
        p += "/";
    }
}
async function initGroupMetadata(store, path = null, overwrite = false) {
    path = normalizeStoragePath(path);
    // Guard conditions
    if (overwrite) {
        throw Error("Group overwriting not implemented yet :(");
    }
    else if (await containsArray(store, path)) {
        throw new ContainsArrayError(path);
    }
    else if (await containsGroup(store, path)) {
        throw new ContainsGroupError(path);
    }
    const metadata = { zarr_format: 2 };
    const key = pathToPrefix(path) + GROUP_META_KEY;
    await store.setItem(key, JSON.stringify(metadata));
}
/**
 *  Initialize a group store. Note that this is a low-level function and there should be no
 *  need to call this directly from user code.
 */
async function initGroup(store, path = null, chunkStore = null, overwrite = false) {
    path = normalizeStoragePath(path);
    await requireParentGroup(store, path, chunkStore, overwrite);
    await initGroupMetadata(store, path, overwrite);
}
async function initArrayMetadata(store, shape, chunks, dtype, path, compressor, fillValue, order, overwrite, chunkStore, filters, dimensionSeparator) {
    // Guard conditions
    if (overwrite) {
        throw Error("Array overwriting not implemented yet :(");
    }
    else if (await containsArray(store, path)) {
        throw new ContainsArrayError(path);
    }
    else if (await containsGroup(store, path)) {
        throw new ContainsGroupError(path);
    }
    // Normalize metadata,  does type checking too.
    dtype = normalizeDtype(dtype);
    shape = normalizeShape(shape);
    chunks = normalizeChunks(chunks, shape);
    order = normalizeOrder(order);
    fillValue = normalizeFillValue(fillValue);
    if (filters !== null && filters.length > 0) {
        throw Error("Filters are not supported yet");
    }
    let serializedFillValue = fillValue;
    if (typeof fillValue === "number") {
        if (Number.isNaN(fillValue))
            serializedFillValue = "NaN";
        if (Number.POSITIVE_INFINITY === fillValue)
            serializedFillValue = "Infinity";
        if (Number.NEGATIVE_INFINITY === fillValue)
            serializedFillValue = "-Infinity";
    }
    filters = null;
    const metadata = {
        zarr_format: 2,
        shape: shape,
        chunks: chunks,
        dtype: dtype,
        fill_value: serializedFillValue,
        order: order,
        compressor: compressor,
        filters: filters,
    };
    if (dimensionSeparator) {
        metadata.dimension_separator = dimensionSeparator;
    }
    const metaKey = pathToPrefix(path) + ARRAY_META_KEY;
    await store.setItem(metaKey, JSON.stringify(metadata));
}
/**
 *
 * Initialize an array store with the given configuration. Note that this is a low-level
 * function and there should be no need to call this directly from user code
 */
async function initArray(store, shape, chunks, dtype, path = null, compressor = null, fillValue = null, order = "C", overwrite = false, chunkStore = null, filters = null, dimensionSeparator) {
    path = normalizeStoragePath(path);
    await requireParentGroup(store, path, chunkStore, overwrite);
    await initArrayMetadata(store, shape, chunks, dtype, path, compressor, fillValue, order, overwrite, chunkStore, filters, dimensionSeparator);
}

function parseMetadata(s) {
    // Here we allow that a store may return an already-parsed metadata object,
    // or a string of JSON that we will parse here. We allow for an already-parsed
    // object to accommodate a consolidated metadata store, where all the metadata for
    // all groups and arrays will already have been parsed from JSON.
    if (typeof s !== 'string') {
        // tslint:disable-next-line: strict-type-predicates
        if (IS_NODE && Buffer.isBuffer(s)) {
            return JSON.parse(s.toString());
        }
        else if (isArrayBufferLike(s)) {
            const utf8Decoder = new TextDecoder();
            const bytes = new Uint8Array(s);
            return JSON.parse(utf8Decoder.decode(bytes));
        }
        else {
            return s;
        }
    }
    return JSON.parse(s);
}

/**
 * Class providing access to user attributes on an array or group. Should not be
 * instantiated directly, will be available via the `.attrs` property of an array or
 * group.
 */
class Attributes {
    constructor(store, key, readOnly, cache = true) {
        this.store = store;
        this.key = key;
        this.readOnly = readOnly;
        this.cache = cache;
        this.cachedValue = null;
    }
    /**
     * Retrieve all attributes as a JSON object.
     */
    async asObject() {
        if (this.cache && this.cachedValue !== null) {
            return this.cachedValue;
        }
        const o = await this.getNoSync();
        if (this.cache) {
            this.cachedValue = o;
        }
        return o;
    }
    async getNoSync() {
        try {
            const data = await this.store.getItem(this.key);
            // TODO fix typing?
            return parseMetadata(data);
        }
        catch (error) {
            return {};
        }
    }
    async setNoSync(key, value) {
        const d = await this.getNoSync();
        d[key] = value;
        await this.putNoSync(d);
        return true;
    }
    async putNoSync(m) {
        await this.store.setItem(this.key, JSON.stringify(m));
        if (this.cache) {
            this.cachedValue = m;
        }
    }
    async delNoSync(key) {
        const d = await this.getNoSync();
        delete d[key];
        await this.putNoSync(d);
        return true;
    }
    /**
     * Overwrite all attributes with the provided object in a single operation
     */
    async put(d) {
        if (this.readOnly) {
            throw new PermissionError("attributes are read-only");
        }
        return this.putNoSync(d);
    }
    async setItem(key, value) {
        if (this.readOnly) {
            throw new PermissionError("attributes are read-only");
        }
        return this.setNoSync(key, value);
    }
    async getItem(key) {
        return (await this.asObject())[key];
    }
    async deleteItem(key) {
        if (this.readOnly) {
            throw new PermissionError("attributes are read-only");
        }
        return this.delNoSync(key);
    }
    async containsItem(key) {
        return (await this.asObject())[key] !== undefined;
    }
    proxy() {
        return createProxy(this);
    }
}

// eslint-disable-next-line @typescript-eslint/naming-convention
const Float16Array = globalThis.Float16Array;
const DTYPE_TYPEDARRAY_MAPPING = {
    '|b': Int8Array,
    '|b1': Uint8Array,
    '|B': Uint8Array,
    '|u1': Uint8Array,
    '|i1': Int8Array,
    '<b': Int8Array,
    '<B': Uint8Array,
    '<u1': Uint8Array,
    '<i1': Int8Array,
    '<u2': Uint16Array,
    '<i2': Int16Array,
    '<u4': Uint32Array,
    '<i4': Int32Array,
    '<f4': Float32Array,
    '<f2': Float16Array,
    '<f8': Float64Array,
    '<u8': BigUint64Array,
    '<i8': BigInt64Array,
    '>b': Int8Array,
    '>B': Uint8Array,
    '>u1': Uint8Array,
    '>i1': Int8Array,
    '>u2': Uint16Array,
    '>i2': Int16Array,
    '>u4': Uint32Array,
    '>i4': Int32Array,
    '>f4': Float32Array,
    '>f2': Float16Array,
    '>f8': Float64Array,
    '>u8': BigUint64Array,
    '>i8': BigInt64Array
};
function getTypedArrayCtr(dtype) {
    const ctr = DTYPE_TYPEDARRAY_MAPPING[dtype];
    if (!ctr) {
        if (dtype.slice(1) === 'f2') {
            throw Error(`'${dtype}' is not supported natively in zarr.js. ` +
                `In order to access this dataset you must make Float16Array available as a global. ` +
                `See https://github.com/gzuidhof/zarr.js/issues/127`);
        }
        throw Error(`Dtype not recognized or not supported in zarr.js, got ${dtype}.`);
    }
    return ctr;
}
/*
 * Called by NestedArray and RawArray constructors only.
 * We byte-swap the buffer of a store after decoding
 * since TypedArray views are little endian only.
 *
 * This means NestedArrays and RawArrays will always be little endian,
 * unless a numpy-like library comes around and can handle endianess
 * for buffer views.
 */
function getTypedArrayDtypeString(t) {
    // Favour the types below instead of small and big B
    if (t instanceof Uint8Array)
        return '|u1';
    if (t instanceof Int8Array)
        return '|i1';
    if (t instanceof Uint16Array)
        return '<u2';
    if (t instanceof Int16Array)
        return '<i2';
    if (t instanceof Uint32Array)
        return '<u4';
    if (t instanceof Int32Array)
        return '<i4';
    if (t instanceof Float32Array)
        return '<f4';
    if (t instanceof Float64Array)
        return '<f8';
    if (t instanceof BigUint64Array)
        return '<u8';
    if (t instanceof BigInt64Array)
        return '<i8';
    throw new ValueError('Mapping for TypedArray to Dtypestring not known');
}

/**
 * Digs down into the dimensions of given array to find the TypedArray and returns its constructor.
 * Better to use sparingly.
 */
function getNestedArrayConstructor(arr) {
    // TODO fix typing
    // tslint:disable-next-line: strict-type-predicates
    if (arr.byteLength !== undefined) {
        return (arr).constructor;
    }
    return getNestedArrayConstructor(arr[0]);
}
/**
 * Returns both the slice result and new output shape
 * @param arr NestedArray to slice
 * @param shape The shape of the NestedArray
 * @param selection
 */
function sliceNestedArray(arr, shape, selection) {
    // This translates "...", ":", null into a list of slices or integer selections
    const normalizedSelection = normalizeArraySelection(selection, shape);
    const [sliceIndices, outShape] = selectionToSliceIndices(normalizedSelection, shape);
    const outArray = _sliceNestedArray(arr, shape, sliceIndices);
    return [outArray, outShape];
}
function _sliceNestedArray(arr, shape, selection) {
    const currentSlice = selection[0];
    // Is this necessary?
    // // This is possible when a slice list is passed shorter than the amount of dimensions
    // // tslint:disable-next-line: strict-type-predicates
    // if (currentSlice === undefined) {
    //     return arr.slice();
    // }
    // When a number is passed that dimension is squeezed
    if (typeof currentSlice === "number") {
        // Assume already normalized integer selection here.
        if (shape.length === 1) {
            return arr[currentSlice];
        }
        else {
            return _sliceNestedArray(arr[currentSlice], shape.slice(1), selection.slice(1));
        }
    }
    const [from, to, step, outputSize] = currentSlice;
    if (outputSize === 0) {
        return new (getNestedArrayConstructor(arr))(0);
    }
    if (shape.length === 1) {
        if (step === 1) {
            return arr.slice(from, to);
        }
        const newArrData = new arr.constructor(outputSize);
        for (let i = 0; i < outputSize; i++) {
            newArrData[i] = arr[from + i * step];
        }
        return newArrData;
    }
    let newArr = new Array(outputSize);
    for (let i = 0; i < outputSize; i++) {
        newArr[i] = _sliceNestedArray(arr[from + i * step], shape.slice(1), selection.slice(1));
    }
    // This is necessary to ensure that the return value is a NestedArray if the last dimension is squeezed
    // e.g. shape [2,1] with slice [:, 0] would otherwise result in a list of numbers instead of a valid NestedArray
    if (outputSize > 0 && (typeof newArr[0] === "number" || typeof newArr[0] === "bigint")) {
        const typedArrayConstructor = arr[0].constructor;
        newArr = typedArrayConstructor.from(newArr);
    }
    return newArr;
}
function setNestedArrayToScalar(dstArr, value, destShape, selection) {
    // This translates "...", ":", null, etc into a list of slices.
    const normalizedSelection = normalizeArraySelection(selection, destShape, true);
    // Above we force the results to be SliceIndicesIndices only, without integer selections making this cast is safe.
    const [sliceIndices, _outShape] = selectionToSliceIndices(normalizedSelection, destShape);
    _setNestedArrayToScalar(dstArr, value, destShape, sliceIndices);
}
function setNestedArray(dstArr, sourceArr, destShape, sourceShape, selection) {
    // This translates "...", ":", null, etc into a list of slices.
    const normalizedSelection = normalizeArraySelection(selection, destShape, false);
    const [sliceIndices, outShape] = selectionToSliceIndices(normalizedSelection, destShape);
    // TODO: replace with non stringify equality check
    if (JSON.stringify(outShape) !== JSON.stringify(sourceShape)) {
        throw new ValueError(`Shape mismatch in target and source NestedArray: ${outShape} and ${sourceShape}`);
    }
    _setNestedArray(dstArr, sourceArr, destShape, sliceIndices);
}
function _setNestedArray(dstArr, sourceArr, shape, selection) {
    const currentSlice = selection[0];
    if (typeof sourceArr === "number" || typeof sourceArr === "bigint") {
        _setNestedArrayToScalar(dstArr, sourceArr, shape, selection.map(x => typeof x === "number" ? [x, x + 1, 1, 1] : x));
        return;
    }
    // This dimension is squeezed.
    if (typeof currentSlice === "number") {
        _setNestedArray(dstArr[currentSlice], sourceArr, shape.slice(1), selection.slice(1));
        return;
    }
    const [from, _to, step, outputSize] = currentSlice;
    if (shape.length === 1) {
        if (step === 1) {
            const values = sourceArr;
            dstArr.set(values, from);
        }
        else {
            for (let i = 0; i < outputSize; i++) {
                dstArr[from + i * step] = (sourceArr)[i];
            }
        }
        return;
    }
    for (let i = 0; i < outputSize; i++) {
        _setNestedArray(dstArr[from + i * step], sourceArr[i], shape.slice(1), selection.slice(1));
    }
}
function _setNestedArrayToScalar(dstArr, value, shape, selection) {
    const currentSlice = selection[0];
    const [from, to, step, outputSize] = currentSlice;
    if (shape.length === 1) {
        if (step === 1) {
            dstArr.fill(value, from, to);
        }
        else {
            for (let i = 0; i < outputSize; i++) {
                dstArr[from + i * step] = value;
            }
        }
        return;
    }
    for (let i = 0; i < outputSize; i++) {
        _setNestedArrayToScalar(dstArr[from + i * step], value, shape.slice(1), selection.slice(1));
    }
}
function flattenNestedArray(arr, shape, constr) {
    if (constr === undefined) {
        constr = getNestedArrayConstructor(arr);
    }
    const size = shape.reduce((x, y) => x * y, 1);
    const outArr = new constr(size);
    _flattenNestedArray(arr, shape, outArr, 0);
    return outArr;
}
function _flattenNestedArray(arr, shape, outArr, offset) {
    if (shape.length === 1) {
        // This is only ever reached if called with rank 1 shape, never reached through recursion.
        // We just slice set the array directly from one level above to save some function calls.
        const values = arr;
        outArr.set(values, offset);
        return;
    }
    if (shape.length === 2) {
        for (let i = 0; i < shape[0]; i++) {
            const values = arr;
            outArr.set(values[i], offset + shape[1] * i);
        }
        return arr;
    }
    const nextShape = shape.slice(1);
    // Small optimization possible here: this can be precomputed for different levels of depth and passed on.
    const mult = nextShape.reduce((x, y) => x * y, 1);
    for (let i = 0; i < shape[0]; i++) {
        _flattenNestedArray(arr[i], nextShape, outArr, offset + mult * i);
    }
    return arr;
}

class NestedArray {
    constructor(data, shape, dtype) {
        const dataIsTypedArray = data !== null && !!data.BYTES_PER_ELEMENT;
        if (shape === undefined) {
            if (!dataIsTypedArray) {
                throw new ValueError("Shape argument is required unless you pass in a TypedArray");
            }
            shape = [data.length];
        }
        if (dtype === undefined) {
            if (!dataIsTypedArray) {
                throw new ValueError("Dtype argument is required unless you pass in a TypedArray");
            }
            dtype = getTypedArrayDtypeString(data);
        }
        shape = normalizeShape(shape);
        this.shape = shape;
        this.dtype = dtype;
        console.log(dtype);
        if (dataIsTypedArray && shape.length !== 1) {
            data = data.buffer;
        }
        // Zero dimension array.. they are a bit weirdly represented now, they will only ever occur internally
        console.log(IS_NODE, isArrayBufferLike(data), data === null);
        if (this.shape.length === 0) {
            this.data = new (getTypedArrayCtr(dtype))(1);
        }
        else if (
        // tslint:disable-next-line: strict-type-predicates
        (IS_NODE && Buffer.isBuffer(data))
            || isArrayBufferLike(data) && dtype[dtype.length - 1] != "O"
            || data === null && dtype[dtype.length - 1] != "O") {
            // Create from ArrayBuffer or Buffer
            const numShapeElements = shape.reduce((x, y) => x * y, 1);
            if (data === null) {
                data = new ArrayBuffer(numShapeElements * parseInt(dtype[dtype.length - 1], 10));
            }
            const numDataElements = data.byteLength / parseInt(dtype[dtype.length - 1], 10);
            if (numShapeElements !== numDataElements) {
                throw new Error(`Buffer has ${numDataElements} of dtype ${dtype}, shape is too large or small ${shape} (flat=${numShapeElements})`);
            }
            const typeConstructor = getTypedArrayCtr(dtype);
            this.data = createNestedArray(data, typeConstructor, shape);
        }
        else {
            this.data = data;
        }
    }
    get(selection) {
        const [sliceResult, outShape] = sliceNestedArray(this.data, this.shape, selection);
        if (outShape.length === 0) {
            return sliceResult;
        }
        else {
            return new NestedArray(sliceResult, outShape, this.dtype);
        }
    }
    set(selection = null, value) {
        if (selection === null) {
            selection = [slice(null)];
        }
        if (typeof value === "number" || typeof value === "bigint") {
            if (this.shape.length === 0) {
                // Zero dimension array...
                if (typeof this.data[0] === "number") {
                    this.data[0] = value;
                }
                else {
                    this.data[0] = BigInt(value);
                }
            }
            else {
                setNestedArrayToScalar(this.data, value, this.shape, selection);
            }
        }
        else {
            setNestedArray(this.data, value.data, this.shape, value.shape, selection);
        }
    }
    flatten() {
        if (this.shape.length === 1) {
            return this.data;
        }
        return flattenNestedArray(this.data, this.shape, getTypedArrayCtr(this.dtype));
    }
    /**
     * Currently only supports a single integer as the size, TODO: support start, stop, step.
     */
    static arange(size, dtype = "<i4") {
        const constr = getTypedArrayCtr(dtype);
        const data = rangeTypedArray([size], constr);
        return new NestedArray(data, [size], dtype);
    }
}
/**
 * Creates a TypedArray with values 0 through N where N is the product of the shape.
 */
function rangeTypedArray(shape, tContructor) {
    const size = shape.reduce((x, y) => x * y, 1);
    const data = new tContructor(size);
    let values = [];
    if (data[Symbol.toStringTag] === 'BigUint64Array' || data[Symbol.toStringTag] === 'BigInt64Array') {
        values = [...Array(size).keys()].map(BigInt);
    }
    else {
        values = [...Array(size).keys()];
    }
    data.set(values);
    return data;
}
/**
 * Creates multi-dimensional (rank > 1) array given input data and shape recursively.
 * What it does is create a Array<Array<...<Array<Uint8Array>>> or some other typed array.
 * This is for internal use, there should be no need to call this from user code.
 * @param data a buffer containing the data for this array.
 * @param t constructor for the datatype of choice
 * @param shape list of numbers describing the size in each dimension
 * @param offset in bytes for this dimension
 */
function createNestedArray(data, t, shape, offset = 0) {
    if (shape.length === 1) {
        // This is only ever reached if called with rank 1 shape, never reached through recursion.
        // We just slice set the array directly from one level above to save some function calls.
        return new t(data.slice(offset, offset + shape[0] * t.BYTES_PER_ELEMENT));
    }
    const arr = new Array(shape[0]);
    if (shape.length === 2) {
        for (let i = 0; i < shape[0]; i++) {
            arr[i] = new t(data.slice(offset + shape[1] * i * t.BYTES_PER_ELEMENT, offset + shape[1] * (i + 1) * t.BYTES_PER_ELEMENT));
        }
        return arr;
    }
    const nextShape = shape.slice(1);
    // Small optimization possible here: this can be precomputed for different levels of depth and passed on.
    const mult = nextShape.reduce((x, y) => x * y, 1);
    for (let i = 0; i < shape[0]; i++) {
        arr[i] = createNestedArray(data, t, nextShape, offset + mult * i * t.BYTES_PER_ELEMENT);
    }
    return arr;
}

function setRawArrayToScalar(dstArr, dstStrides, dstShape, dstSelection, value) {
    // This translates "...", ":", null, etc into a list of slices.
    const normalizedSelection = normalizeArraySelection(dstSelection, dstShape, true);
    const [sliceIndices] = selectionToSliceIndices(normalizedSelection, dstShape);
    // Above we force the results to be SliceIndicesIndices only, without integer selections making this cast is safe.
    _setRawArrayToScalar(value, dstArr, dstStrides, sliceIndices);
}
function setRawArray(dstArr, dstStrides, dstShape, dstSelection, sourceArr, sourceStrides, sourceShape) {
    // This translates "...", ":", null, etc into a list of slices.
    const normalizedDstSelection = normalizeArraySelection(dstSelection, dstShape, false);
    const [dstSliceIndices, outShape] = selectionToSliceIndices(normalizedDstSelection, dstShape);
    // TODO: replace with non stringify equality check
    if (JSON.stringify(outShape) !== JSON.stringify(sourceShape)) {
        throw new ValueError(`Shape mismatch in target and source RawArray: ${outShape} and ${sourceShape}`);
    }
    _setRawArray(dstArr, dstStrides, dstSliceIndices, sourceArr, sourceStrides);
}
function setRawArrayFromChunkItem(dstArr, dstStrides, dstShape, dstSelection, sourceArr, sourceStrides, sourceShape, sourceSelection) {
    // This translates "...", ":", null, etc into a list of slices.
    const normalizedDstSelection = normalizeArraySelection(dstSelection, dstShape, true);
    // Above we force the results to be dstSliceIndices only, without integer selections making this cast is safe.
    const [dstSliceIndices] = selectionToSliceIndices(normalizedDstSelection, dstShape);
    const normalizedSourceSelection = normalizeArraySelection(sourceSelection, sourceShape, false);
    const [sourceSliceIndicies] = selectionToSliceIndices(normalizedSourceSelection, sourceShape);
    // TODO check to ensure chunk and dest selection are same shape?
    // As is, this only gets called in ZarrArray.getRaw where this condition should be ensured, and check might hinder performance.
    _setRawArrayFromChunkItem(dstArr, dstStrides, dstSliceIndices, sourceArr, sourceStrides, sourceSliceIndicies);
}
function _setRawArrayToScalar(value, dstArr, dstStrides, dstSliceIndices) {
    const [currentDstSlice, ...nextDstSliceIndices] = dstSliceIndices;
    const [currentDstStride, ...nextDstStrides] = dstStrides;
    const [from, _to, step, outputSize] = currentDstSlice;
    if (dstStrides.length === 1) {
        if (step === 1 && currentDstStride === 1) {
            dstArr.fill(value, from, from + outputSize);
        }
        else {
            for (let i = 0; i < outputSize; i++) {
                dstArr[currentDstStride * (from + (step * i))] = value;
            }
        }
        return;
    }
    for (let i = 0; i < outputSize; i++) {
        _setRawArrayToScalar(value, dstArr.subarray(currentDstStride * (from + (step * i))), nextDstStrides, nextDstSliceIndices);
    }
}
function _setRawArray(dstArr, dstStrides, dstSliceIndices, sourceArr, sourceStrides) {
    if (dstSliceIndices.length === 0) {
        const values = sourceArr;
        dstArr.set(values);
        return;
    }
    const [currentDstSlice, ...nextDstSliceIndices] = dstSliceIndices;
    const [currentDstStride, ...nextDstStrides] = dstStrides;
    // This dimension is squeezed.
    if (typeof currentDstSlice === "number") {
        _setRawArray(dstArr.subarray(currentDstSlice * currentDstStride), nextDstStrides, nextDstSliceIndices, sourceArr, sourceStrides);
        return;
    }
    const [currentSourceStride, ...nextSourceStrides] = sourceStrides;
    const [from, _to, step, outputSize] = currentDstSlice;
    if (dstStrides.length === 1) {
        if (step === 1 && currentDstStride === 1 && currentSourceStride === 1) {
            const values = sourceArr.subarray(0, outputSize);
            dstArr.set(values, from);
        }
        else {
            for (let i = 0; i < outputSize; i++) {
                dstArr[currentDstStride * (from + (step * i))] = sourceArr[currentSourceStride * i];
            }
        }
        return;
    }
    for (let i = 0; i < outputSize; i++) {
        // Apply strides as above, using both destination and source-specific strides.
        _setRawArray(dstArr.subarray(currentDstStride * (from + (i * step))), nextDstStrides, nextDstSliceIndices, sourceArr.subarray(currentSourceStride * i), nextSourceStrides);
    }
}
function _setRawArrayFromChunkItem(dstArr, dstStrides, dstSliceIndices, sourceArr, sourceStrides, sourceSliceIndices) {
    if (sourceSliceIndices.length === 0) {
        // Case when last source dimension is squeezed
        const values = sourceArr.subarray(0, dstArr.length);
        dstArr.set(values);
        return;
    }
    // Get current indicies and strides for both destination and source arrays
    const [currentDstSlice, ...nextDstSliceIndices] = dstSliceIndices;
    const [currentSourceSlice, ...nextSourceSliceIndices] = sourceSliceIndices;
    const [currentDstStride, ...nextDstStrides] = dstStrides;
    const [currentSourceStride, ...nextSourceStrides] = sourceStrides;
    // This source dimension is squeezed
    if (typeof currentSourceSlice === "number") {
        /*
        Sets dimension offset for squeezed dimension.

        Ex. if 0th dimension is squeezed to 2nd index (numpy : arr[2,i])

            sourceArr[stride[0]* 2 + i] --> sourceArr.subarray(stride[0] * 2)[i] (sourceArr[i] in next call)

        Thus, subsequent squeezed dims are appended to the source offset.
        */
        _setRawArrayFromChunkItem(
        // Don't update destination offset/slices, just source
        dstArr, dstStrides, dstSliceIndices, sourceArr.subarray(currentSourceStride * currentSourceSlice), nextSourceStrides, nextSourceSliceIndices);
        return;
    }
    const [from, _to, step, outputSize] = currentDstSlice; // just need start and size
    const [sfrom, _sto, sstep, _soutputSize] = currentSourceSlice; // Will always be subset of dst, so don't need output size just start
    if (dstStrides.length === 1 && sourceStrides.length === 1) {
        if (step === 1 && currentDstStride === 1 && sstep === 1 && currentSourceStride === 1) {
            const values = sourceArr.subarray(sfrom, sfrom + outputSize);
            dstArr.set(values, from);
        }
        else {
            for (let i = 0; i < outputSize; i++) {
                dstArr[currentDstStride * (from + (step * i))] = sourceArr[currentSourceStride * (sfrom + (sstep * i))];
            }
        }
        return;
    }
    for (let i = 0; i < outputSize; i++) {
        // Apply strides as above, using both destination and source-specific strides.
        _setRawArrayFromChunkItem(dstArr.subarray(currentDstStride * (from + (i * step))), nextDstStrides, nextDstSliceIndices, sourceArr.subarray(currentSourceStride * (sfrom + (i * sstep))), nextSourceStrides, nextSourceSliceIndices);
    }
}

class RawArray {
    constructor(data, shape, dtype, strides) {
        const dataIsTypedArray = data !== null && !!data.BYTES_PER_ELEMENT;
        if (shape === undefined) {
            if (!dataIsTypedArray) {
                throw new ValueError("Shape argument is required unless you pass in a TypedArray");
            }
            shape = [data.length];
        }
        shape = normalizeShape(shape);
        if (dtype === undefined) {
            if (!dataIsTypedArray) {
                throw new ValueError("Dtype argument is required unless you pass in a TypedArray");
            }
            dtype = getTypedArrayDtypeString(data);
        }
        if (strides === undefined) {
            strides = getStrides(shape);
        }
        this.shape = shape;
        this.dtype = dtype;
        this.strides = strides;
        if (dataIsTypedArray && shape.length !== 1) {
            data = data.buffer;
        }
        // Zero dimension array.. they are a bit weirdly represented now, they will only ever occur internally
        if (this.shape.length === 0) {
            this.data = new (getTypedArrayCtr(dtype))(1);
        }
        else if (
        // tslint:disable-next-line: strict-type-predicates
        (IS_NODE && Buffer.isBuffer(data))
            || isArrayBufferLike(data)
            || data === null) {
            // Create from ArrayBuffer or Buffer
            const numShapeElements = shape.reduce((x, y) => x * y, 1);
            if (data === null) {
                data = new ArrayBuffer(numShapeElements * parseInt(dtype[dtype.length - 1], 10));
            }
            const numDataElements = data.byteLength / parseInt(dtype[dtype.length - 1], 10);
            if (numShapeElements !== numDataElements) {
                throw new Error(`Buffer has ${numDataElements} of dtype ${dtype}, shape is too large or small ${shape} (flat=${numShapeElements})`);
            }
            const typeConstructor = getTypedArrayCtr(dtype);
            this.data = new typeConstructor(data);
        }
        else {
            this.data = data;
        }
    }
    set(selection = null, value, chunkSelection) {
        if (selection === null) {
            selection = [slice(null)];
        }
        if (typeof value === "number" || typeof value === "bigint") {
            if (this.shape.length === 0) {
                // Zero dimension array..
                this.data[0] = value;
            }
            else {
                setRawArrayToScalar(this.data, this.strides, this.shape, selection, value);
            }
        }
        else if (value instanceof RawArray && chunkSelection) {
            // Copy directly from decoded chunk to destination array
            setRawArrayFromChunkItem(this.data, this.strides, this.shape, selection, value.data, value.strides, value.shape, chunkSelection);
        }
        else {
            setRawArray(this.data, this.strides, this.shape, selection, value.data, value.strides, value.shape);
        }
    }
}

var eventemitter3 = {exports: {}};

(function (module) {

var has = Object.prototype.hasOwnProperty
  , prefix = '~';

/**
 * Constructor to create a storage for our `EE` objects.
 * An `Events` instance is a plain object whose properties are event names.
 *
 * @constructor
 * @private
 */
function Events() {}

//
// We try to not inherit from `Object.prototype`. In some engines creating an
// instance in this way is faster than calling `Object.create(null)` directly.
// If `Object.create(null)` is not supported we prefix the event names with a
// character to make sure that the built-in object properties are not
// overridden or used as an attack vector.
//
if (Object.create) {
  Events.prototype = Object.create(null);

  //
  // This hack is needed because the `__proto__` property is still inherited in
  // some old browsers like Android 4, iPhone 5.1, Opera 11 and Safari 5.
  //
  if (!new Events().__proto__) prefix = false;
}

/**
 * Representation of a single event listener.
 *
 * @param {Function} fn The listener function.
 * @param {*} context The context to invoke the listener with.
 * @param {Boolean} [once=false] Specify if the listener is a one-time listener.
 * @constructor
 * @private
 */
function EE(fn, context, once) {
  this.fn = fn;
  this.context = context;
  this.once = once || false;
}

/**
 * Add a listener for a given event.
 *
 * @param {EventEmitter} emitter Reference to the `EventEmitter` instance.
 * @param {(String|Symbol)} event The event name.
 * @param {Function} fn The listener function.
 * @param {*} context The context to invoke the listener with.
 * @param {Boolean} once Specify if the listener is a one-time listener.
 * @returns {EventEmitter}
 * @private
 */
function addListener(emitter, event, fn, context, once) {
  if (typeof fn !== 'function') {
    throw new TypeError('The listener must be a function');
  }

  var listener = new EE(fn, context || emitter, once)
    , evt = prefix ? prefix + event : event;

  if (!emitter._events[evt]) emitter._events[evt] = listener, emitter._eventsCount++;
  else if (!emitter._events[evt].fn) emitter._events[evt].push(listener);
  else emitter._events[evt] = [emitter._events[evt], listener];

  return emitter;
}

/**
 * Clear event by name.
 *
 * @param {EventEmitter} emitter Reference to the `EventEmitter` instance.
 * @param {(String|Symbol)} evt The Event name.
 * @private
 */
function clearEvent(emitter, evt) {
  if (--emitter._eventsCount === 0) emitter._events = new Events();
  else delete emitter._events[evt];
}

/**
 * Minimal `EventEmitter` interface that is molded against the Node.js
 * `EventEmitter` interface.
 *
 * @constructor
 * @public
 */
function EventEmitter() {
  this._events = new Events();
  this._eventsCount = 0;
}

/**
 * Return an array listing the events for which the emitter has registered
 * listeners.
 *
 * @returns {Array}
 * @public
 */
EventEmitter.prototype.eventNames = function eventNames() {
  var names = []
    , events
    , name;

  if (this._eventsCount === 0) return names;

  for (name in (events = this._events)) {
    if (has.call(events, name)) names.push(prefix ? name.slice(1) : name);
  }

  if (Object.getOwnPropertySymbols) {
    return names.concat(Object.getOwnPropertySymbols(events));
  }

  return names;
};

/**
 * Return the listeners registered for a given event.
 *
 * @param {(String|Symbol)} event The event name.
 * @returns {Array} The registered listeners.
 * @public
 */
EventEmitter.prototype.listeners = function listeners(event) {
  var evt = prefix ? prefix + event : event
    , handlers = this._events[evt];

  if (!handlers) return [];
  if (handlers.fn) return [handlers.fn];

  for (var i = 0, l = handlers.length, ee = new Array(l); i < l; i++) {
    ee[i] = handlers[i].fn;
  }

  return ee;
};

/**
 * Return the number of listeners listening to a given event.
 *
 * @param {(String|Symbol)} event The event name.
 * @returns {Number} The number of listeners.
 * @public
 */
EventEmitter.prototype.listenerCount = function listenerCount(event) {
  var evt = prefix ? prefix + event : event
    , listeners = this._events[evt];

  if (!listeners) return 0;
  if (listeners.fn) return 1;
  return listeners.length;
};

/**
 * Calls each of the listeners registered for a given event.
 *
 * @param {(String|Symbol)} event The event name.
 * @returns {Boolean} `true` if the event had listeners, else `false`.
 * @public
 */
EventEmitter.prototype.emit = function emit(event, a1, a2, a3, a4, a5) {
  var evt = prefix ? prefix + event : event;

  if (!this._events[evt]) return false;

  var listeners = this._events[evt]
    , len = arguments.length
    , args
    , i;

  if (listeners.fn) {
    if (listeners.once) this.removeListener(event, listeners.fn, undefined, true);

    switch (len) {
      case 1: return listeners.fn.call(listeners.context), true;
      case 2: return listeners.fn.call(listeners.context, a1), true;
      case 3: return listeners.fn.call(listeners.context, a1, a2), true;
      case 4: return listeners.fn.call(listeners.context, a1, a2, a3), true;
      case 5: return listeners.fn.call(listeners.context, a1, a2, a3, a4), true;
      case 6: return listeners.fn.call(listeners.context, a1, a2, a3, a4, a5), true;
    }

    for (i = 1, args = new Array(len -1); i < len; i++) {
      args[i - 1] = arguments[i];
    }

    listeners.fn.apply(listeners.context, args);
  } else {
    var length = listeners.length
      , j;

    for (i = 0; i < length; i++) {
      if (listeners[i].once) this.removeListener(event, listeners[i].fn, undefined, true);

      switch (len) {
        case 1: listeners[i].fn.call(listeners[i].context); break;
        case 2: listeners[i].fn.call(listeners[i].context, a1); break;
        case 3: listeners[i].fn.call(listeners[i].context, a1, a2); break;
        case 4: listeners[i].fn.call(listeners[i].context, a1, a2, a3); break;
        default:
          if (!args) for (j = 1, args = new Array(len -1); j < len; j++) {
            args[j - 1] = arguments[j];
          }

          listeners[i].fn.apply(listeners[i].context, args);
      }
    }
  }

  return true;
};

/**
 * Add a listener for a given event.
 *
 * @param {(String|Symbol)} event The event name.
 * @param {Function} fn The listener function.
 * @param {*} [context=this] The context to invoke the listener with.
 * @returns {EventEmitter} `this`.
 * @public
 */
EventEmitter.prototype.on = function on(event, fn, context) {
  return addListener(this, event, fn, context, false);
};

/**
 * Add a one-time listener for a given event.
 *
 * @param {(String|Symbol)} event The event name.
 * @param {Function} fn The listener function.
 * @param {*} [context=this] The context to invoke the listener with.
 * @returns {EventEmitter} `this`.
 * @public
 */
EventEmitter.prototype.once = function once(event, fn, context) {
  return addListener(this, event, fn, context, true);
};

/**
 * Remove the listeners of a given event.
 *
 * @param {(String|Symbol)} event The event name.
 * @param {Function} fn Only remove the listeners that match this function.
 * @param {*} context Only remove the listeners that have this context.
 * @param {Boolean} once Only remove one-time listeners.
 * @returns {EventEmitter} `this`.
 * @public
 */
EventEmitter.prototype.removeListener = function removeListener(event, fn, context, once) {
  var evt = prefix ? prefix + event : event;

  if (!this._events[evt]) return this;
  if (!fn) {
    clearEvent(this, evt);
    return this;
  }

  var listeners = this._events[evt];

  if (listeners.fn) {
    if (
      listeners.fn === fn &&
      (!once || listeners.once) &&
      (!context || listeners.context === context)
    ) {
      clearEvent(this, evt);
    }
  } else {
    for (var i = 0, events = [], length = listeners.length; i < length; i++) {
      if (
        listeners[i].fn !== fn ||
        (once && !listeners[i].once) ||
        (context && listeners[i].context !== context)
      ) {
        events.push(listeners[i]);
      }
    }

    //
    // Reset the array, or remove it completely if we have no more listeners.
    //
    if (events.length) this._events[evt] = events.length === 1 ? events[0] : events;
    else clearEvent(this, evt);
  }

  return this;
};

/**
 * Remove all listeners, or those of the specified event.
 *
 * @param {(String|Symbol)} [event] The event name.
 * @returns {EventEmitter} `this`.
 * @public
 */
EventEmitter.prototype.removeAllListeners = function removeAllListeners(event) {
  var evt;

  if (event) {
    evt = prefix ? prefix + event : event;
    if (this._events[evt]) clearEvent(this, evt);
  } else {
    this._events = new Events();
    this._eventsCount = 0;
  }

  return this;
};

//
// Alias methods names because people roll like that.
//
EventEmitter.prototype.off = EventEmitter.prototype.removeListener;
EventEmitter.prototype.addListener = EventEmitter.prototype.on;

//
// Expose the prefix.
//
EventEmitter.prefixed = prefix;

//
// Allow `EventEmitter` to be imported as module namespace.
//
EventEmitter.EventEmitter = EventEmitter;

//
// Expose the module.
//
{
  module.exports = EventEmitter;
}
}(eventemitter3));

var EventEmitter = eventemitter3.exports;

class TimeoutError extends Error {
	constructor(message) {
		super(message);
		this.name = 'TimeoutError';
	}
}

function pTimeout(promise, milliseconds, fallback, options) {
	let timer;
	const cancelablePromise = new Promise((resolve, reject) => {
		if (typeof milliseconds !== 'number' || milliseconds < 0) {
			throw new TypeError('Expected `milliseconds` to be a positive number');
		}

		if (milliseconds === Number.POSITIVE_INFINITY) {
			resolve(promise);
			return;
		}

		options = {
			customTimers: {setTimeout, clearTimeout},
			...options
		};

		timer = options.customTimers.setTimeout.call(undefined, () => {
			if (typeof fallback === 'function') {
				try {
					resolve(fallback());
				} catch (error) {
					reject(error);
				}

				return;
			}

			const message = typeof fallback === 'string' ? fallback : `Promise timed out after ${milliseconds} milliseconds`;
			const timeoutError = fallback instanceof Error ? fallback : new TimeoutError(message);

			if (typeof promise.cancel === 'function') {
				promise.cancel();
			}

			reject(timeoutError);
		}, milliseconds);

		(async () => {
			try {
				resolve(await promise);
			} catch (error) {
				reject(error);
			} finally {
				options.customTimers.clearTimeout.call(undefined, timer);
			}
		})();
	});

	cancelablePromise.clear = () => {
		clearTimeout(timer);
		timer = undefined;
	};

	return cancelablePromise;
}

// Port of lower_bound from https://en.cppreference.com/w/cpp/algorithm/lower_bound
// Used to compute insertion index to keep queue sorted after insertion
function lowerBound(array, value, comparator) {
    let first = 0;
    let count = array.length;
    while (count > 0) {
        const step = Math.trunc(count / 2);
        let it = first + step;
        if (comparator(array[it], value) <= 0) {
            first = ++it;
            count -= step + 1;
        }
        else {
            count = step;
        }
    }
    return first;
}

class PriorityQueue {
    constructor() {
        Object.defineProperty(this, "_queue", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
    }
    enqueue(run, options) {
        var _a;
        options = {
            priority: 0,
            ...options
        };
        const element = {
            priority: options.priority,
            run
        };
        if (this.size && ((_a = this._queue[this.size - 1]) === null || _a === void 0 ? void 0 : _a.priority) >= options.priority) {
            this._queue.push(element);
            return;
        }
        const index = lowerBound(this._queue, element, (a, b) => b.priority - a.priority);
        this._queue.splice(index, 0, element);
    }
    dequeue() {
        const item = this._queue.shift();
        return item === null || item === void 0 ? void 0 : item.run;
    }
    filter(options) {
        return this._queue.filter((element) => element.priority === options.priority).map((element) => element.run);
    }
    get size() {
        return this._queue.length;
    }
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
const empty$1 = () => { };
const timeoutError = new TimeoutError();
/**
Promise queue with concurrency control.
*/
class PQueue extends EventEmitter {
    constructor(options) {
        var _a, _b, _c, _d;
        super();
        Object.defineProperty(this, "_carryoverConcurrencyCount", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_isIntervalIgnored", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_intervalCount", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "_intervalCap", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_interval", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_intervalEnd", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "_intervalId", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_timeoutId", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_queue", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_queueClass", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_pendingCount", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        // The `!` is needed because of https://github.com/microsoft/TypeScript/issues/32194
        Object.defineProperty(this, "_concurrency", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_isPaused", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_resolveEmpty", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: empty$1
        });
        Object.defineProperty(this, "_resolveIdle", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: empty$1
        });
        Object.defineProperty(this, "_timeout", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "_throwOnTimeout", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
        options = {
            carryoverConcurrencyCount: false,
            intervalCap: Number.POSITIVE_INFINITY,
            interval: 0,
            concurrency: Number.POSITIVE_INFINITY,
            autoStart: true,
            queueClass: PriorityQueue,
            ...options
        };
        if (!(typeof options.intervalCap === 'number' && options.intervalCap >= 1)) {
            throw new TypeError(`Expected \`intervalCap\` to be a number from 1 and up, got \`${(_b = (_a = options.intervalCap) === null || _a === void 0 ? void 0 : _a.toString()) !== null && _b !== void 0 ? _b : ''}\` (${typeof options.intervalCap})`);
        }
        if (options.interval === undefined || !(Number.isFinite(options.interval) && options.interval >= 0)) {
            throw new TypeError(`Expected \`interval\` to be a finite number >= 0, got \`${(_d = (_c = options.interval) === null || _c === void 0 ? void 0 : _c.toString()) !== null && _d !== void 0 ? _d : ''}\` (${typeof options.interval})`);
        }
        this._carryoverConcurrencyCount = options.carryoverConcurrencyCount;
        this._isIntervalIgnored = options.intervalCap === Number.POSITIVE_INFINITY || options.interval === 0;
        this._intervalCap = options.intervalCap;
        this._interval = options.interval;
        this._queue = new options.queueClass();
        this._queueClass = options.queueClass;
        this.concurrency = options.concurrency;
        this._timeout = options.timeout;
        this._throwOnTimeout = options.throwOnTimeout === true;
        this._isPaused = options.autoStart === false;
    }
    get _doesIntervalAllowAnother() {
        return this._isIntervalIgnored || this._intervalCount < this._intervalCap;
    }
    get _doesConcurrentAllowAnother() {
        return this._pendingCount < this._concurrency;
    }
    _next() {
        this._pendingCount--;
        this._tryToStartAnother();
        this.emit('next');
    }
    _resolvePromises() {
        this._resolveEmpty();
        this._resolveEmpty = empty$1;
        if (this._pendingCount === 0) {
            this._resolveIdle();
            this._resolveIdle = empty$1;
            this.emit('idle');
        }
    }
    _onResumeInterval() {
        this._onInterval();
        this._initializeIntervalIfNeeded();
        this._timeoutId = undefined;
    }
    _isIntervalPaused() {
        const now = Date.now();
        if (this._intervalId === undefined) {
            const delay = this._intervalEnd - now;
            if (delay < 0) {
                // Act as the interval was done
                // We don't need to resume it here because it will be resumed on line 160
                this._intervalCount = (this._carryoverConcurrencyCount) ? this._pendingCount : 0;
            }
            else {
                // Act as the interval is pending
                if (this._timeoutId === undefined) {
                    this._timeoutId = setTimeout(() => {
                        this._onResumeInterval();
                    }, delay);
                }
                return true;
            }
        }
        return false;
    }
    _tryToStartAnother() {
        if (this._queue.size === 0) {
            // We can clear the interval ("pause")
            // Because we can redo it later ("resume")
            if (this._intervalId) {
                clearInterval(this._intervalId);
            }
            this._intervalId = undefined;
            this._resolvePromises();
            return false;
        }
        if (!this._isPaused) {
            const canInitializeInterval = !this._isIntervalPaused();
            if (this._doesIntervalAllowAnother && this._doesConcurrentAllowAnother) {
                const job = this._queue.dequeue();
                if (!job) {
                    return false;
                }
                this.emit('active');
                job();
                if (canInitializeInterval) {
                    this._initializeIntervalIfNeeded();
                }
                return true;
            }
        }
        return false;
    }
    _initializeIntervalIfNeeded() {
        if (this._isIntervalIgnored || this._intervalId !== undefined) {
            return;
        }
        this._intervalId = setInterval(() => {
            this._onInterval();
        }, this._interval);
        this._intervalEnd = Date.now() + this._interval;
    }
    _onInterval() {
        if (this._intervalCount === 0 && this._pendingCount === 0 && this._intervalId) {
            clearInterval(this._intervalId);
            this._intervalId = undefined;
        }
        this._intervalCount = this._carryoverConcurrencyCount ? this._pendingCount : 0;
        this._processQueue();
    }
    /**
    Executes all queued functions until it reaches the limit.
    */
    _processQueue() {
        // eslint-disable-next-line no-empty
        while (this._tryToStartAnother()) { }
    }
    get concurrency() {
        return this._concurrency;
    }
    set concurrency(newConcurrency) {
        if (!(typeof newConcurrency === 'number' && newConcurrency >= 1)) {
            throw new TypeError(`Expected \`concurrency\` to be a number from 1 and up, got \`${newConcurrency}\` (${typeof newConcurrency})`);
        }
        this._concurrency = newConcurrency;
        this._processQueue();
    }
    /**
    Adds a sync or async task to the queue. Always returns a promise.
    */
    async add(fn, options = {}) {
        return new Promise((resolve, reject) => {
            const run = async () => {
                this._pendingCount++;
                this._intervalCount++;
                try {
                    const operation = (this._timeout === undefined && options.timeout === undefined) ? fn() : pTimeout(Promise.resolve(fn()), (options.timeout === undefined ? this._timeout : options.timeout), () => {
                        if (options.throwOnTimeout === undefined ? this._throwOnTimeout : options.throwOnTimeout) {
                            reject(timeoutError);
                        }
                        return undefined;
                    });
                    const result = await operation;
                    resolve(result);
                    this.emit('completed', result);
                }
                catch (error) {
                    reject(error);
                    this.emit('error', error);
                }
                this._next();
            };
            this._queue.enqueue(run, options);
            this._tryToStartAnother();
            this.emit('add');
        });
    }
    /**
    Same as `.add()`, but accepts an array of sync or async functions.

    @returns A promise that resolves when all functions are resolved.
    */
    async addAll(functions, options) {
        return Promise.all(functions.map(async (function_) => this.add(function_, options)));
    }
    /**
    Start (or resume) executing enqueued tasks within concurrency limit. No need to call this if queue is not paused (via `options.autoStart = false` or by `.pause()` method.)
    */
    start() {
        if (!this._isPaused) {
            return this;
        }
        this._isPaused = false;
        this._processQueue();
        return this;
    }
    /**
    Put queue execution on hold.
    */
    pause() {
        this._isPaused = true;
    }
    /**
    Clear the queue.
    */
    clear() {
        this._queue = new this._queueClass();
    }
    /**
    Can be called multiple times. Useful if you for example add additional items at a later time.

    @returns A promise that settles when the queue becomes empty.
    */
    async onEmpty() {
        // Instantly resolve if the queue is empty
        if (this._queue.size === 0) {
            return;
        }
        return new Promise(resolve => {
            const existingResolve = this._resolveEmpty;
            this._resolveEmpty = () => {
                existingResolve();
                resolve();
            };
        });
    }
    /**
    @returns A promise that settles when the queue size is less than the given limit: `queue.size < limit`.

    If you want to avoid having the queue grow beyond a certain size you can `await queue.onSizeLessThan()` before adding a new item.

    Note that this only limits the number of items waiting to start. There could still be up to `concurrency` jobs already running that this call does not include in its calculation.
    */
    async onSizeLessThan(limit) {
        // Instantly resolve if the queue is empty.
        if (this._queue.size < limit) {
            return;
        }
        return new Promise(resolve => {
            const listener = () => {
                if (this._queue.size < limit) {
                    this.removeListener('next', listener);
                    resolve();
                }
            };
            this.on('next', listener);
        });
    }
    /**
    The difference with `.onEmpty` is that `.onIdle` guarantees that all work from the queue has finished. `.onEmpty` merely signals that the queue is empty, but it could mean that some promises haven't completed yet.

    @returns A promise that settles when the queue becomes empty, and all promises have completed; `queue.size === 0 && queue.pending === 0`.
    */
    async onIdle() {
        // Instantly resolve if none pending and if nothing else is queued
        if (this._pendingCount === 0 && this._queue.size === 0) {
            return;
        }
        return new Promise(resolve => {
            const existingResolve = this._resolveIdle;
            this._resolveIdle = () => {
                existingResolve();
                resolve();
            };
        });
    }
    /**
    Size of the queue, the number of queued items waiting to run.
    */
    get size() {
        return this._queue.size;
    }
    /**
    Size of the queue, filtered by the given options.

    For example, this can be used to find the number of items remaining in the queue with a specific priority level.
    */
    sizeBy(options) {
        // eslint-disable-next-line unicorn/no-array-callback-reference
        return this._queue.filter(options).length;
    }
    /**
    Number of running items (no longer in the queue).
    */
    get pending() {
        return this._pendingCount;
    }
    /**
    Whether the queue is currently paused.
    */
    get isPaused() {
        return this._isPaused;
    }
    get timeout() {
        return this._timeout;
    }
    /**
    Set the timeout for future operations.
    */
    set timeout(milliseconds) {
        this._timeout = milliseconds;
    }
}

class ZarrArray {
    /**
     * Instantiate an array from an initialized store.
     * @param store Array store, already initialized.
     * @param path Storage path.
     * @param metadata The initial value for the metadata
     * @param readOnly True if array should be protected against modification.
     * @param chunkStore Separate storage for chunks. If not provided, `store` will be used for storage of both chunks and metadata.
     * @param cacheMetadata If true (default), array configuration metadata will be cached for the lifetime of the object.
     * If false, array metadata will be reloaded prior to all data access and modification operations (may incur overhead depending on storage and data access pattern).
     * @param cacheAttrs If true (default), user attributes will be cached for attribute read operations.
     * If false, user attributes are reloaded from the store prior to all attribute read operations.
     */
    constructor(store, path = null, metadata, readOnly = false, chunkStore = null, cacheMetadata = true, cacheAttrs = true) {
        // N.B., expect at this point store is fully initialized with all
        // configuration metadata fully specified and normalized
        this.store = store;
        this._chunkStore = chunkStore;
        this.path = normalizeStoragePath(path);
        this.keyPrefix = pathToPrefix(this.path);
        this.readOnly = readOnly;
        this.cacheMetadata = cacheMetadata;
        this.cacheAttrs = cacheAttrs;
        this.meta = metadata;
        if (this.meta.compressor !== null) {
            this.compressor = getCodec(this.meta.compressor);
        }
        else {
            this.compressor = null;
        }
        const attrKey = this.keyPrefix + ATTRS_META_KEY;
        this.attrs = new Attributes(this.store, attrKey, this.readOnly, cacheAttrs);
    }
    /**
     * A `Store` providing the underlying storage for array chunks.
     */
    get chunkStore() {
        if (this._chunkStore) {
            return this._chunkStore;
        }
        return this.store;
    }
    /**
     * Array name following h5py convention.
     */
    get name() {
        if (this.path.length > 0) {
            if (this.path[0] !== "/") {
                return "/" + this.path;
            }
            return this.path;
        }
        return null;
    }
    /**
     * Final component of name.
     */
    get basename() {
        const name = this.name;
        if (name === null) {
            return null;
        }
        const parts = name.split("/");
        return parts[parts.length - 1];
    }
    /**
     * "A list of integers describing the length of each dimension of the array.
     */
    get shape() {
        // this.refreshMetadata();
        return this.meta.shape;
    }
    /**
     * A list of integers describing the length of each dimension of a chunk of the array.
     */
    get chunks() {
        return this.meta.chunks;
    }
    /**
     * Integer describing how many element a chunk contains
     */
    get chunkSize() {
        return this.chunks.reduce((x, y) => x * y, 1);
    }
    /**
     *  The NumPy data type.
     */
    get dtype() {
        return this.meta.dtype;
    }
    /**
     *  A value used for uninitialized portions of the array.
     */
    get fillValue() {
        const fillTypeValue = this.meta.fill_value;
        // TODO extract into function
        if (fillTypeValue === "NaN") {
            return NaN;
        }
        else if (fillTypeValue === "Infinity") {
            return Infinity;
        }
        else if (fillTypeValue === "-Infinity") {
            return -Infinity;
        }
        return this.meta.fill_value;
    }
    /**
     *  Number of dimensions.
     */
    get nDims() {
        return this.meta.shape.length;
    }
    /**
     *  The total number of elements in the array.
     */
    get size() {
        // this.refreshMetadata()
        return this.meta.shape.reduce((x, y) => x * y, 1);
    }
    get length() {
        return this.shape[0];
    }
    get _chunkDataShape() {
        if (this.shape.length === 0) {
            return [1];
        }
        else {
            const s = [];
            for (let i = 0; i < this.shape.length; i++) {
                s[i] = Math.ceil(this.shape[i] / this.chunks[i]);
            }
            return s;
        }
    }
    /**
     * A tuple of integers describing the number of chunks along each
     * dimension of the array.
     */
    get chunkDataShape() {
        // this.refreshMetadata();
        return this._chunkDataShape;
    }
    /**
     * Total number of chunks.
     */
    get numChunks() {
        // this.refreshMetadata();
        return this.chunkDataShape.reduce((x, y) => x * y, 1);
    }
    /**
     * Instantiate an array from an initialized store.
     * @param store Array store, already initialized.
     * @param path Storage path.
     * @param readOnly True if array should be protected against modification.
     * @param chunkStore Separate storage for chunks. If not provided, `store` will be used for storage of both chunks and metadata.
     * @param cacheMetadata If true (default), array configuration metadata will be cached for the lifetime of the object.
     * If false, array metadata will be reloaded prior to all data access and modification operations (may incur overhead depending on storage and data access pattern).
     * @param cacheAttrs If true (default), user attributes will be cached for attribute read operations.
     * If false, user attributes are reloaded from the store prior to all attribute read operations.
     */
    static async create(store, path = null, readOnly = false, chunkStore = null, cacheMetadata = true, cacheAttrs = true) {
        const metadata = await this.loadMetadataForConstructor(store, path);
        return new ZarrArray(store, path, metadata, readOnly, chunkStore, cacheMetadata, cacheAttrs);
    }
    static async loadMetadataForConstructor(store, path) {
        try {
            path = normalizeStoragePath(path);
            const keyPrefix = pathToPrefix(path);
            const metaStoreValue = await store.getItem(keyPrefix + ARRAY_META_KEY);
            return parseMetadata(metaStoreValue);
        }
        catch (error) {
            if (await containsGroup(store, path)) {
                throw new ContainsGroupError(path !== null && path !== void 0 ? path : '');
            }
            throw new Error("Failed to load metadata for ZarrArray:" + error.toString());
        }
    }
    /**
     * (Re)load metadata from store
     */
    async reloadMetadata() {
        const metaKey = this.keyPrefix + ARRAY_META_KEY;
        const metaStoreValue = this.store.getItem(metaKey);
        this.meta = parseMetadata(await metaStoreValue);
        return this.meta;
    }
    async refreshMetadata() {
        if (!this.cacheMetadata) {
            await this.reloadMetadata();
        }
    }
    get(selection = null, opts = {}) {
        return this.getBasicSelection(selection, false, opts);
    }
    getRaw(selection = null, opts = {}) {
        return this.getBasicSelection(selection, true, opts);
    }
    async getBasicSelection(selection, asRaw = false, { concurrencyLimit = 10, progressCallback } = {}) {
        // Refresh metadata
        if (!this.cacheMetadata) {
            await this.reloadMetadata();
        }
        // Check fields (TODO?)
        if (this.shape.length === 0) {
            throw new Error("Shape [] indexing is not supported yet");
        }
        else {
            return this.getBasicSelectionND(selection, asRaw, concurrencyLimit, progressCallback);
        }
    }
    getBasicSelectionND(selection, asRaw, concurrencyLimit, progressCallback) {
        const indexer = new BasicIndexer(selection, this);
        return this.getSelection(indexer, asRaw, concurrencyLimit, progressCallback);
    }
    async getSelection(indexer, asRaw, concurrencyLimit, progressCallback) {
        // We iterate over all chunks which overlap the selection and thus contain data
        // that needs to be extracted. Each chunk is processed in turn, extracting the
        // necessary data and storing into the correct location in the output array.
        // N.B., it is an important optimisation that we only visit chunks which overlap
        // the selection. This minimises the number of iterations in the main for loop.
        // check fields are sensible (TODO?)
        const outDtype = this.dtype;
        const outShape = indexer.shape;
        const outSize = indexer.shape.reduce((x, y) => x * y, 1);
        if (asRaw && (outSize === this.chunkSize)) {
            // Optimization: if output strided array _is_ chunk exactly,
            // decode directly as new TypedArray and return
            const itr = indexer.iter();
            const proj = itr.next(); // ensure there is only one projection
            if (proj.done === false && itr.next().done === true) {
                const chunkProjection = proj.value;
                const out = await this.decodeDirectToRawArray(chunkProjection, outShape, outSize);
                return out;
            }
        }
        const out = asRaw
            ? new RawArray(null, outShape, outDtype)
            : new NestedArray(null, outShape, outDtype);
        if (outSize === 0) {
            return out;
        }
        // create promise queue with concurrency control
        const queue = new PQueue({ concurrency: concurrencyLimit });
        if (progressCallback) {
            let progress = 0;
            let queueSize = 0;
            for (const _ of indexer.iter())
                queueSize += 1;
            progressCallback({ progress: 0, queueSize: queueSize });
            for (const proj of indexer.iter()) {
                (async () => {
                    await queue.add(() => this.chunkGetItem(proj.chunkCoords, proj.chunkSelection, out, proj.outSelection, indexer.dropAxes));
                    progress += 1;
                    progressCallback({ progress: progress, queueSize: queueSize });
                })();
            }
        }
        else {
            for (const proj of indexer.iter()) {
                queue.add(() => this.chunkGetItem(proj.chunkCoords, proj.chunkSelection, out, proj.outSelection, indexer.dropAxes));
            }
        }
        // guarantees that all work on queue has finished
        await queue.onIdle();
        // Return scalar instead of zero-dimensional array.
        if (out.shape.length === 0) {
            return out.data[0];
        }
        return out;
    }
    /**
     * Obtain part or whole of a chunk.
     * @param chunkCoords Indices of the chunk.
     * @param chunkSelection Location of region within the chunk to extract.
     * @param out Array to store result in.
     * @param outSelection Location of region within output array to store results in.
     * @param dropAxes Axes to squeeze out of the chunk.
     */
    async chunkGetItem(chunkCoords, chunkSelection, out, outSelection, dropAxes) {
        if (chunkCoords.length !== this._chunkDataShape.length) {
            throw new ValueError(`Inconsistent shapes: chunkCoordsLength: ${chunkCoords.length}, cDataShapeLength: ${this.chunkDataShape.length}`);
        }
        const cKey = this.chunkKey(chunkCoords);
        try {
            const cdata = await this.chunkStore.getItem(cKey);
            const decodedChunk = await this.decodeChunk(cdata);
            if (out instanceof NestedArray) {
                if (isContiguousSelection(outSelection) && isTotalSlice(chunkSelection, this.chunks) && !this.meta.filters) {
                    // Optimization: we want the whole chunk, and the destination is
                    // contiguous, so we can decompress directly from the chunk
                    // into the destination array
                    // TODO check order
                    // TODO filters...
                    out.set(outSelection, this.toNestedArray(decodedChunk));
                    return;
                }
                // Decode chunk
                const chunk = this.toNestedArray(decodedChunk);
                const tmp = chunk.get(chunkSelection);
                if (dropAxes !== null) {
                    throw new Error("Drop axes is not supported yet");
                }
                out.set(outSelection, tmp);
            }
            else {
                /* RawArray
                Copies chunk by index directly into output. Doesn't matter if selection is contiguous
                since store/output are different shapes/strides.
                */
                out.set(outSelection, this.chunkBufferToRawArray(decodedChunk), chunkSelection);
            }
        }
        catch (error) {
            if (isKeyError(error)) {
                // fill with scalar if cKey doesn't exist in store
                if (this.fillValue !== null) {
                    out.set(outSelection, this.fillValue);
                }
            }
            else {
                // Different type of error - rethrow
                throw error;
            }
        }
    }
    async getRawChunk(chunkCoords, opts) {
        if (chunkCoords.length !== this.shape.length) {
            throw new Error(`Chunk coordinates ${chunkCoords.join(".")} do not correspond to shape ${this.shape}.`);
        }
        try {
            for (let i = 0; i < chunkCoords.length; i++) {
                const dimLength = Math.ceil(this.shape[i] / this.chunks[i]);
                chunkCoords[i] = normalizeIntegerSelection(chunkCoords[i], dimLength);
            }
        }
        catch (error) {
            if (error instanceof BoundsCheckError) {
                throw new BoundsCheckError(`index ${chunkCoords.join(".")} is out of bounds for shape: ${this.shape} and chunks ${this.chunks}`);
            }
            else {
                throw error;
            }
        }
        const cKey = this.chunkKey(chunkCoords);
        const cdata = this.chunkStore.getItem(cKey, opts === null || opts === void 0 ? void 0 : opts.storeOptions);
        const buffer = await this.decodeChunk(await cdata);
        const outShape = this.chunks.filter(d => d !== 1); // squeeze chunk dim if 1
        return new RawArray(buffer, outShape, this.dtype);
    }
    chunkKey(chunkCoords) {
        var _a;
        const sep = (_a = this.meta.dimension_separator) !== null && _a !== void 0 ? _a : ".";
        return this.keyPrefix + chunkCoords.join(sep);
    }
    ensureByteArray(chunkData) {
        if (typeof chunkData === "string") {
            return new Uint8Array(Buffer.from(chunkData).buffer);
        }
        return new Uint8Array(chunkData);
    }
    toTypedArray(buffer) {
        return new (getTypedArrayCtr(this.dtype))(buffer);
    }
    toNestedArray(data) {
        const buffer = this.ensureByteArray(data).buffer;
        return new NestedArray(buffer, this.chunks, this.dtype);
    }
    async decodeChunk(chunkData) {
        let bytes = this.ensureByteArray(chunkData);
        if (this.compressor !== null) {
            bytes = await (await this.compressor).decode(bytes);
        }
        if (this.dtype.includes('>')) {
            // Need to flip bytes for Javascript TypedArrays
            // We flip bytes in-place to avoid creating an extra copy of the decoded buffer.
            byteSwapInplace(this.toTypedArray(bytes.buffer));
        }
        if (this.meta.order === "F" && this.nDims > 1) {
            // We need to transpose the array, because this library only support C-order.
            const src = this.toTypedArray(bytes.buffer);
            const out = new (getTypedArrayCtr(this.dtype))(src.length);
            convertColMajorToRowMajor(src, out, this.chunks);
            return out.buffer;
        }
        // TODO filtering etc
        return bytes.buffer;
    }
    chunkBufferToRawArray(buffer) {
        return new RawArray(buffer, this.chunks, this.dtype);
    }
    async decodeDirectToRawArray({ chunkCoords }, outShape, outSize) {
        const cKey = this.chunkKey(chunkCoords);
        try {
            const cdata = await this.chunkStore.getItem(cKey);
            return new RawArray(await this.decodeChunk(cdata), outShape, this.dtype);
        }
        catch (error) {
            if (isKeyError(error)) {
                // fill with scalar if item doesn't exist
                const data = new (getTypedArrayCtr(this.dtype))(outSize);
                return new RawArray(data.fill(this.fillValue), outShape);
            }
            else {
                // Different type of error - rethrow
                throw error;
            }
        }
    }
    async set(selection = null, value, opts = {}) {
        await this.setBasicSelection(selection, value, opts);
    }
    async setBasicSelection(selection, value, { concurrencyLimit = 10, progressCallback } = {}) {
        if (this.readOnly) {
            throw new PermissionError("Object is read only");
        }
        if (!this.cacheMetadata) {
            await this.reloadMetadata();
        }
        if (this.shape.length === 0) {
            throw new Error("Shape [] indexing is not supported yet");
        }
        else {
            await this.setBasicSelectionND(selection, value, concurrencyLimit, progressCallback);
        }
    }
    async setBasicSelectionND(selection, value, concurrencyLimit, progressCallback) {
        const indexer = new BasicIndexer(selection, this);
        await this.setSelection(indexer, value, concurrencyLimit, progressCallback);
    }
    getChunkValue(proj, indexer, value, selectionShape) {
        let chunkValue;
        if (selectionShape.length === 0) {
            chunkValue = value;
        }
        else if (typeof value === "number" || typeof value === "bigint") {
            chunkValue = value;
        }
        else {
            chunkValue = value.get(proj.outSelection);
            // tslint:disable-next-line: strict-type-predicates
            if (indexer.dropAxes !== null) {
                throw new Error("Handling drop axes not supported yet");
            }
        }
        return chunkValue;
    }
    async setSelection(indexer, value, concurrencyLimit, progressCallback) {
        // We iterate over all chunks which overlap the selection and thus contain data
        // that needs to be replaced. Each chunk is processed in turn, extracting the
        // necessary data from the value array and storing into the chunk array.
        // N.B., it is an important optimisation that we only visit chunks which overlap
        // the selection. This minimises the number of iterations in the main for loop.
        // TODO? check fields are sensible
        // Determine indices of chunks overlapping the selection
        const selectionShape = indexer.shape;
        // Check value shape
        if (selectionShape.length === 0) ;
        else if (typeof value === "number" || typeof value === "bigint") ;
        else if (value instanceof NestedArray) {
            // TODO: non stringify equality check
            if (!arrayEquals1D(value.shape, selectionShape)) {
                throw new ValueError(`Shape mismatch in source NestedArray and set selection: ${value.shape} and ${selectionShape}`);
            }
        }
        else {
            // TODO support TypedArrays, buffers, etc
            throw new Error("Unknown data type for setting :(");
        }
        const queue = new PQueue({ concurrency: concurrencyLimit });
        if (progressCallback) {
            let queueSize = 0;
            for (const _ of indexer.iter())
                queueSize += 1;
            let progress = 0;
            progressCallback({ progress: 0, queueSize: queueSize });
            for (const proj of indexer.iter()) {
                const chunkValue = this.getChunkValue(proj, indexer, value, selectionShape);
                (async () => {
                    await queue.add(() => this.chunkSetItem(proj.chunkCoords, proj.chunkSelection, chunkValue));
                    progress += 1;
                    progressCallback({ progress: progress, queueSize: queueSize });
                })();
            }
        }
        else {
            for (const proj of indexer.iter()) {
                const chunkValue = this.getChunkValue(proj, indexer, value, selectionShape);
                queue.add(() => this.chunkSetItem(proj.chunkCoords, proj.chunkSelection, chunkValue));
            }
        }
        // guarantees that all work on queue has finished
        await queue.onIdle();
    }
    async chunkSetItem(chunkCoords, chunkSelection, value) {
        if (this.meta.order === "F" && this.nDims > 1) {
            throw new Error("Setting content for arrays in F-order is not supported.");
        }
        // Obtain key for chunk storage
        const chunkKey = this.chunkKey(chunkCoords);
        let chunk = null;
        const dtypeConstr = getTypedArrayCtr(this.dtype);
        const chunkSize = this.chunkSize;
        if (isTotalSlice(chunkSelection, this.chunks)) {
            // Totally replace chunk
            // Optimization: we are completely replacing the chunk, so no need
            // to access the existing chunk data
            if (typeof value === "number" || typeof value === "bigint") {
                // TODO get the right type here
                chunk = new dtypeConstr(chunkSize);
                chunk.fill(value);
            }
            else {
                chunk = value.flatten();
            }
        }
        else {
            // partially replace the contents of this chunk
            // Existing chunk data
            let chunkData;
            try {
                // Chunk is initialized if this does not error
                const chunkStoreData = await this.chunkStore.getItem(chunkKey);
                const dBytes = await this.decodeChunk(chunkStoreData);
                chunkData = this.toTypedArray(dBytes);
            }
            catch (error) {
                if (isKeyError(error)) {
                    // Chunk is not initialized
                    chunkData = new dtypeConstr(chunkSize);
                    if (this.fillValue !== null) {
                        chunkData.fill(this.fillValue);
                    }
                }
                else {
                    // Different type of error - rethrow
                    throw error;
                }
            }
            const chunkNestedArray = new NestedArray(chunkData, this.chunks, this.dtype);
            chunkNestedArray.set(chunkSelection, value);
            chunk = chunkNestedArray.flatten();
        }
        const chunkData = await this.encodeChunk(chunk);
        this.chunkStore.setItem(chunkKey, chunkData);
    }
    async encodeChunk(chunk) {
        if (this.dtype.includes('>')) {
            /*
             * If big endian, flip bytes before applying compression and setting store.
             *
             * Here we create a copy (not in-place byteswapping) to avoid flipping the
             * bytes in the buffers of user-created Raw- and NestedArrays.
            */
            chunk = byteSwap(chunk);
        }
        if (this.compressor !== null) {
            const bytes = new Uint8Array(chunk.buffer);
            const cbytes = await (await this.compressor).encode(bytes);
            return cbytes.buffer;
        }
        // TODO: filters, etc
        return chunk.buffer;
    }
}

class MemoryStore {
    constructor(root = {}) {
        this.root = root;
    }
    proxy() {
        return createProxy(this);
    }
    getParent(item) {
        let parent = this.root;
        const segments = item.split('/');
        // find the parent container
        for (const k of segments.slice(0, segments.length - 1)) {
            parent = parent[k];
            if (!parent) {
                throw Error(item);
            }
            // if not isinstance(parent, self.cls):
            //     raise KeyError(item)
        }
        return [parent, segments[segments.length - 1]];
    }
    requireParent(item) {
        let parent = this.root;
        const segments = item.split('/');
        // require the parent container
        for (const k of segments.slice(0, segments.length - 1)) {
            // TODO: verify correct implementation
            if (parent[k] === undefined) {
                parent[k] = {};
            }
            parent = parent[k];
        }
        return [parent, segments[segments.length - 1]];
    }
    getItem(item) {
        const [parent, key] = this.getParent(item);
        const value = parent[key];
        if (value === undefined) {
            throw new KeyError(item);
        }
        return value;
    }
    setItem(item, value) {
        const [parent, key] = this.requireParent(item);
        parent[key] = value;
        return true;
    }
    deleteItem(item) {
        const [parent, key] = this.getParent(item);
        return delete parent[key];
    }
    containsItem(item) {
        // TODO: more sane implementation
        try {
            return this.getItem(item) !== undefined;
        }
        catch (e) {
            return false;
        }
    }
    keys() {
        throw new Error("Method not implemented.");
    }
}

var HTTPMethod;
(function (HTTPMethod) {
    HTTPMethod["HEAD"] = "HEAD";
    HTTPMethod["GET"] = "GET";
    HTTPMethod["PUT"] = "PUT";
})(HTTPMethod || (HTTPMethod = {}));
const DEFAULT_METHODS = [HTTPMethod.HEAD, HTTPMethod.GET, HTTPMethod.PUT];
class HTTPStore {
    constructor(url, options = {}) {
        this.url = url;
        const { fetchOptions = {}, supportedMethods = DEFAULT_METHODS } = options;
        this.fetchOptions = fetchOptions;
        this.supportedMethods = new Set(supportedMethods);
    }
    keys() {
        throw new Error('Method not implemented.');
    }
    async getItem(item, opts) {
        const url = resolveUrl(this.url, item);
        const value = await fetch(url, { ...this.fetchOptions, ...opts });
        if (value.status === 404) {
            // Item is not found
            throw new KeyError(item);
        }
        else if (value.status !== 200) {
            throw new HTTPError(String(value.status));
        }
        // only decode if 200
        if (IS_NODE) {
            return Buffer.from(await value.arrayBuffer());
        }
        else {
            return value.arrayBuffer(); // Browser
        }
    }
    async setItem(item, value) {
        if (!this.supportedMethods.has(HTTPMethod.PUT)) {
            throw new Error('HTTP PUT no a supported method for store.');
        }
        const url = resolveUrl(this.url, item);
        if (typeof value === 'string') {
            value = new TextEncoder().encode(value).buffer;
        }
        const set = await fetch(url, { ...this.fetchOptions, method: HTTPMethod.PUT, body: value });
        return set.status.toString()[0] === '2';
    }
    deleteItem(_item) {
        throw new Error('Method not implemented.');
    }
    async containsItem(item) {
        const url = resolveUrl(this.url, item);
        // Just check headers if HEAD method supported
        const method = this.supportedMethods.has(HTTPMethod.HEAD) ? HTTPMethod.HEAD : HTTPMethod.GET;
        const value = await fetch(url, { ...this.fetchOptions, method });
        return value.status === 200;
    }
}

/**
 *
 * @param shape Array shape.
 * @param chunks  Chunk shape. If `true`, will be guessed from `shape` and `dtype`. If
 *      `false`, will be set to `shape`, i.e., single chunk for the whole array.
 *      If an int, the chunk size in each dimension will be given by the value
 *      of `chunks`. Default is `true`.
 * @param dtype NumPy dtype.
 * @param compressor Primary compressor.
 * @param fillValue Default value to use for uninitialized portions of the array.
 * @param order Memory layout to be used within each chunk.
 * @param store Store or path to directory in file system or name of zip file.
 * @param overwrite  If True, delete all pre-existing data in `store` at `path` before creating the array.
 * @param path Path under which array is stored.
 * @param chunkStore Separate storage for chunks. If not provided, `store` will be used for storage of both chunks and metadata.
 * @param filters Sequence of filters to use to encode chunk data prior to compression.
 * @param cacheMetadata If `true` (default), array configuration metadata will be cached for the
 *      lifetime of the object. If `false`, array metadata will be reloaded
 *      prior to all data access and modification operations (may incur
 *      overhead depending on storage and data access pattern).
 * @param cacheAttrs If `true` (default), user attributes will be cached for attribute read
 *      operations. If `false`, user attributes are reloaded from the store prior
 *      to all attribute read operations.
 * @param readOnly `true` if array should be protected against modification, defaults to `false`.
 * @param dimensionSeparator if specified, defines an alternate string separator placed between the dimension chunks.
 */
async function create({ shape, chunks = true, dtype = "<i4", compressor = null, fillValue = null, order = "C", store, overwrite = false, path, chunkStore, filters, cacheMetadata = true, cacheAttrs = true, readOnly = false, dimensionSeparator }) {
    store = normalizeStoreArgument(store);
    await initArray(store, shape, chunks, dtype, path, compressor, fillValue, order, overwrite, chunkStore, filters, dimensionSeparator);
    const z = await ZarrArray.create(store, path, readOnly, chunkStore, cacheMetadata, cacheAttrs);
    return z;
}
/**
 * Create an empty array.
 */
async function empty(shape, opts = {}) {
    opts.fillValue = null;
    return create({ shape, ...opts });
}
/**
 * Create an array, with zero being used as the default value for
 * uninitialized portions of the array.
 */
async function zeros(shape, opts = {}) {
    opts.fillValue = 0;
    return create({ shape, ...opts });
}
/**
 * Create an array, with one being used as the default value for
 * uninitialized portions of the array.
 */
async function ones(shape, opts = {}) {
    opts.fillValue = 1;
    return create({ shape, ...opts });
}
/**
 * Create an array, with `fill_value` being used as the default value for
 * uninitialized portions of the array
 */
async function full(shape, fillValue, opts = {}) {
    opts.fillValue = fillValue;
    return create({ shape, ...opts });
}
async function array(data, opts = {}) {
    // TODO: infer chunks?
    let shape = null;
    if (data instanceof NestedArray) {
        shape = data.shape;
        opts.dtype = opts.dtype === undefined ? data.dtype : opts.dtype;
    }
    else {
        shape = data.byteLength;
        // TODO: infer datatype
    }
    // TODO: support TypedArray
    const wasReadOnly = opts.readOnly === undefined ? false : opts.readOnly;
    opts.readOnly = false;
    const z = await create({ shape, ...opts });
    await z.set(null, data);
    z.readOnly = wasReadOnly;
    return z;
}
async function openArray({ shape, mode = "a", chunks = true, dtype = "<i4", compressor = null, fillValue = null, order = "C", store, overwrite = false, path = null, chunkStore, filters, cacheMetadata = true, cacheAttrs = true, dimensionSeparator } = {}) {
    store = normalizeStoreArgument(store);
    console.log("store openArray", store);
    if (chunkStore === undefined) {
        chunkStore = normalizeStoreArgument(store);
    }
    path = normalizeStoragePath(path);
    if (mode === "r" || mode === "r+") {
        if (!await containsArray(store, path)) {
            if (await containsGroup(store, path)) {
                throw new ContainsGroupError(path);
            }
            throw new ArrayNotFoundError(path);
        }
    }
    else if (mode === "w") {
        if (shape === undefined) {
            throw new ValueError("Shape can not be undefined when creating a new array");
        }
        await initArray(store, shape, chunks, dtype, path, compressor, fillValue, order, overwrite, chunkStore, filters, dimensionSeparator);
    }
    else if (mode === "a") {
        if (!await containsArray(store, path)) {
            if (await containsGroup(store, path)) {
                throw new ContainsGroupError(path);
            }
            if (shape === undefined) {
                throw new ValueError("Shape can not be undefined when creating a new array");
            }
            await initArray(store, shape, chunks, dtype, path, compressor, fillValue, order, overwrite, chunkStore, filters, dimensionSeparator);
        }
    }
    else if (mode === "w-" || mode === "x") {
        if (await containsArray(store, path)) {
            throw new ContainsArrayError(path);
        }
        else if (await containsGroup(store, path)) {
            throw new ContainsGroupError(path);
        }
        else {
            if (shape === undefined) {
                throw new ValueError("Shape can not be undefined when creating a new array");
            }
            await initArray(store, shape, chunks, dtype, path, compressor, fillValue, order, overwrite, chunkStore, filters, dimensionSeparator);
        }
    }
    else {
        throw new ValueError(`Invalid mode argument: ${mode}`);
    }
    const readOnly = mode === "r";
    return ZarrArray.create(store, path, readOnly, chunkStore, cacheMetadata, cacheAttrs);
}
function normalizeStoreArgument(store) {
    if (store === undefined) {
        return new MemoryStore();
    }
    else if (typeof store === "string") {
        return new HTTPStore(store);
    }
    return store;
}

class Group {
    constructor(store, path = null, metadata, readOnly = false, chunkStore = null, cacheAttrs = true) {
        this.store = store;
        this._chunkStore = chunkStore;
        this.path = normalizeStoragePath(path);
        this.keyPrefix = pathToPrefix(this.path);
        this.readOnly = readOnly;
        this.meta = metadata;
        // Initialize attributes
        const attrKey = this.keyPrefix + ATTRS_META_KEY;
        this.attrs = new Attributes(this.store, attrKey, this.readOnly, cacheAttrs);
    }
    /**
     * Group name following h5py convention.
     */
    get name() {
        if (this.path.length > 0) {
            if (this.path[0] !== "/") {
                return "/" + this.path;
            }
            return this.path;
        }
        return "/";
    }
    /**
     * Final component of name.
     */
    get basename() {
        const parts = this.name.split("/");
        return parts[parts.length - 1];
    }
    /**
     * A `Store` providing the underlying storage for array chunks.
     */
    get chunkStore() {
        if (this._chunkStore) {
            return this._chunkStore;
        }
        return this.store;
    }
    static async create(store, path = null, readOnly = false, chunkStore = null, cacheAttrs = true) {
        const metadata = await this.loadMetadataForConstructor(store, path);
        return new Group(store, path, metadata, readOnly, chunkStore, cacheAttrs);
    }
    static async loadMetadataForConstructor(store, path) {
        path = normalizeStoragePath(path);
        const keyPrefix = pathToPrefix(path);
        try {
            const metaStoreValue = await store.getItem(keyPrefix + GROUP_META_KEY);
            return parseMetadata(metaStoreValue);
        }
        catch (error) {
            if (await containsArray(store, path)) {
                throw new ContainsArrayError(path);
            }
            throw new GroupNotFoundError(path);
        }
    }
    itemPath(item) {
        const absolute = typeof item === "string" && item.length > 0 && item[0] === '/';
        const path = normalizeStoragePath(item);
        // Absolute path
        if (!absolute && this.path.length > 0) {
            return this.keyPrefix + path;
        }
        return path;
    }
    /**
     * Create a sub-group.
     */
    async createGroup(name, overwrite = false) {
        if (this.readOnly) {
            throw new PermissionError("group is read only");
        }
        const path = this.itemPath(name);
        await initGroup(this.store, path, this._chunkStore, overwrite);
        return Group.create(this.store, path, this.readOnly, this._chunkStore, this.attrs.cache);
    }
    /**
     * Obtain a sub-group, creating one if it doesn't exist.
     */
    async requireGroup(name, overwrite = false) {
        if (this.readOnly) {
            throw new PermissionError("group is read only");
        }
        const path = this.itemPath(name);
        if (!await containsGroup(this.store, path)) {
            await initGroup(this.store, path, this._chunkStore, overwrite);
        }
        return Group.create(this.store, path, this.readOnly, this._chunkStore, this.attrs.cache);
    }
    getOptsForArrayCreation(name, opts = {}) {
        const path = this.itemPath(name);
        opts.path = path;
        if (opts.cacheAttrs === undefined) {
            opts.cacheAttrs = this.attrs.cache;
        }
        opts.store = this.store;
        opts.chunkStore = this.chunkStore;
        return opts;
    }
    /**
     * Creates an array
     */
    array(name, data, opts, overwrite) {
        if (this.readOnly) {
            throw new PermissionError("group is read only");
        }
        opts = this.getOptsForArrayCreation(name, opts);
        opts.overwrite = overwrite === undefined ? opts.overwrite : overwrite;
        return array(data, opts);
    }
    empty(name, shape, opts = {}) {
        if (this.readOnly) {
            throw new PermissionError("group is read only");
        }
        opts = this.getOptsForArrayCreation(name, opts);
        return empty(shape, opts);
    }
    zeros(name, shape, opts = {}) {
        if (this.readOnly) {
            throw new PermissionError("group is read only");
        }
        opts = this.getOptsForArrayCreation(name, opts);
        return zeros(shape, opts);
    }
    ones(name, shape, opts = {}) {
        if (this.readOnly) {
            throw new PermissionError("group is read only");
        }
        opts = this.getOptsForArrayCreation(name, opts);
        return ones(shape, opts);
    }
    full(name, shape, fillValue, opts = {}) {
        if (this.readOnly) {
            throw new PermissionError("group is read only");
        }
        opts = this.getOptsForArrayCreation(name, opts);
        return full(shape, fillValue, opts);
    }
    createDataset(name, shape, data, opts) {
        if (this.readOnly) {
            throw new PermissionError("group is read only");
        }
        opts = this.getOptsForArrayCreation(name, opts);
        let z;
        if (data === undefined) {
            if (shape === undefined) {
                throw new ValueError("Shape must be set if no data is passed to CreateDataset");
            }
            z = create({ shape, ...opts });
        }
        else {
            z = array(data, opts);
        }
        return z;
    }
    async getItem(item) {
        const path = this.itemPath(item);
        if (await containsArray(this.store, path)) {
            return ZarrArray.create(this.store, path, this.readOnly, this.chunkStore, undefined, this.attrs.cache);
        }
        else if (await containsGroup(this.store, path)) {
            return Group.create(this.store, path, this.readOnly, this._chunkStore, this.attrs.cache);
        }
        throw new KeyError(item);
    }
    async setItem(item, value) {
        await this.array(item, value, {}, true);
        return true;
    }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    async deleteItem(_item) {
        if (this.readOnly) {
            throw new PermissionError("group is read only");
        }
        throw new Error("Method not implemented.");
    }
    async containsItem(item) {
        const path = this.itemPath(item);
        return await containsArray(this.store, path) || containsGroup(this.store, path);
    }
    proxy() {
        return createProxy(this);
    }
}
/**
 * Create a group.
 * @param store Store or path to directory in file system.
 * @param path Group path within store.
 * @param chunkStore Separate storage for chunks. If not provided, `store` will be used for storage of both chunks and metadata.
 * @param overwrite If `true`, delete any pre-existing data in `store` at `path` before creating the group.
 * @param cacheAttrs If `true` (default), user attributes will be cached for attribute read operations.
 *   If `false`, user attributes are reloaded from the store prior to all attribute read operations.
 */
async function group(store, path = null, chunkStore, overwrite = false, cacheAttrs = true) {
    store = normalizeStoreArgument(store);
    path = normalizeStoragePath(path);
    if (overwrite || await containsGroup(store)) {
        await initGroup(store, path, chunkStore, overwrite);
    }
    return Group.create(store, path, false, chunkStore, cacheAttrs);
}
/**
 * Open a group using file-mode-like semantics.
 * @param store Store or path to directory in file system or name of zip file.
 * @param path Group path within store.
 * @param mode Persistence mode, see `PersistenceMode` type.
 * @param chunkStore Store or path to directory in file system or name of zip file.
 * @param cacheAttrs If `true` (default), user attributes will be cached for attribute read operations
 *   If False, user attributes are reloaded from the store prior to all attribute read operations.
 *
 */
async function openGroup(store, path = null, mode = "a", chunkStore, cacheAttrs = true) {
    store = normalizeStoreArgument(store);
    console.log("store openGroup", store);
    if (chunkStore !== undefined) {
        chunkStore = normalizeStoreArgument(store);
    }
    path = normalizeStoragePath(path);
    if (mode === "r" || mode === "r+") {
        if (!await containsGroup(store, path)) {
            if (await containsArray(store, path)) {
                throw new ContainsArrayError(path);
            }
            throw new GroupNotFoundError(path);
        }
    }
    else if (mode === "w") {
        await initGroup(store, path, chunkStore, true);
    }
    else if (mode === "a") {
        if (!await containsGroup(store, path)) {
            if (await containsArray(store, path)) {
                throw new ContainsArrayError(path);
            }
            await initGroup(store, path, chunkStore);
        }
    }
    else if (mode === "w-" || mode === "x") {
        if (await containsArray(store, path)) {
            throw new ContainsArrayError(path);
        }
        else if (await containsGroup(store, path)) {
            throw new ContainsGroupError(path);
        }
        else {
            await initGroup(store, path, chunkStore);
        }
    }
    else {
        throw new ValueError(`Invalid mode argument: ${mode}`);
    }
    const readOnly = mode === "r";
    return Group.create(store, path, readOnly, chunkStore, cacheAttrs);
}

class ObjectStore {
    constructor() {
        this.object = {};
    }
    getItem(item) {
        if (!Object.prototype.hasOwnProperty.call(this.object, item)) {
            throw new KeyError(item);
        }
        return this.object[item];
    }
    setItem(item, value) {
        this.object[item] = value;
        return true;
    }
    deleteItem(item) {
        return delete this.object[item];
    }
    containsItem(item) {
        return Object.prototype.hasOwnProperty.call(this.object, item);
    }
    proxy() {
        return createProxy(this);
    }
    keys() {
        return Object.getOwnPropertyNames(this.object);
    }
}

export { ArrayNotFoundError, BoundsCheckError, ContainsArrayError, ContainsGroupError, Group, GroupNotFoundError, HTTPError, HTTPStore, InvalidSliceError, KeyError, MemoryStore, NegativeStepError, NestedArray, ObjectStore, PathNotFoundError, PermissionError, TooManyIndicesError, ValueError, ZarrArray, addCodec, array, create, createProxy, empty, full, getCodec, getTypedArrayCtr, getTypedArrayDtypeString, group, isKeyError, normalizeStoreArgument, ones, openArray, openGroup, rangeTypedArray, slice, sliceIndices, zeros, containsArray, containsGroup };
//# sourceMappingURL=core.mjs.map
