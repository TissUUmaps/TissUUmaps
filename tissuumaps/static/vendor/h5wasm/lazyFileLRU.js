// node_modules/typescript-lru-cache/src/LRUCacheNode.ts
var LRUCacheNode = class {
  constructor(key, value, options) {
    const {
      entryExpirationTimeInMS = null,
      next = null,
      prev = null,
      onEntryEvicted,
      onEntryMarkedAsMostRecentlyUsed,
      clone,
      cloneFn
    } = options ?? {};
    if (typeof entryExpirationTimeInMS === "number" && (entryExpirationTimeInMS <= 0 || Number.isNaN(entryExpirationTimeInMS))) {
      throw new Error("entryExpirationTimeInMS must either be null (no expiry) or greater than 0");
    }
    this.clone = clone ?? false;
    this.cloneFn = cloneFn ?? this.defaultClone;
    this.key = key;
    this.internalValue = this.clone ? this.cloneFn(value) : value;
    this.created = Date.now();
    this.entryExpirationTimeInMS = entryExpirationTimeInMS;
    this.next = next;
    this.prev = prev;
    this.onEntryEvicted = onEntryEvicted;
    this.onEntryMarkedAsMostRecentlyUsed = onEntryMarkedAsMostRecentlyUsed;
  }
  get value() {
    return this.clone ? this.cloneFn(this.internalValue) : this.internalValue;
  }
  get isExpired() {
    return typeof this.entryExpirationTimeInMS === "number" && Date.now() - this.created > this.entryExpirationTimeInMS;
  }
  invokeOnEvicted() {
    if (this.onEntryEvicted) {
      const { key, value, isExpired } = this;
      this.onEntryEvicted({ key, value, isExpired });
    }
  }
  invokeOnEntryMarkedAsMostRecentlyUsed() {
    if (this.onEntryMarkedAsMostRecentlyUsed) {
      const { key, value } = this;
      this.onEntryMarkedAsMostRecentlyUsed({ key, value });
    }
  }
  defaultClone(value) {
    if (typeof value === "boolean" || typeof value === "string" || typeof value === "number") {
      return value;
    }
    return JSON.parse(JSON.stringify(value));
  }
};

// node_modules/typescript-lru-cache/src/LRUCache.ts
var LRUCache = class {
  constructor(options) {
    this.lookupTable = /* @__PURE__ */ new Map();
    this.head = null;
    this.tail = null;
    const {
      maxSize = 25,
      entryExpirationTimeInMS = null,
      onEntryEvicted,
      onEntryMarkedAsMostRecentlyUsed,
      cloneFn,
      clone
    } = options ?? {};
    if (Number.isNaN(maxSize) || maxSize <= 0) {
      throw new Error("maxSize must be greater than 0.");
    }
    if (typeof entryExpirationTimeInMS === "number" && (entryExpirationTimeInMS <= 0 || Number.isNaN(entryExpirationTimeInMS))) {
      throw new Error("entryExpirationTimeInMS must either be null (no expiry) or greater than 0");
    }
    this.maxSizeInternal = maxSize;
    this.entryExpirationTimeInMS = entryExpirationTimeInMS;
    this.onEntryEvicted = onEntryEvicted;
    this.onEntryMarkedAsMostRecentlyUsed = onEntryMarkedAsMostRecentlyUsed;
    this.clone = clone;
    this.cloneFn = cloneFn;
  }
  get size() {
    return this.lookupTable.size;
  }
  get remainingSize() {
    return this.maxSizeInternal - this.size;
  }
  get newest() {
    if (!this.head) {
      return null;
    }
    return this.mapNodeToEntry(this.head);
  }
  get oldest() {
    if (!this.tail) {
      return null;
    }
    return this.mapNodeToEntry(this.tail);
  }
  get maxSize() {
    return this.maxSizeInternal;
  }
  set maxSize(value) {
    if (Number.isNaN(value) || value <= 0) {
      throw new Error("maxSize must be greater than 0.");
    }
    this.maxSizeInternal = value;
    this.enforceSizeLimit();
  }
  set(key, value, entryOptions) {
    const currentNodeForKey = this.lookupTable.get(key);
    if (currentNodeForKey) {
      this.removeNodeFromListAndLookupTable(currentNodeForKey);
    }
    const node = new LRUCacheNode(key, value, {
      entryExpirationTimeInMS: this.entryExpirationTimeInMS,
      onEntryEvicted: this.onEntryEvicted,
      onEntryMarkedAsMostRecentlyUsed: this.onEntryMarkedAsMostRecentlyUsed,
      clone: this.clone,
      cloneFn: this.cloneFn,
      ...entryOptions
    });
    this.setNodeAsHead(node);
    this.lookupTable.set(key, node);
    this.enforceSizeLimit();
    return this;
  }
  get(key) {
    const node = this.lookupTable.get(key);
    if (!node) {
      return null;
    }
    if (node.isExpired) {
      this.removeNodeFromListAndLookupTable(node);
      return null;
    }
    this.setNodeAsHead(node);
    return node.value;
  }
  peek(key) {
    const node = this.lookupTable.get(key);
    if (!node) {
      return null;
    }
    if (node.isExpired) {
      this.removeNodeFromListAndLookupTable(node);
      return null;
    }
    return node.value;
  }
  delete(key) {
    const node = this.lookupTable.get(key);
    if (!node) {
      return false;
    }
    return this.removeNodeFromListAndLookupTable(node);
  }
  has(key) {
    return this.lookupTable.has(key);
  }
  clear() {
    this.head = null;
    this.tail = null;
    this.lookupTable.clear();
  }
  find(condition) {
    let node = this.head;
    while (node) {
      if (node.isExpired) {
        const next = node.next;
        this.removeNodeFromListAndLookupTable(node);
        node = next;
        continue;
      }
      const entry = this.mapNodeToEntry(node);
      if (condition(entry)) {
        this.setNodeAsHead(node);
        return entry;
      }
      node = node.next;
    }
    return null;
  }
  forEach(callback) {
    let node = this.head;
    let index = 0;
    while (node) {
      if (node.isExpired) {
        const next = node.next;
        this.removeNodeFromListAndLookupTable(node);
        node = next;
        continue;
      }
      callback(node.value, node.key, index);
      node = node.next;
      index++;
    }
  }
  *values() {
    let node = this.head;
    while (node) {
      if (node.isExpired) {
        const next = node.next;
        this.removeNodeFromListAndLookupTable(node);
        node = next;
        continue;
      }
      yield node.value;
      node = node.next;
    }
  }
  *keys() {
    let node = this.head;
    while (node) {
      if (node.isExpired) {
        const next = node.next;
        this.removeNodeFromListAndLookupTable(node);
        node = next;
        continue;
      }
      yield node.key;
      node = node.next;
    }
  }
  *entries() {
    let node = this.head;
    while (node) {
      if (node.isExpired) {
        const next = node.next;
        this.removeNodeFromListAndLookupTable(node);
        node = next;
        continue;
      }
      yield this.mapNodeToEntry(node);
      node = node.next;
    }
  }
  *[Symbol.iterator]() {
    let node = this.head;
    while (node) {
      if (node.isExpired) {
        const next = node.next;
        this.removeNodeFromListAndLookupTable(node);
        node = next;
        continue;
      }
      yield this.mapNodeToEntry(node);
      node = node.next;
    }
  }
  enforceSizeLimit() {
    let node = this.tail;
    while (node !== null && this.size > this.maxSizeInternal) {
      const prev = node.prev;
      this.removeNodeFromListAndLookupTable(node);
      node = prev;
    }
  }
  mapNodeToEntry({ key, value }) {
    return {
      key,
      value
    };
  }
  setNodeAsHead(node) {
    this.removeNodeFromList(node);
    if (!this.head) {
      this.head = node;
      this.tail = node;
    } else {
      node.next = this.head;
      this.head.prev = node;
      this.head = node;
    }
    node.invokeOnEntryMarkedAsMostRecentlyUsed();
  }
  removeNodeFromList(node) {
    if (node.prev !== null) {
      node.prev.next = node.next;
    }
    if (node.next !== null) {
      node.next.prev = node.prev;
    }
    if (this.head === node) {
      this.head = node.next;
    }
    if (this.tail === node) {
      this.tail = node.prev;
    }
    node.next = null;
    node.prev = null;
  }
  removeNodeFromListAndLookupTable(node) {
    node.invokeOnEvicted();
    this.removeNodeFromList(node);
    return this.lookupTable.delete(node.key);
  }
};

// src/lazyFileLRU.ts
var defaultCache = class {
  constructor() {
    this.values = [];
  }
  get(key) {
    return this.values[key];
  }
  set(key, value) {
    this.values[key] = value;
  }
  has(key) {
    return typeof this.values[key] === "undefined";
  }
  get size() {
    return this.values.filter(function(value) {
      return value !== void 0;
    }).length;
  }
};
var LazyUint8Array = class {
  constructor(config) {
    this.serverChecked = false;
    this.totalFetchedBytes = 0;
    this.totalRequests = 0;
    this.readPages = [];
    this.readHeads = [];
    this.lastGet = -1;
    var _a, _b;
    this._chunkSize = config.requestChunkSize;
    this.maxSpeed = Math.round((config.maxReadSpeed || 5 * 1024 * 1024) / this._chunkSize);
    this.maxReadHeads = (_a = config.maxReadHeads) != null ? _a : 3;
    this.rangeMapper = config.rangeMapper;
    this.logPageReads = (_b = config.logPageReads) != null ? _b : false;
    if (config.fileLength) {
      this._length = config.fileLength;
    }
    this.requestLimiter = config.requestLimiter == null ? (ignored) => {
    } : config.requestLimiter;
    const LRUSize = config.LRUSize;
    if (LRUSize !== void 0) {
      this.cache = new LRUCache({ maxSize: LRUSize });
    } else {
      this.cache = new defaultCache();
    }
    //console.log(this.cache);
  }
  copyInto(buffer, outOffset, length, start) {
    if (start >= this.length)
      return 0;
    length = Math.min(this.length - start, length);
    const end = start + length;
    let i = 0;
    while (i < length) {
      const idx = start + i;
      const chunkOffset = idx % this.chunkSize;
      const chunkNum = idx / this.chunkSize | 0;
      const wantedSize = Math.min(this.chunkSize, end - idx);
      let inChunk = this.getChunk(chunkNum);
      if (chunkOffset !== 0 || wantedSize !== this.chunkSize) {
        inChunk = inChunk.subarray(chunkOffset, chunkOffset + wantedSize);
      }
      buffer.set(inChunk, outOffset + i);
      i += inChunk.length;
    }
    return length;
  }
  moveReadHead(wantedChunkNum) {
    for (const [i, head] of this.readHeads.entries()) {
      const fetchStartChunkNum = head.startChunk + head.speed;
      const newSpeed = Math.min(this.maxSpeed, head.speed * 2);
      const wantedIsInNextFetchOfHead = wantedChunkNum >= fetchStartChunkNum && wantedChunkNum < fetchStartChunkNum + newSpeed;
      if (wantedIsInNextFetchOfHead) {
        head.speed = newSpeed;
        head.startChunk = fetchStartChunkNum;
        if (i !== 0) {
          this.readHeads.splice(i, 1);
          this.readHeads.unshift(head);
        }
        return head;
      }
    }
    const newHead = {
      startChunk: wantedChunkNum,
      speed: 1
    };
    this.readHeads.unshift(newHead);
    while (this.readHeads.length > this.maxReadHeads)
      this.readHeads.pop();
    return newHead;
  }
  getChunk(wantedChunkNum) {
    let wasCached = true;
    //console.log(`cache size: ${this.cache.size}`);
    if (!this.cache.has(wantedChunkNum)) {
      wasCached = false;
      const head = this.moveReadHead(wantedChunkNum);
      const chunksToFetch = head.speed;
      //console.log(`fetching: ${chunksToFetch} chunks`);
      const startByte = head.startChunk * this.chunkSize;
      let endByte = (head.startChunk + chunksToFetch) * this.chunkSize - 1;
      endByte = Math.min(endByte, this.length - 1);
      const buf = this.doXHR(startByte, endByte);
      for (let i = 0; i < chunksToFetch; i++) {
        const curChunk = head.startChunk + i;
        if (i * this.chunkSize >= buf.byteLength)
          break;
        const curSize = (i + 1) * this.chunkSize > buf.byteLength ? buf.byteLength - i * this.chunkSize : this.chunkSize;
        this.cache.set(curChunk, new Uint8Array(buf, i * this.chunkSize, curSize));
      }
    }
    if (!this.cache.has(wantedChunkNum))
      throw new Error("doXHR failed (bug)!");
    const boring = !this.logPageReads || this.lastGet == wantedChunkNum;
    if (!boring) {
      this.lastGet = wantedChunkNum;
      this.readPages.push({
        pageno: wantedChunkNum,
        wasCached,
        prefetch: wasCached ? 0 : this.readHeads[0].speed - 1
      });
    }
    return this.cache.get(wantedChunkNum);
  }
  checkServer() {
    var xhr = new XMLHttpRequest();
    const url = this.rangeMapper(0, 0).url;
    xhr.open("HEAD", url, false);
    xhr.send(null);
    if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304))
      throw new Error("Couldn't load " + url + ". Status: " + xhr.status);
    var datalength = Number(xhr.getResponseHeader("Content-length"));
    var hasByteServing = xhr.getResponseHeader("Accept-Ranges") === "bytes";
    const encoding = xhr.getResponseHeader("Content-Encoding");
    var usesCompression = encoding && encoding !== "identity";
    if (!hasByteServing) {
      const msg = "Warning: The server did not respond with Accept-Ranges=bytes. It either does not support byte serving or does not advertise it (`Accept-Ranges: bytes` header missing), or your database is hosted on CORS and the server doesn't mark the accept-ranges header as exposed. This may lead to incorrect results.";
      console.warn(msg, "(seen response headers:", xhr.getAllResponseHeaders(), ")");
      throw new Error("The server did not respond with Accept-Ranges=bytes.");
    }
    if (usesCompression) {
      console.warn(`Warning: The server responded with ${encoding} encoding to a HEAD request. Ignoring since it may not do so for Range HTTP requests, but this will lead to incorrect results otherwise since the ranges will be based on the compressed data instead of the uncompressed data.`);
    }
    if (usesCompression) {
      datalength = null;
    }
    if (!this._length) {
      if (!datalength) {
        console.error("response headers", xhr.getAllResponseHeaders());
        throw Error("Length of the file not known. It must either be supplied in the config or given by the HTTP server.");
      }
      this._length = datalength;
    }
    this.serverChecked = true;
  }
  get length() {
    if (!this.serverChecked) {
      this.checkServer();
    }
    return this._length;
  }
  get chunkSize() {
    if (!this.serverChecked) {
      this.checkServer();
    }
    return this._chunkSize;
  }
  doXHR(absoluteFrom, absoluteTo) {
    //console.log(`[xhr of size ${(absoluteTo + 1 - absoluteFrom) / 1024} KiB @ ${absoluteFrom / 1024} KiB]`);
    this.requestLimiter(absoluteTo - absoluteFrom);
    this.totalFetchedBytes += absoluteTo - absoluteFrom;
    this.totalRequests++;
    if (absoluteFrom > absoluteTo)
      throw new Error("invalid range (" + absoluteFrom + ", " + absoluteTo + ") or no bytes requested!");
    if (absoluteTo > this.length - 1)
      throw new Error("only " + this.length + " bytes available! programmer error!");
    const {
      fromByte: from,
      toByte: to,
      url
    } = this.rangeMapper(absoluteFrom, absoluteTo);
    var xhr = new XMLHttpRequest();
    xhr.open("GET", url, false);
    if (this.length !== this.chunkSize)
      xhr.setRequestHeader("Range", "bytes=" + from + "-" + to);
    xhr.responseType = "arraybuffer";
    if (xhr.overrideMimeType) {
      xhr.overrideMimeType("text/plain; charset=x-user-defined");
    }
    xhr.send(null);
    if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304))
      throw new Error("Couldn't load " + url + ". Status: " + xhr.status);
    if (xhr.response !== void 0) {
      return xhr.response;
    } else {
      throw Error("xhr did not return uint8array");
    }
  }
};
function createLazyFile(FS, parent, name, canRead, canWrite, lazyFileConfig) {
  var lazyArray = new LazyUint8Array(lazyFileConfig);
  var properties = { isDevice: false, contents: lazyArray };
  var node = FS.createFile(parent, name, properties, canRead, canWrite);
  node.contents = lazyArray;
  Object.defineProperties(node, {
    usedBytes: {
      get: function() {
        return this.contents.length;
      }
    }
  });
  var stream_ops = {};
  var keys = Object.keys(node.stream_ops);
  keys.forEach(function(key) {
    var fn = node.stream_ops[key];
    stream_ops[key] = function forceLoadLazyFile() {
      FS.forceLoadFile(node);
      return fn.apply(null, arguments);
    };
  });
  stream_ops.read = function stream_ops_read(stream, buffer, offset, length, position) {
    FS.forceLoadFile(node);
    const contents = stream.node.contents;
    return contents.copyInto(buffer, offset, length, position);
  };
  node.stream_ops = stream_ops;
  return node;
}
/*export {
  LazyUint8Array,
  createLazyFile
};*/