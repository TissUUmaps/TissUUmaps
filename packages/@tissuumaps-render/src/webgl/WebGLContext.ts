import { AsyncUtils, type Dims, type TypedArray } from "@tissuumaps/core";

/**
 * A wrapper around a WebGL2RenderingContext that provides utility methods for
 * creating shaders, programs, buffers, and textures
 */
export class WebGLContext {
  // https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/canvas#maximum_canvas_size
  private static readonly _maxCanvasSize = 4096;

  readonly gl: WebGL2RenderingContext;
  /** The largest width and height of a texture that the GPU supports, in texels */
  readonly maxTextureSize: number;

  /**
   * Creates a new WebGLContext for the given canvas element
   *
   * @param canvas - The HTML canvas element to use for rendering
   * @throws Error if the browser does not support WebGL 2.0
   */
  constructor(canvas: HTMLCanvasElement) {
    // every draw clears and redraws the whole canvas (see WebGLFrameScheduler),
    // so the drawing buffer needs no preserving; nothing uses depth or stencil
    const gl = canvas.getContext("webgl2", {
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
    });
    if (gl === null) {
      throw new Error("WebGL 2.0 is not supported by the browser.");
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    this.gl = gl;
    this.maxTextureSize = gl.getParameter(
      WebGL2RenderingContext.MAX_TEXTURE_SIZE,
    ) as number;
  }

  /**
   * Resizes the canvas to match the given screen-space dimensions,
   * accounting for `devicePixelRatio` and clamping to {@link _maxCanvasSize}
   *
   * @param canvas - The HTML canvas element to resize
   * @param newCanvasSize - Desired canvas size in screen-space pixels
   * @returns `true` if the canvas size actually changed, `false` otherwise
   */
  resizeCanvas(canvas: HTMLCanvasElement, newCanvasSize: Dims): boolean {
    let width = Math.floor(newCanvasSize.width * window.devicePixelRatio);
    let height = Math.floor(newCanvasSize.height * window.devicePixelRatio);
    if (width <= 0 || height <= 0) {
      width = 1;
      height = 1;
    } else if (
      width > WebGLContext._maxCanvasSize ||
      height > WebGLContext._maxCanvasSize
    ) {
      const scale = WebGLContext._maxCanvasSize / Math.max(width, height);
      width = Math.floor(width * scale);
      height = Math.floor(height * scale);
    }
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      this.gl.viewport(0, 0, width, height);
      return true;
    }
    return false;
  }

  /**
   * Clears the color buffer with transparent black
   */
  clear(): void {
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
  }

  /**
   * Compiles vertex and fragment shaders, links them into a program, and
   * returns the linked program
   *
   * The individual shaders are flagged for deletion after linking.
   *
   * @param vertexShaderSource - GLSL source for the vertex shader
   * @param fragmentShaderSource - GLSL source for the fragment shader
   * @throws Error if shader compilation or program linking fails
   */
  createProgram(
    vertexShaderSource: string,
    fragmentShaderSource: string,
  ): WebGLProgram {
    const vertexShader = this.gl.createShader(
      WebGL2RenderingContext.VERTEX_SHADER,
    );
    if (vertexShader === null) {
      throw new Error("Failed to create vertex shader.");
    }
    const fragmentShader = this.gl.createShader(
      WebGL2RenderingContext.FRAGMENT_SHADER,
    );
    if (fragmentShader === null) {
      throw new Error("Failed to create fragment shader.");
    }
    try {
      const program = this.gl.createProgram();
      if (program === null) {
        throw new Error("Failed to create shader program.");
      }
      try {
        for (const [shader, shaderSource] of [
          [vertexShader, vertexShaderSource],
          [fragmentShader, fragmentShaderSource],
        ] as const) {
          this.gl.shaderSource(shader, shaderSource);
          this.gl.compileShader(shader);
          this.gl.attachShader(program, shader);
        }
        this.gl.linkProgram(program);
        if (
          !this.gl.getProgramParameter(
            program,
            WebGL2RenderingContext.LINK_STATUS,
          )
        ) {
          const programInfoLog = this.gl.getProgramInfoLog(program);
          const vertexShaderInfoLog = this.gl.getShaderInfoLog(vertexShader);
          const fragmentShaderInfoLog =
            this.gl.getShaderInfoLog(fragmentShader);
          throw new Error(
            `Shader program linking failed: ${programInfoLog}\n` +
              `Vertex shader log: ${vertexShaderInfoLog}\n` +
              `Fragment shader log: ${fragmentShaderInfoLog}`,
          );
        }
        return program;
      } catch (error) {
        this.gl.deleteProgram(program);
        throw error;
      }
    } finally {
      // flag shader for deletion (i.e., delete them when no longer in use)
      // https://registry.khronos.org/OpenGL-Refpages/gl4/html/glDeleteShader.xhtml
      this.gl.deleteShader(vertexShader);
      this.gl.deleteShader(fragmentShader);
    }
  }

  /**
   * Returns the location of a uniform variable in a shader program
   *
   * @param program - The shader program
   * @param name - The uniform name as declared in the shader
   * @throws Error if the uniform is not found
   */
  getUniformLocation(
    program: WebGLProgram,
    name: string,
  ): WebGLUniformLocation {
    const uniformLocation = this.gl.getUniformLocation(program, name);
    if (uniformLocation === null) {
      throw new Error(`Failed to get uniform location for ${name}`);
    }
    return uniformLocation;
  }

  /**
   * Creates a new vertex array object (VAO)
   *
   * @throws Error if VAO creation fails
   */
  createVertexArray(): WebGLVertexArrayObject {
    const vao = this.gl.createVertexArray();
    if (vao === null) {
      throw new Error("Failed to create vertex array object.");
    }
    return vao;
  }

  /**
   * Binds a buffer to a vertex attribute as a floating-point attribute
   *
   * @param target - Buffer binding target
   * @param buffer - The buffer containing attribute data
   * @param index - Attribute location index
   * @param size - Number of components per vertex attribute
   * @param type - Data type (e.g. `gl.FLOAT`)
   * @param options - Optional normalized, stride, offset, and divisor settings
   */
  configureVertexFloatAttribute(
    target: GLenum,
    buffer: WebGLBuffer,
    index: number,
    size: number,
    type: GLenum,
    options?: {
      normalized?: boolean;
      stride?: number;
      offset?: number;
      divisor?: number;
    },
  ): void {
    const {
      normalized = false,
      stride = 0,
      offset = 0,
      divisor = 0,
    } = options ?? {};
    this.gl.bindBuffer(target, buffer);
    this.gl.enableVertexAttribArray(index);
    this.gl.vertexAttribPointer(index, size, type, normalized, stride, offset);
    this.gl.vertexAttribDivisor(index, divisor);
    this.gl.bindBuffer(target, null);
  }

  /**
   * Binds a buffer to a vertex attribute as an integer attribute
   *
   * @param target - Buffer binding target
   * @param buffer - The buffer containing attribute data
   * @param index - Attribute location index
   * @param size - Number of components per vertex attribute
   * @param type - Data type (e.g. `gl.UNSIGNED_INT`)
   * @param options - Optional stride, offset, and divisor settings
   */
  configureVertexIntAttribute(
    target: GLenum,
    buffer: WebGLBuffer,
    index: number,
    size: number,
    type: GLenum,
    options?: { stride?: number; offset?: number; divisor?: number },
  ): void {
    const { stride = 0, offset = 0, divisor = 0 } = options ?? {};
    this.gl.bindBuffer(target, buffer);
    this.gl.enableVertexAttribArray(index);
    this.gl.vertexAttribIPointer(index, size, type, stride, offset);
    this.gl.vertexAttribDivisor(index, divisor);
    this.gl.bindBuffer(target, null);
  }

  /**
   * Creates an immutable 2D data texture with nearest-neighbor filtering
   * and clamp-to-edge wrapping
   *
   * @param internalformat - Internal texture format (e.g. `gl.RGBA32F`)
   * @param width - Texture width in texels
   * @param height - Texture height in texels
   * @param format - Pixel data format (e.g. `gl.RGBA`)
   * @param type - Pixel data type (e.g. `gl.FLOAT`)
   * @param data - Optional initial pixel data
   * @throws Error if texture creation fails
   */
  createDataTexture(
    internalformat: GLenum,
    width: number,
    height: number,
    format: GLenum,
    type: GLenum,
    data?: TypedArray,
  ): WebGLTexture {
    const texture = this.gl.createTexture();
    if (texture === null) {
      throw new Error("Failed to create texture.");
    }
    this.gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
    this.gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_MAG_FILTER,
      WebGL2RenderingContext.NEAREST,
    );
    this.gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_MIN_FILTER,
      WebGL2RenderingContext.NEAREST,
    );
    this.gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_WRAP_S,
      WebGL2RenderingContext.CLAMP_TO_EDGE,
    );
    this.gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_WRAP_T,
      WebGL2RenderingContext.CLAMP_TO_EDGE,
    );
    this.gl.texStorage2D(
      WebGL2RenderingContext.TEXTURE_2D,
      1,
      internalformat,
      width,
      height,
    );
    if (data !== undefined) {
      // texStorage2D creates immutable storage, so we need to use texSubImage2D instead of texImage2D
      this.gl.texSubImage2D(
        WebGL2RenderingContext.TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        format,
        type,
        data,
      );
    }
    this.gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
    return texture;
  }

  /**
   * Uploads new pixel data into an existing immutable 2D data texture
   *
   * @param texture - The target texture
   * @param width - Texture width in texels
   * @param height - Texture height in texels
   * @param format - Pixel data format (e.g. `gl.RGBA`)
   * @param type - Pixel data type (e.g. `gl.FLOAT`)
   * @param data - The pixel data to upload
   */
  loadDataTexture(
    texture: WebGLTexture,
    width: number,
    height: number,
    format: GLenum,
    type: GLenum,
    data: TypedArray,
  ): void {
    this.gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
    this.gl.texSubImage2D(
      WebGL2RenderingContext.TEXTURE_2D,
      0,
      0,
      0,
      width,
      height,
      format,
      type,
      data,
    );
    this.gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
  }

  /**
   * Loads an image from a URL and creates a WebGL texture from it
   *
   * If the image fails to load, or the operation is aborted, the pending image load is
   * cancelled and the texture is deleted again before rejecting.
   *
   * @param url - The image URL
   * @param options - Optional mipmap generation flag and abort signal
   * @throws Error if texture creation fails, the image fails to load, or the
   * operation was aborted
   */
  async loadImageTextureFromUrl(
    url: string,
    options?: { signal?: AbortSignal; mipmap?: boolean },
  ): Promise<WebGLTexture> {
    const { signal, mipmap = false } = options ?? {};
    signal?.throwIfAborted();
    const texture = this.gl.createTexture();
    if (texture === null) {
      throw new Error("Failed to create texture.");
    }
    this.gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
    this.gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_MAG_FILTER,
      WebGL2RenderingContext.LINEAR,
    );
    this.gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_MIN_FILTER,
      mipmap
        ? WebGL2RenderingContext.LINEAR_MIPMAP_LINEAR
        : WebGL2RenderingContext.LINEAR,
    );
    this.gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_WRAP_S,
      WebGL2RenderingContext.CLAMP_TO_EDGE,
    );
    this.gl.texParameteri(
      WebGL2RenderingContext.TEXTURE_2D,
      WebGL2RenderingContext.TEXTURE_WRAP_T,
      WebGL2RenderingContext.CLAMP_TO_EDGE,
    );
    this.gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
    const img = new Image();
    const promise = new Promise<void>((resolve, reject) => {
      img.onload = () => {
        if (signal?.aborted) {
          reject(new Error("Image load aborted."));
          return;
        }
        this.gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
        this.gl.texImage2D(
          WebGL2RenderingContext.TEXTURE_2D,
          0,
          WebGL2RenderingContext.RGBA,
          WebGL2RenderingContext.RGBA,
          WebGL2RenderingContext.UNSIGNED_BYTE,
          img,
        );
        if (mipmap) {
          this.gl.generateMipmap(WebGL2RenderingContext.TEXTURE_2D);
        }
        this.gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
        resolve();
      };
      img.onerror = (event) => {
        reject(new Error(`Failed to load image: ${url}`, { cause: event }));
      };
      img.src = url;
    });
    try {
      await AsyncUtils.raceSignal(promise, { signal });
    } catch (error) {
      img.removeAttribute("src"); // cancel a still pending image load
      this.gl.deleteTexture(texture);
      throw error;
    }
    return texture;
  }

  /**
   * Creates a new WebGL buffer
   *
   * @throws Error if buffer creation fails
   */
  createBuffer(): WebGLBuffer {
    const buffer = this.gl.createBuffer();
    if (buffer === null) {
      throw new Error("Failed to create buffer.");
    }
    return buffer;
  }

  /**
   * Allocates (or re-allocates) storage for a buffer and fills it with the
   * given data, discarding previous contents
   *
   * @param target - Buffer binding target (e.g. `gl.ARRAY_BUFFER`)
   * @param buffer - The buffer to allocate
   * @param data - The data to fill the buffer with; sets its size
   * @param usage - Usage hint (e.g. `gl.STATIC_DRAW`)
   */
  allocateBuffer(
    target: GLenum,
    buffer: WebGLBuffer,
    data: Exclude<TypedArray, Float64Array>,
    usage: GLenum,
  ): void {
    this.gl.bindBuffer(target, buffer);
    this.gl.bufferData(target, data, usage);
    this.gl.bindBuffer(target, null);
  }

  /**
   * Uploads typed array data into a region of an existing buffer
   *
   * @param target - Buffer binding target
   * @param buffer - The target buffer
   * @param data - The data to upload
   * @param options - Optional element offset within the buffer
   */
  loadBuffer(
    target: GLenum,
    buffer: WebGLBuffer,
    data: Exclude<TypedArray, Float64Array>,
    options?: { offset?: number },
  ): void {
    const { offset = 0 } = options ?? {};
    this.gl.bindBuffer(target, buffer);
    this.gl.bufferSubData(target, offset * data.BYTES_PER_ELEMENT, data);
    this.gl.bindBuffer(target, null);
  }

  /**
   * Enables premultiplied-alpha blending (Porter-Duff "over" operator)
   */
  enableAlphaBlending(): void {
    // https://en.wikipedia.org/wiki/Alpha_compositing
    // https://learnopengl.com/Advanced-OpenGL/Blending
    // https://www.khronos.org/opengl/wiki/Blending
    // https://www.realtimerendering.com/blog/gpus-prefer-premultiplication/
    this.gl.enable(WebGL2RenderingContext.BLEND);
    this.gl.blendEquation(WebGL2RenderingContext.FUNC_ADD);
    this.gl.blendFuncSeparate(
      WebGL2RenderingContext.ONE, // alpha is premultiplied in fragment shader
      WebGL2RenderingContext.ONE_MINUS_SRC_ALPHA,
      WebGL2RenderingContext.ONE,
      WebGL2RenderingContext.ONE_MINUS_SRC_ALPHA,
    );
  }

  /**
   * Disables blending and restores the default blend state
   */
  disableAlphaBlending(): void {
    this.gl.disable(WebGL2RenderingContext.BLEND);
    this.gl.blendEquation(WebGL2RenderingContext.FUNC_ADD);
    this.gl.blendFuncSeparate(
      WebGL2RenderingContext.ONE,
      WebGL2RenderingContext.ZERO,
      WebGL2RenderingContext.ONE,
      WebGL2RenderingContext.ZERO,
    );
  }

  /**
   * Destroys the WebGLContext by forcing the underlying WebGL context to be
   * lost, releasing its GPU resources immediately.
   *
   * Browsers cap the number of concurrently live WebGL contexts, so a context
   * that is never released (e.g. across React StrictMode / HMR remounts)
   * eventually prevents new contexts from being created. Callers are expected
   * to have already destroyed any resources (programs, buffers, textures, VAOs)
   * they created through this context.
   */
  destroy(): void {
    const glLoseContextExtension = this.gl.getExtension("WEBGL_lose_context");
    if (glLoseContextExtension !== null) {
      glLoseContextExtension.loseContext();
    }
  }
}
