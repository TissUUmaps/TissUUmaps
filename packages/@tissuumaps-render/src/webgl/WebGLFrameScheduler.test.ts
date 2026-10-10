import { afterEach, describe, expect, it, vi } from "vitest";

import type { Rect } from "@tissuumaps/core";

import type { WebGLContext } from "./WebGLContext";
import { WebGLFrameScheduler } from "./WebGLFrameScheduler";

/**
 * Creates a scheduler that draws one renderer on a stand-in WebGL context,
 * with the animation frames, the clock and the fence status set by the test
 */
function createTestScheduler() {
  let now = 0;
  let signaled = false;
  let frameCallbacks: FrameRequestCallback[] = [];
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) =>
    frameCallbacks.push(callback),
  );
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.stubGlobal("WebGL2RenderingContext", {
    SYNC_GPU_COMMANDS_COMPLETE: 0x9117,
    SYNC_STATUS: 0x9114,
    UNSIGNALED: 0x9118,
    SIGNALED: 0x9119,
  });
  vi.spyOn(performance, "now").mockImplementation(() => now);
  const gl = {
    fenceSync: vi.fn(() => ({})),
    getSyncParameter: vi.fn(() => (signaled ? 0x9119 : 0x9118)),
    deleteSync: vi.fn(),
    flush: vi.fn(),
  };
  const resizeCanvas = vi.fn();
  const context = {
    gl,
    clear: vi.fn(),
    resizeCanvas,
  } as unknown as WebGLContext;
  const canvas = { style: {} } as unknown as HTMLCanvasElement;
  const renderer = {
    viewport: { x: 0, y: 0, width: 1, height: 1 },
    draw: vi.fn(),
  };
  const scheduler = new WebGLFrameScheduler(context, canvas, [renderer]);
  const runFrame = (time: number, fenceSignaled: boolean) => {
    now = time;
    signaled = fenceSignaled;
    const callbacks = frameCallbacks;
    frameCallbacks = [];
    for (const callback of callbacks) {
      callback(time);
    }
  };
  return { scheduler, canvas, renderer, resizeCanvas, runFrame };
}

const viewport: Rect = { x: 0, y: 0, width: 100, height: 100 };
const zoomedViewport: Rect = { x: 10, y: 20, width: 50, height: 50 };

describe("WebGLFrameScheduler", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  describe("setViewport", () => {
    it("draws right away when the GPU is idle", () => {
      const { scheduler, canvas, renderer } = createTestScheduler();
      scheduler.setViewport(viewport);
      expect(renderer.draw).toHaveBeenCalledOnce();
      expect(renderer.viewport).toEqual(viewport);
      expect(canvas.style.transform).toBe("");
    });

    it("redraws a light overlay only every other frame, moving the drawn pixels in between", () => {
      const { scheduler, canvas, renderer, runFrame } = createTestScheduler();
      scheduler.setViewport(viewport);
      scheduler.setViewport(zoomedViewport);
      expect(canvas.style.transform).toBe("translate(-20%, -40%) scale(2, 2)");
      runFrame(8, true);
      expect(renderer.draw).toHaveBeenCalledOnce();
      runFrame(16, true);
      expect(renderer.draw).toHaveBeenCalledTimes(2);
      expect(renderer.viewport).toEqual(zoomedViewport);
      expect(canvas.style.transform).toBe("");
    });

    it("waits as long again after a slow redraw", () => {
      const { scheduler, renderer, runFrame } = createTestScheduler();
      scheduler.setViewport(viewport);
      scheduler.setViewport(zoomedViewport);
      runFrame(8, false);
      runFrame(150, true);
      runFrame(299, true);
      expect(renderer.draw).toHaveBeenCalledOnce();
      runFrame(300, true);
      expect(renderer.draw).toHaveBeenCalledTimes(2);
    });

    it("does not hold back a redraw after the page was hidden", () => {
      const { scheduler, renderer, runFrame } = createTestScheduler();
      scheduler.setViewport(viewport);
      scheduler.setViewport(zoomedViewport);
      runFrame(8, false);
      runFrame(60_000, true);
      expect(renderer.draw).toHaveBeenCalledTimes(2);
    });
  });

  describe("setContainerSize", () => {
    it("resizes the canvas right before the next redraw", () => {
      const { scheduler, canvas, renderer, resizeCanvas, runFrame } =
        createTestScheduler();
      scheduler.setViewport(viewport);
      scheduler.setContainerSize({ width: 800, height: 600 });
      runFrame(8, true);
      expect(resizeCanvas).not.toHaveBeenCalled();
      runFrame(16, true);
      expect(resizeCanvas).toHaveBeenCalledWith(canvas, {
        width: 800,
        height: 600,
      });
      expect(renderer.draw).toHaveBeenCalledTimes(2);
    });
  });
});
