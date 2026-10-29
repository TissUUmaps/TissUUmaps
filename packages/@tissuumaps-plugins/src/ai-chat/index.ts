// the store types of @tissuumaps/core use Zustand's Immer middleware
import type {} from "zustand/middleware/immer";

import type { Plugin } from "@tissuumaps/core";

import { mountAiChat } from "./panel";

/**
 * A chat panel that lets the user ask a model about the open project and
 * change how its objects are shown: show and hide objects, change their
 * opacity and point size, and color and filter labels and points by the
 * columns of their tables
 *
 * The model runs at OpenAI or Anthropic, with the user's own API key, or
 * through Ollama, on the user's computer or online. Each request sends the
 * model a summary of the project. The changes go through the project store,
 * and "Undo" takes them back.
 */
export const aiChatPlugin: Plugin = {
  id: "ai-chat",
  name: "AI chat",
  mount: mountAiChat,
};
