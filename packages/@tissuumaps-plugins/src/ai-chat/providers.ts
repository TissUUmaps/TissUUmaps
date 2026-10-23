// The services that can answer in the chat, and how the panel talks to them.
//
// The conversation is kept in one format of the panel's own (ChatMessage).
// Each provider turns it into the format of its API, sends it with the tools,
// and turns the answer back, so that the tools work the same with all of
// them. What only one API understands, such as the model's thinking, which
// has to be sent back as it came, is kept with the answer and sent back only
// to that kind of API.
//
// OpenAI and Anthropic only accept the model's thinking back while everything
// sent before it is unchanged, and they bill less for a beginning they have
// seen before. So for them the beginning of each request stays the same: the
// fixed instructions, the tools, then the conversation, to which only new
// messages are added. What changes (the project summary and the personality)
// goes with the user's message that started the current turn, and the
// thinking goes back only within that turn, while the model calls tools.
// If a service still refuses the thinking, the request is sent again once
// without it.
//
// An API key is sent only to the API of its own provider, and never shown:
// error messages that could contain it go through hideKey.

/** The services the chat can use */
export type ProviderId = "openai" | "anthropic" | "ollama" | "ollama-cloud";

/** The kinds of chat API, each with its own format */
type ApiFormat = "openai" | "anthropic" | "ollama";

/** A tool the model can call, as described to it */
export type ToolSpec = {
  name: string;
  description: string;
  /** The arguments, as a JSON schema */
  parameters: object;
};

/** A tool call requested by the model */
export type ToolCall = {
  /** Links the call to its result; made up when the API gives none */
  id: string;
  name: string;
  arguments: Record<string, unknown>;
};

/** An answer of the model */
export type AssistantMessage = {
  role: "assistant";
  text: string;
  toolCalls: ToolCall[];
  /**
   * The answer as the API sent it, with what only that API understands (e.g.
   * the model's thinking); sent back as it is to the same kind of API
   */
  original?: { api: ApiFormat; items: unknown[] };
  /** The tokens of the request and of the answer, if the API said */
  usage?: Usage;
};

/**
 * The tokens of one or more requests, as the services bill them
 *
 * `input` counts only the tokens billed at the full price: those read from or
 * written to the prompt cache are counted apart.
 */
export type Usage = {
  input: number;
  cacheRead: number;
  cacheWrite: number;
  output: number;
};

/** The price of a model, in dollars per million tokens of each kind */
export type Prices = Usage;

/** A message of the conversation, in the panel's own format */
export type ChatMessage =
  | { role: "user"; text: string }
  | AssistantMessage
  | { role: "tool"; callId: string; name: string; result: string };

/** What the panel asks a provider */
export type ChatRequest = {
  model: string;
  /** The API key, or "" for the providers that need none */
  key: string;
  /** The instructions, which are the same for every request */
  instructions: string;
  /**
   * What changes between requests: the personality and the project summary,
   * the same for all the requests of one turn
   */
  context: string;
  messages: ChatMessage[];
  tools: ToolSpec[];
};

/** The answer of the model, or an error message the user can act on */
export type ChatAnswer = { message: AssistantMessage } | { error: string };

/** A service that can answer in the chat */
export type Provider = {
  id: ProviderId;
  /** The name shown to the user, e.g. "OpenAI" */
  name: string;
  /**
   * The models offered in the menu, the default first, with their prices for
   * the services that bill them
   */
  models: { name: string; note: string; prices?: Prices }[];
  /** Whether the user has to give an API key */
  needsKey: boolean;
  /**
   * Whether the models are strong enough, with a long enough context, to be
   * told the types and the state of the whole app, besides the summary
   */
  wholeApp: boolean;
  /**
   * Sends the conversation to the model
   *
   * @param request - The model, key, instructions, conversation and tools
   * @param onText - Called with the text so far, each time a piece arrives
   * @param options - Optional abort signal
   * @returns The answer, or an error message
   * @throws TypeError if the service cannot be reached
   */
  chat: (
    request: ChatRequest,
    onText: (text: string) => void,
    options?: { signal?: AbortSignal },
  ) => Promise<ChatAnswer>;
  /**
   * Checks an API key with a request that costs nothing
   *
   * @param key - The API key
   * @param options - Optional abort signal
   * @returns An error message, or `undefined` if the key works
   * @throws TypeError if the service cannot be reached
   */
  checkKey?: (
    key: string,
    options?: { signal?: AbortSignal },
  ) => Promise<string | undefined>;
};

/** What replaces an API key in text shown to the user */
const hiddenKey = "[key hidden]";

/**
 * Hides an API key in a text, e.g. in an error message of a provider, which
 * may repeat the key or a part of it
 *
 * @param text - The text
 * @param key - The API key, or ""
 * @returns The text, with the key and anything that looks like a key hidden
 */
export function hideKey(text: string, key: string): string {
  const withoutKey = key.length >= 8 ? text.split(key).join(hiddenKey) : text;
  // keys of OpenAI and Anthropic start with "sk-", e.g. "sk-proj-..." or "sk-ant-..."
  return withoutKey.replace(/\bsk-[\w*.-]{4,}/g, hiddenKey);
}

/**
 * Where the current turn starts: the last message of the user, after which
 * there are only the model's answers and tool results of this turn
 *
 * @param messages - The conversation
 * @returns The index of that message, or -1 if there is none
 */
function turnStart(messages: ChatMessage[]): number {
  return messages.findLastIndex((message) => message.role === "user");
}

/**
 * The user's message that starts the turn, with the context of the turn
 *
 * @param text - The user's message
 * @param context - The personality and the project summary
 * @returns The text sent to the model
 */
function withContext(text: string, context: string): string {
  return `[Context from the TissUUmaps chat panel, not typed by the user]\n${context}\n[End of the context]\n\n${text}`;
}

/**
 * Whether an answer of the model, at a place in the conversation, can be sent
 * back as the API sent it, thinking included: only within the current turn,
 * since the thinking of earlier turns belongs to their own context
 *
 * @param message - The answer
 * @param index - Its place in the conversation
 * @param start - Where the current turn starts, see {@link turnStart}
 * @param api - The kind of API the request goes to
 * @param keepThinking - Whether thinking can be sent at all
 * @returns The answer as the API sent it, or `undefined` to send it as text
 * and tool calls only
 */
function originalItems(
  message: AssistantMessage,
  index: number,
  start: number,
  api: ApiFormat,
  keepThinking: boolean,
): unknown[] | undefined {
  return keepThinking && index > start && message.original?.api === api
    ? message.original.items
    : undefined;
}

/**
 * Makes up an ID for a tool call, for the APIs that give none (Ollama), since
 * the other APIs need one to link the call to its result
 *
 * @returns The ID
 */
function newCallId(): string {
  return `call_${crypto.randomUUID()}`;
}

/**
 * Reads the arguments of a tool call that arrive as JSON text
 *
 * @param text - The JSON text
 * @returns The arguments, or no arguments if the text is not a JSON object
 */
function parseArguments(text: string | undefined): Record<string, unknown> {
  try {
    const value: unknown = JSON.parse(
      text === undefined || text === "" ? "{}" : text,
    );
    return typeof value === "object" && value !== null && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

/**
 * Reads an answer that arrives piece by piece, one line at a time
 *
 * @param response - The response to read
 * @param onLine - Called with each line while the answer arrives
 * @param options - Optional abort signal, which is also the signal of the
 * request: aborting it ends the stream
 */
async function readLines(
  response: Response,
  onLine: (line: string) => void,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  if (response.body === null) {
    throw new Error("the service sent an empty answer");
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let text = "";
  while (true) {
    const { done, value } = await reader.read();
    text += done ? "\n" : value;
    const lines = text.split("\n");
    // the last line may not be complete yet
    text = lines.pop() ?? "";
    for (const line of lines) {
      onLine(line);
    }
    if (done) {
      return;
    }
  }
}

/**
 * Reads an answer of Ollama that arrives as one line of JSON per update (the
 * progress of a download, or a chat answer piece by piece)
 *
 * @typeParam TLine - The type of each line
 * @param response - The response to read
 * @param onLine - Called with each line while the answer arrives
 * @param options - Optional abort signal, which is also the signal of the
 * request: aborting it ends the stream
 */
export async function readJsonLines<TLine>(
  response: Response,
  onLine: (line: TLine) => void,
  options?: { signal?: AbortSignal },
): Promise<void> {
  await readLines(
    response,
    (line) => {
      if (line.trim() !== "") {
        onLine(JSON.parse(line) as TLine);
      }
    },
    options,
  );
}

/**
 * Reads an answer that arrives as server-sent events, as the answers of
 * OpenAI and Anthropic do
 *
 * In these APIs, the data of each event is one line of JSON, which also says
 * the type of the event; the other lines only name the event or are empty.
 *
 * @typeParam TEvent - The type of each event
 * @param response - The response to read
 * @param onEvent - Called with each event while the answer arrives
 * @param options - Optional abort signal
 */
async function readServerEvents<TEvent>(
  response: Response,
  onEvent: (event: TEvent) => void,
  options?: { signal?: AbortSignal },
): Promise<void> {
  await readLines(
    response,
    (line) => {
      if (!line.startsWith("data:")) {
        return;
      }
      const data = line.slice("data:".length).trim();
      if (data !== "" && data !== "[DONE]") {
        onEvent(JSON.parse(data) as TEvent);
      }
    },
    options,
  );
}

// --- OpenAI and Anthropic ---

/** An error answer of OpenAI or Anthropic */
type ApiError = { type: string; message: string };

/**
 * Reads the error in an answer of OpenAI or Anthropic that did not succeed
 *
 * @param response - The response
 * @returns The type and message of the error
 */
async function readError(response: Response): Promise<ApiError> {
  const text = await response.text();
  try {
    const { error } = JSON.parse(text) as {
      error?: { type?: string; code?: string | null; message?: string };
    };
    return {
      type: error?.code ?? error?.type ?? "",
      message: error?.message ?? text,
    };
  } catch {
    return { type: "", message: text };
  }
}

/**
 * Turns an error answer of OpenAI or Anthropic into a message the user can
 * act on, which never contains the key
 *
 * @param name - The name of the provider
 * @param status - The HTTP status of the answer
 * @param error - The error in the answer
 * @param key - The API key, to hide it
 * @param model - The model, if the request was for one
 * @returns The message
 */
function explainApiError(
  name: string,
  status: number,
  error: ApiError,
  key: string,
  model?: string,
): string {
  // the message of a 401 can repeat a part of the key: it is not shown
  if (status === 401) {
    return `${name} did not accept the API key. Check that you copied all of it and that it has not expired or been deleted, or make a new key (see "Help").`;
  }
  const message = hideKey(error.message, key).slice(0, 300);
  if (/workspace/i.test(message)) {
    return `This ${name} key is not tied to one workspace, so ${name} does not know which one to use. Make a new key and choose one workspace for it (see "Help"). (${message})`;
  }
  if (status === 429 || error.type === "insufficient_quota") {
    return `${name}: too many requests, or no credit left. Wait a moment and try again, or check the billing of your ${name} account. (${message})`;
  }
  if (status === 404 && model !== undefined) {
    return `${name} does not offer the model "${model}" to your account. Choose another model. (${message})`;
  }
  if (status >= 500) {
    return `${name} has a problem right now (error ${status}). Try again in a moment.`;
  }
  return `${name} error ${status}: ${message}`;
}

/** The address of OpenAI's API */
const openAiUrl = "https://api.openai.com/v1";

/** An item of the output of OpenAI's Responses API */
type OpenAiItem = {
  type: string;
  call_id?: string;
  name?: string;
  arguments?: string;
  content?: { type: string; text?: string }[];
};

/** An event of a streamed answer of OpenAI's Responses API */
type OpenAiEvent = {
  type: string;
  delta?: string;
  message?: string;
  response?: {
    output?: OpenAiItem[];
    error?: { message?: string } | null;
    usage?: {
      input_tokens?: number;
      input_tokens_details?: {
        cached_tokens?: number;
        cache_write_tokens?: number;
      };
      output_tokens?: number;
    };
  };
};

/**
 * Turns the conversation into the input of OpenAI's Responses API
 *
 * The answers of earlier turns go without their reasoning and without the IDs
 * that would ask for it.
 *
 * @param messages - The conversation
 * @param context - The context of the current turn
 * @param keepThinking - Whether the reasoning of the current turn is sent back
 * @returns The input items
 */
function toOpenAiInput(
  messages: ChatMessage[],
  context: string,
  keepThinking: boolean,
): unknown[] {
  const start = turnStart(messages);
  return messages.flatMap((message, index): unknown[] => {
    switch (message.role) {
      case "user":
        return [
          {
            role: "user",
            content:
              index === start
                ? withContext(message.text, context)
                : message.text,
          },
        ];
      case "tool":
        return [
          {
            type: "function_call_output",
            call_id: message.callId,
            output: message.result,
          },
        ];
      case "assistant":
        return (
          originalItems(message, index, start, "openai", keepThinking) ?? [
            ...(message.text === ""
              ? []
              : [{ role: "assistant", content: message.text }]),
            ...message.toolCalls.map((call) => ({
              type: "function_call",
              call_id: call.id,
              name: call.name,
              arguments: JSON.stringify(call.arguments),
            })),
          ]
        );
    }
  });
}

/** What a sending function returns when the service refused the thinking */
const thinkingRefused = "thinking refused";

/**
 * Asks a model of OpenAI, with the Responses API, which OpenAI recommends for
 * tools with its current models; sends the request again once without the
 * reasoning if OpenAI refuses it
 */
async function chatWithOpenAi(
  request: ChatRequest,
  onText: (text: string) => void,
  options?: { signal?: AbortSignal },
): Promise<ChatAnswer> {
  const answer = await sendToOpenAi(request, true, onText, options);
  return answer === thinkingRefused
    ? ((await sendToOpenAi(request, false, onText, options)) as ChatAnswer)
    : answer;
}

/**
 * Sends one request to OpenAI
 *
 * Nothing is stored by OpenAI between requests (`store: false`): the whole
 * conversation is sent each time, with the model's reasoning of the current
 * turn in the encrypted form that OpenAI gives back for that.
 *
 * @param request - The request
 * @param keepThinking - Whether to send the reasoning of the current turn
 * @param onText - Called with the text so far
 * @param options - Optional abort signal
 * @returns The answer, or {@link thinkingRefused} if OpenAI refused the
 * reasoning that was sent
 */
async function sendToOpenAi(
  { model, key, instructions, context, messages, tools }: ChatRequest,
  keepThinking: boolean,
  onText: (text: string) => void,
  options?: { signal?: AbortSignal },
): Promise<ChatAnswer | typeof thinkingRefused> {
  const { signal } = options ?? {};
  const response = await fetch(`${openAiUrl}/responses`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      instructions,
      input: toOpenAiInput(messages, context, keepThinking),
      tools: tools.map((tool) => ({
        type: "function",
        ...tool,
        strict: false,
      })),
      stream: true,
      store: false,
      include: ["reasoning.encrypted_content"],
    }),
    signal,
  });
  if (!response.ok) {
    const error = await readError(response);
    if (
      keepThinking &&
      response.status === 400 &&
      /reasoning|encrypted/i.test(error.message)
    ) {
      return thinkingRefused;
    }
    return {
      error: explainApiError("OpenAI", response.status, error, key, model),
    };
  }
  let text = "";
  let output: OpenAiItem[] = [];
  let usage: Usage | undefined;
  let error: string | undefined;
  await readServerEvents<OpenAiEvent>(
    response,
    (event) => {
      switch (event.type) {
        case "response.output_text.delta":
          text += event.delta ?? "";
          onText(text);
          break;
        case "response.completed":
        case "response.incomplete": {
          output = event.response?.output ?? [];
          const tokens = event.response?.usage;
          if (tokens !== undefined) {
            // OpenAI counts the cached tokens also in input_tokens
            const cacheRead = tokens.input_tokens_details?.cached_tokens ?? 0;
            const cacheWrite =
              tokens.input_tokens_details?.cache_write_tokens ?? 0;
            usage = {
              input: (tokens.input_tokens ?? 0) - cacheRead - cacheWrite,
              cacheRead,
              cacheWrite,
              output: tokens.output_tokens ?? 0,
            };
          }
          break;
        }
        case "response.failed":
          error ??= event.response?.error?.message ?? "the answer failed";
          break;
        case "error":
          error ??= event.message ?? "unknown error";
          break;
      }
    },
    { signal },
  );
  if (error !== undefined) {
    return { error: `OpenAI: ${hideKey(error, key)}` };
  }
  const finalText = output
    .filter((item) => item.type === "message")
    .flatMap((item) => item.content ?? [])
    .map((part) => part.text ?? "")
    .join("");
  return {
    message: {
      role: "assistant",
      text: finalText === "" ? text : finalText,
      toolCalls: output
        .filter((item) => item.type === "function_call")
        .map((item) => ({
          id: item.call_id ?? newCallId(),
          name: item.name ?? "",
          arguments: parseArguments(item.arguments),
        })),
      original: { api: "openai", items: output },
      usage,
    },
  };
}

/**
 * Checks an OpenAI key by listing the models, which costs nothing
 */
async function checkOpenAiKey(
  key: string,
  options?: { signal?: AbortSignal },
): Promise<string | undefined> {
  const { signal } = options ?? {};
  const response = await fetch(`${openAiUrl}/models`, {
    headers: { Authorization: `Bearer ${key}` },
    signal,
  });
  return response.ok
    ? undefined
    : explainApiError(
        "OpenAI",
        response.status,
        await readError(response),
        key,
      );
}

/** The address of Anthropic's API */
const anthropicUrl = "https://api.anthropic.com/v1";

/**
 * The longest answer asked from Anthropic's models, in tokens, which Anthropic
 * requires; it includes the model's thinking
 */
const anthropicMaxTokens = 8192;

/** A content block of a message of Anthropic's Messages API */
type AnthropicBlock = {
  type: string;
  text?: string;
  thinking?: string;
  signature?: string;
  id?: string;
  name?: string;
  input?: unknown;
  tool_use_id?: string;
  content?: string;
};

/** An event of a streamed answer of Anthropic's Messages API */
type AnthropicEvent = {
  type: string;
  index?: number;
  content_block?: AnthropicBlock;
  delta?: {
    type?: string;
    text?: string;
    thinking?: string;
    signature?: string;
    partial_json?: string;
  };
  error?: { message?: string };
  /** In "message_start": the usage so far */
  message?: { usage?: AnthropicUsage };
  /** In "message_delta": the usage so far, which replaces the earlier one */
  usage?: AnthropicUsage;
};

/** The tokens of a request to Anthropic and of its answer */
type AnthropicUsage = {
  input_tokens?: number;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  output_tokens?: number;
};

/** A message of Anthropic's Messages API */
type AnthropicMessage = {
  role: "user" | "assistant";
  content: AnthropicBlock[];
};

/**
 * The headers of every request to Anthropic
 *
 * @param key - The API key
 * @returns The headers
 */
function anthropicHeaders(key: string): Record<string, string> {
  return {
    Authorization: `Bearer ${key}`,
    "anthropic-version": "2023-06-01",
    // without it, Anthropic does not answer pages in a browser (CORS)
    "anthropic-dangerous-direct-browser-access": "true",
    "Content-Type": "application/json",
  };
}

/**
 * Turns the conversation into the messages of Anthropic's Messages API
 *
 * Tool results go in a "user" message; messages that follow one another with
 * the same role are joined, since Anthropic expects the roles to alternate.
 * The answers of earlier turns go without their thinking.
 *
 * @param messages - The conversation
 * @param context - The context of the current turn
 * @param keepThinking - Whether the thinking of the current turn is sent back
 * @returns The messages
 */
function toAnthropicMessages(
  messages: ChatMessage[],
  context: string,
  keepThinking: boolean,
): AnthropicMessage[] {
  const start = turnStart(messages);
  const result: AnthropicMessage[] = [];
  function add(role: AnthropicMessage["role"], blocks: AnthropicBlock[]) {
    const last = result.at(-1);
    if (last?.role === role) {
      last.content.push(...blocks);
    } else if (blocks.length > 0) {
      result.push({ role, content: blocks });
    }
  }
  for (const [index, message] of messages.entries()) {
    switch (message.role) {
      case "user":
        add("user", [
          {
            type: "text",
            text:
              index === start
                ? withContext(message.text, context)
                : message.text,
          },
        ]);
        break;
      case "tool":
        add("user", [
          {
            type: "tool_result",
            tool_use_id: message.callId,
            content: message.result,
          },
        ]);
        break;
      case "assistant":
        add(
          "assistant",
          (originalItems(message, index, start, "anthropic", keepThinking) as
            AnthropicBlock[] | undefined) ?? [
            { type: "text", text: message.text },
            ...message.toolCalls.map((call) => ({
              type: "tool_use",
              id: call.id,
              name: call.name,
              input: call.arguments,
            })),
          ],
        );
        break;
    }
  }
  // Anthropic refuses empty text blocks, e.g. of an answer with only tool calls
  for (const message of result) {
    message.content = message.content.filter(
      (block) => block.type !== "text" || block.text !== "",
    );
  }
  return result.filter((message) => message.content.length > 0);
}

/**
 * Asks a model of Anthropic, with the Messages API; sends the request again
 * once without the thinking if Anthropic refuses it
 */
async function chatWithAnthropic(
  request: ChatRequest,
  onText: (text: string) => void,
  options?: { signal?: AbortSignal },
): Promise<ChatAnswer> {
  const answer = await sendToAnthropic(request, true, onText, options);
  return answer === thinkingRefused
    ? ((await sendToAnthropic(request, false, onText, options)) as ChatAnswer)
    : answer;
}

/**
 * Sends one request to Anthropic
 *
 * The answer arrives in content blocks: text, tool calls (whose arguments
 * arrive as pieces of JSON) and the model's thinking, which is kept to be sent
 * back as it came within the turn. The fixed beginning (the tools and the
 * instructions) is marked for prompt caching, so that later requests pay less
 * for it.
 *
 * @param request - The request
 * @param keepThinking - Whether to send the thinking of the current turn
 * @param onText - Called with the text so far
 * @param options - Optional abort signal
 * @returns The answer, or {@link thinkingRefused} if Anthropic refused the
 * thinking that was sent
 */
async function sendToAnthropic(
  { model, key, instructions, context, messages, tools }: ChatRequest,
  keepThinking: boolean,
  onText: (text: string) => void,
  options?: { signal?: AbortSignal },
): Promise<ChatAnswer | typeof thinkingRefused> {
  const { signal } = options ?? {};
  const response = await fetch(`${anthropicUrl}/messages`, {
    method: "POST",
    headers: anthropicHeaders(key),
    body: JSON.stringify({
      model,
      max_tokens: anthropicMaxTokens,
      system: [
        {
          type: "text",
          text: instructions,
          cache_control: { type: "ephemeral" },
        },
      ],
      messages: toAnthropicMessages(messages, context, keepThinking),
      tools: tools.map(({ name, description, parameters }) => ({
        name,
        description,
        input_schema: parameters,
      })),
      stream: true,
    }),
    signal,
  });
  if (!response.ok) {
    const error = await readError(response);
    // e.g. "Invalid `signature` in `thinking` block. The block is bound to a
    // different conversation."
    if (
      keepThinking &&
      response.status === 400 &&
      /signature|thinking/i.test(error.message)
    ) {
      return thinkingRefused;
    }
    return {
      error: explainApiError("Anthropic", response.status, error, key, model),
    };
  }
  const blocks: AnthropicBlock[] = [];
  // the arguments of tool calls, which arrive as pieces of JSON
  const json = new Map<number, string>();
  let error: string | undefined;
  const usage: Usage = { input: 0, cacheRead: 0, cacheWrite: 0, output: 0 };
  function textSoFar(): string {
    return blocks
      .filter((block) => block.type === "text")
      .map((block) => block.text ?? "")
      .join("\n\n");
  }
  /** Keeps the numbers that Anthropic sends, which are totals so far */
  function countTokens(tokens: AnthropicUsage | undefined): void {
    usage.input = tokens?.input_tokens ?? usage.input;
    usage.cacheRead = tokens?.cache_read_input_tokens ?? usage.cacheRead;
    usage.cacheWrite = tokens?.cache_creation_input_tokens ?? usage.cacheWrite;
    usage.output = tokens?.output_tokens ?? usage.output;
  }
  await readServerEvents<AnthropicEvent>(
    response,
    (event) => {
      const index = event.index ?? 0;
      const block = blocks[index];
      countTokens(event.message?.usage);
      countTokens(event.usage);
      switch (event.type) {
        case "content_block_start":
          if (event.content_block !== undefined) {
            blocks[index] = { ...event.content_block };
          }
          break;
        case "content_block_delta":
          if (block === undefined || event.delta === undefined) {
            break;
          }
          if (event.delta.text !== undefined) {
            block.text = (block.text ?? "") + event.delta.text;
            onText(textSoFar());
          }
          if (event.delta.thinking !== undefined) {
            block.thinking = (block.thinking ?? "") + event.delta.thinking;
          }
          if (event.delta.signature !== undefined) {
            block.signature = (block.signature ?? "") + event.delta.signature;
          }
          if (event.delta.partial_json !== undefined) {
            json.set(index, (json.get(index) ?? "") + event.delta.partial_json);
          }
          break;
        case "content_block_stop":
          if (block?.type === "tool_use") {
            block.input = parseArguments(json.get(index));
          }
          break;
        case "error":
          error ??= event.error?.message ?? "unknown error";
          break;
      }
    },
    { signal },
  );
  if (error !== undefined) {
    return { error: `Anthropic: ${hideKey(error, key)}` };
  }
  const items = blocks.filter((block) => block !== undefined);
  return {
    message: {
      role: "assistant",
      text: textSoFar(),
      toolCalls: items
        .filter((block) => block.type === "tool_use")
        .map((block) => ({
          id: block.id ?? newCallId(),
          name: block.name ?? "",
          arguments: (block.input ?? {}) as Record<string, unknown>,
        })),
      original: { api: "anthropic", items },
      usage,
    },
  };
}

/**
 * Checks an Anthropic key by listing the models, which costs nothing
 */
async function checkAnthropicKey(
  key: string,
  options?: { signal?: AbortSignal },
): Promise<string | undefined> {
  const { signal } = options ?? {};
  const response = await fetch(`${anthropicUrl}/models`, {
    headers: anthropicHeaders(key),
    signal,
  });
  return response.ok
    ? undefined
    : explainApiError(
        "Anthropic",
        response.status,
        await readError(response),
        key,
      );
}

// --- Ollama ---

/** The address of the Ollama server running on the user's computer */
export const ollamaUrl = "http://localhost:11434";

/**
 * The context window requested from Ollama, in tokens
 *
 * The instructions, the project summary and the tools alone take about 3000 to
 * 5500 tokens, while Ollama's default on many computers is 4096 tokens, which
 * would cut the text.
 */
const contextTokens = 16384;

/** The status message when nothing answers on a page opened from this computer */
export const notRunningMessage =
  'Ollama is not running. Start it from the Start menu (Windows) or the Applications folder (Mac), then click "Check again". Not installed yet? Click "Help".';

/** The message for Ollama's "unauthorized" error */
const notSignedInMessage =
  "Online models need you to be signed in to Ollama. Open the Ollama app, go to Settings and sign in (a free account is enough for gemma4:cloud). Then try again.";

/** The message for Ollama's "not found" error */
const notDownloadedMessage =
  'This model is not downloaded yet. Click "Download" next to the model menu.';

/** A message of the conversation, in the format of Ollama's chat API */
type OllamaMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  thinking?: string;
  tool_calls?: {
    function: { name: string; arguments?: Record<string, unknown> };
  }[];
  tool_name?: string;
};

/** One line of a streamed chat answer */
type OllamaChatPiece = {
  message?: Partial<OllamaMessage>;
  error?: string;
  /** In the last line: the tokens of the request */
  prompt_eval_count?: number;
  /** In the last line: the tokens of the answer */
  eval_count?: number;
};

/** One line of the progress of a model download */
export type OllamaPullUpdate = {
  status?: string;
  total?: number;
  completed?: number;
  error?: string;
};

/**
 * Whether TissUUmaps is open from this computer, whose pages Ollama accepts by
 * default
 *
 * @returns Whether the page is served from the local computer
 */
function isLocalPage(): boolean {
  return ["localhost", "127.0.0.1", "[::1]", ""].includes(
    window.location.hostname,
  );
}

/**
 * Explains why nothing answers at the Ollama address
 *
 * On a website, a second request with mode "no-cors" tells a blocked website
 * from a stopped Ollama: Ollama cannot block such a request, whose answer just
 * cannot be read.
 *
 * @param options - Optional abort signal
 * @returns The message for the user
 */
export async function explainNotReachable(options?: {
  signal?: AbortSignal;
}): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  if (isLocalPage()) {
    return notRunningMessage;
  }
  const origin = window.location.origin;
  try {
    await fetch(ollamaUrl, { mode: "no-cors", signal });
    return (
      `Ollama is running, but it blocks this website (${origin}): by default it only accepts pages ` +
      `opened from this computer. To allow it, set OLLAMA_ORIGINS to ${origin} and restart Ollama ` +
      '(see "Help", section 8). Then click "Check again".'
    );
  } catch {
    signal?.throwIfAborted();
    return (
      "Ollama does not answer. If it is not running, start it. If it is running, the browser may " +
      "block this page from reaching it: if the browser asks whether the page may access devices " +
      'on your local network, allow it, and see "Help", section 8. Then click "Check again".'
    );
  }
}

/**
 * Turns an error of Ollama into an instruction the user can follow
 *
 * @param error - The error message of Ollama
 * @returns The message for the user
 */
function explainOllamaError(error: string): string {
  if (/unauthorized/i.test(error)) {
    return notSignedInMessage;
  }
  if (/not found/i.test(error)) {
    return notDownloadedMessage;
  }
  return error;
}

/**
 * Turns the conversation into the messages of Ollama's chat API
 *
 * @param messages - The conversation
 * @returns The messages
 */
function toOllamaMessages(messages: ChatMessage[]): OllamaMessage[] {
  return messages.flatMap((message): OllamaMessage[] => {
    switch (message.role) {
      case "user":
        return [{ role: "user", content: message.text }];
      case "tool":
        return [
          { role: "tool", tool_name: message.name, content: message.result },
        ];
      case "assistant":
        if (message.original?.api === "ollama") {
          return message.original.items as OllamaMessage[];
        }
        return [
          {
            role: "assistant",
            content: message.text,
            ...(message.toolCalls.length === 0
              ? {}
              : {
                  tool_calls: message.toolCalls.map((call) => ({
                    function: { name: call.name, arguments: call.arguments },
                  })),
                }),
          },
        ];
    }
  });
}

/**
 * Asks a model through Ollama on the user's computer, which runs it there or,
 * for the online models, on Ollama's servers
 */
async function chatWithOllama(
  { model, instructions, context, messages, tools }: ChatRequest,
  onText: (text: string) => void,
  options?: { signal?: AbortSignal },
): Promise<ChatAnswer> {
  const { signal } = options ?? {};
  const response = await fetch(`${ollamaUrl}/api/chat`, {
    method: "POST",
    body: JSON.stringify({
      model,
      // Ollama checks no thinking, so the context stays in the system message,
      // where the small local models were tested with it
      messages: [
        { role: "system", content: `${instructions}\n\n${context}` },
        ...toOllamaMessages(messages),
      ],
      tools: tools.map((tool) => ({ type: "function", function: tool })),
      stream: true,
      options: { num_ctx: contextTokens },
    }),
    signal,
  });
  const message: OllamaMessage = { role: "assistant", content: "" };
  let error: string | undefined;
  const usage: Usage = { input: 0, cacheRead: 0, cacheWrite: 0, output: 0 };
  await readJsonLines<OllamaChatPiece>(
    response,
    (piece) => {
      error ??= piece.error;
      usage.input = piece.prompt_eval_count ?? usage.input;
      usage.output = piece.eval_count ?? usage.output;
      const part = piece.message ?? {};
      if (part.thinking !== undefined) {
        message.thinking = (message.thinking ?? "") + part.thinking;
      }
      if (part.tool_calls !== undefined) {
        message.tool_calls = [
          ...(message.tool_calls ?? []),
          ...part.tool_calls,
        ];
      }
      if (part.content) {
        message.content += part.content;
        onText(message.content);
      }
    },
    { signal },
  );
  if (error !== undefined) {
    return { error: explainOllamaError(error) };
  }
  return {
    message: {
      role: "assistant",
      text: message.content,
      toolCalls: (message.tool_calls ?? []).map((call) => ({
        id: newCallId(),
        name: call.function.name,
        arguments: call.function.arguments ?? {},
      })),
      original: { api: "ollama", items: [message] },
      usage,
    },
  };
}

// --- The providers ---

/**
 * The services the chat can use, in menu order: the first is the default
 *
 * The prices are those published by OpenAI and Anthropic in October 2026,
 * only to estimate the cost of the answers: the bill of the service counts.
 */
export const providers: Provider[] = [
  {
    id: "openai",
    name: "OpenAI",
    models: [
      {
        name: "gpt-6-luna",
        note: "fast, lowest cost",
        prices: { input: 0.1, cacheRead: 0.01, cacheWrite: 0.125, output: 0.5 },
      },
      {
        name: "gpt-6.1-sol",
        note: "more capable, costs more",
        prices: { input: 2, cacheRead: 0.1, cacheWrite: 2.5, output: 10 },
      },
      {
        name: "gpt-6-astra",
        note: "most capable, costs the most",
        prices: { input: 10, cacheRead: 1, cacheWrite: 12.5, output: 50 },
      },
    ],
    needsKey: true,
    wholeApp: true,
    chat: chatWithOpenAi,
    checkKey: checkOpenAiKey,
  },
  {
    id: "anthropic",
    name: "Anthropic",
    models: [
      {
        name: "claude-haiku-5-5",
        note: "fast, lowest cost",
        prices: { input: 0.1, cacheRead: 0.01, cacheWrite: 0.125, output: 0.5 },
      },
      {
        name: "claude-sonnet-5-5",
        note: "more capable, costs more",
        prices: { input: 2, cacheRead: 0.1, cacheWrite: 2.5, output: 10 },
      },
      {
        name: "claude-opus-5-5",
        note: "most capable, costs the most",
        prices: { input: 4, cacheRead: 0.2, cacheWrite: 5, output: 20 },
      },
    ],
    needsKey: true,
    wholeApp: true,
    chat: chatWithAnthropic,
    checkKey: checkAnthropicKey,
  },
  {
    id: "ollama",
    name: "Ollama (local)",
    models: [{ name: "gemma4:e2b", note: "local" }],
    needsKey: false,
    wholeApp: false,
    chat: chatWithOllama,
  },
  {
    id: "ollama-cloud",
    name: "Ollama (cloud)",
    models: [{ name: "gemma4:cloud", note: "online" }],
    needsKey: false,
    wholeApp: false,
    chat: chatWithOllama,
  },
];
