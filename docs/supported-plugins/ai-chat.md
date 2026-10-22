---
sidebar_position: 2
---

# AI chat

The **AI chat** plugin adds a chat panel in which you can ask questions about the open project and change how it is shown, in plain words: _"show only the cell labels"_, _"color the cells by leiden"_, _"cluster 3 in red, the others grey"_. The answers come from a language model of [OpenAI](https://platform.openai.com) or [Anthropic](https://platform.claude.com), with your own API key, or from a model that runs through [Ollama](https://ollama.com), on your own computer or online.

Open the panel from the plugins menu (the kebab button in the tab bar), by picking _AI chat_.

## What it can do

The assistant can:

- answer questions about the open project: its images, labels, points and shapes, and the columns of their tables;
- show or hide objects, or show only some of them;
- make objects more or less transparent, and make points bigger or smaller;
- color labels and points by a column of their table: groups get a color each (e.g. clusters or cell types), numbers a color scale (e.g. counts, or the expression of a gene such as `X[EPCAM]`), with the palette and value range you ask for;
- give chosen colors to some groups (_"T cell in blue, the others grey"_), or one color to all items;
- show only some groups of labels or points (_"show only clusters 1 and 3"_), and show all of them again.

It cannot change or delete anything else, and it cannot filter by numbers (e.g. _"cells with an area above 50"_).

Every request sends the model a summary of the project: the objects, how they are colored, and the useful columns of their tables. _Show context_ shows exactly what the model receives.

All changes go through the project, like changes made by hand. _Undo_ takes back the changes of the last request, and _Undo all_ takes back every change made by the chat; settings changed by hand in the meantime are left as they are. _New chat_ starts a new conversation; Undo keeps working for the earlier changes.

A few suggested requests for the open project are shown above the text box; click one to send it, without losing what you are typing. In the text box, the up and down arrows go through the messages you sent before, like in a terminal. To refer to a part of an answer, select it and click _Quote_: it is quoted in your next message.

While the model thinks, three dots blink next to an Italian word that changes every few seconds (_pensando_, _impastando_, _sbirciando nelle cellule_...); point at the word to see what it means in English.

### Dia and Aid

The assistant follows the theme of TissUUmaps. With the light theme it is **Dia**, gentle and kind; with the dark theme it is **Aid**, serious and a little mysterious. The name is shown in the chat instead of the model's, and the tone changes as soon as you switch the theme. Only the tone changes: what the assistant can do, and its rules (such as telling the truth about what it changed, or asking when a request is not clear), stay the same.

:::caution

Language models make mistakes, especially small ones running on a laptop. Check the result in the viewer, and use _Undo_ when something went wrong. The chat shows every tool call the model makes and its result.

:::

## Choosing a service

The settings at the top of the panel choose the service that answers and its model. The panel remembers both in the browser for the next session.

| Service            | Runs on                                           | Needs                        | Cost                      |
| ------------------ | ------------------------------------------------- | ---------------------------- | ------------------------- |
| **OpenAI**         | OpenAI's servers                                  | your own OpenAI API key      | billed to your account    |
| **Anthropic**      | Anthropic's servers                               | your own Anthropic API key   | billed to your account    |
| **Ollama (local)** | your computer                                     | Ollama installed             | free                      |
| **Ollama (cloud)** | Ollama's servers, through Ollama on your computer | Ollama installed, an account | free plan for some models |

With OpenAI, Anthropic and Ollama (cloud), your messages and the project summary are sent to that service. OpenAI and Anthropic are fast on any computer and also work when TissUUmaps is opened from a website; a local model is free, private and works offline, but can be slow on a computer without a good graphics card.

## Using OpenAI or Anthropic

TissUUmaps does not provide API keys: everyone uses their own. An API key is a secret code that lets the chat use your account, and every request is billed to that account, so first add a payment method or buy credit there. With the first model in the menu (`gpt-6-luna` or `claude-haiku-5-5`), a request usually costs well under one cent; the other models cost more.

### Getting a key

- **OpenAI**: sign in at [platform.openai.com](https://platform.openai.com), open the [API keys](https://platform.openai.com/api-keys) page and click _Create new secret key_.
- **Anthropic**: sign in at [platform.claude.com](https://platform.claude.com), open the [API keys](https://platform.claude.com/settings/keys) page and click _Create key_. Choose one workspace for the key (e.g. the _Default Workspace_), otherwise every request would need to name the workspace, and choose when the key expires.

Copy the key (the website shows it only once: if you lose it, make a new one), choose the service in the panel and paste the key into the _API key_ field. The panel checks the key with a request that costs nothing, and the status line says whether it was accepted.

When the key is accepted, the settings fold into one short line, e.g. _Anthropic · claude-haiku-5-5 · key ✓_, and the chat says once what using the key means. The ⚙ button opens the settings again, to change, forget or check the key; they also open by themselves when the key stops working or something goes wrong.

### Keeping your key safe

- Treat the key like a password: do not share it, and do not put it in e-mails, files or screenshots.
- Make a key just for TissUUmaps, and set a monthly spending limit in your account (under _Limits_), so that a stolen key cannot cost much. With Anthropic, you can also make the key expire.
- If someone else may have your key, delete it on the page where you made it: it stops working at once. Then click _Forget key_ in the panel.

The key is sent only to the service it belongs to, and never shown in the panel, in _Show context_ or in error messages.

Without _Remember key on this computer_, the key is forgotten when the page is closed. With it, the key is saved in the browser's storage for TissUUmaps, where anyone who uses this browser on this computer, and any browser extension or script that can read the page, could find it. Do not use it on a shared computer. _Forget key_ removes the key from the panel and from the browser.

:::caution

OpenAI advises against using API keys in code that runs in a browser, and Anthropic calls the header that allows it "dangerous", for the same reason: a key in a browser can be read by others. This is why the plugin never comes with a key, and why your own key should have a spending limit.

:::

## Installing Ollama

Download Ollama from [ollama.com/download](https://ollama.com/download) and install it like any other program. It then runs in the background. If the status line of the panel says that Ollama is not running, start it from the Start menu (Windows) or the Applications folder (Mac), then click _Check again_.

## Choosing an Ollama model

With _Ollama (local)_ and _Ollama (cloud)_, the model menu lists the models installed in Ollama and a few suggested ones:

- **local** models run on your computer. They are free, private and work offline, but can be slow on a computer without a good graphics card. `gemma4:e2b` is a small model that runs on most laptops.
- **not downloaded** models are local models that are not on your computer yet: choose one and click _Download_. This happens only once and takes a few GB of disk space.
- **online** models (with _Ollama (cloud)_; their names end in `cloud`, e.g. `gemma4:cloud`) run on Ollama's servers. They are fast on any computer and make fewer mistakes, but your messages and the project summary are sent to Ollama. `gemma4:cloud` is included in Ollama's free plan.

### Online models

Online models need you to be signed in to Ollama: open the Ollama app, go to _Settings_ and sign in, or create a free account there. In a terminal, `ollama signin` does the same.

## Using Ollama when TissUUmaps is opened from a website

OpenAI and Anthropic need nothing for this. For safety, Ollama only accepts requests from pages opened from your own computer, such as a TissUUmaps started with `pnpm run dev`. When TissUUmaps is opened from a website, for example [tissuumaps.github.io](https://tissuumaps.github.io), Ollama blocks it, and the status line says so: _"Ollama is running, but it blocks this website"_. This applies to online models too, since they are also reached through Ollama on your computer.

To allow the website, give its address to Ollama with the `OLLAMA_ORIGINS` setting, then restart Ollama:

- **Windows**: quit Ollama (right-click its icon next to the clock, then _Quit_). Then either run this command once, in the Command Prompt or in PowerShell:

  ```sh
  setx OLLAMA_ORIGINS "https://tissuumaps.github.io"
  ```

  or open _Settings_, search for "environment variables", choose _Edit environment variables for your account_, click _New_, and enter the name `OLLAMA_ORIGINS` and the value `https://tissuumaps.github.io`. Start Ollama again from the Start menu. The setting is kept when the computer restarts.

- **Mac**: run this command in the Terminal app, then quit the Ollama app and start it again:

  ```sh
  launchctl setenv OLLAMA_ORIGINS "https://tissuumaps.github.io"
  ```

  This setting is **lost when the Mac restarts**: run the command again after a restart, then restart Ollama.

Then click _Check again_ in the panel. If the browser asks whether the page may access devices on your local network, allow it; otherwise the browser itself blocks the page, and the status line says that Ollama does not answer.

See also [Ollama's FAQ](https://docs.ollama.com/faq) on `OLLAMA_ORIGINS`.

## Limitations

- **Gene names**: genes can be named, as in `X[EPCAM]`, only if TissUUmaps could read the gene names from the index of `var/` of an AnnData table. Otherwise, the genes only have numbers (`X[0]`, `X[1]`, ...), and the project summary says that the gene names are not available.
- **Very big tables**: in tables with more than 5 million rows, such as Xenium transcripts, only the columns used now (e.g. for colors) are described to the model, since reading the others would take too much time and memory.
- **Several cluster columns**: if labels or points have several cluster columns (e.g. `leiden` and `kmeans`) and are not colored by one of them, the model is asked to check with you which one you mean.
- **Other objects drawn in the same place**: if another visible object shows the same items (e.g. the same labels added twice), it can cover a filter or new colors. The chat then shows a note.

## API

The plugin is implemented in the [`@tissuumaps/plugins`](../api/@tissuumaps/plugins/index.md) package as `aiChatPlugin`, and registered by the application on startup (see [Plugins](../development/plugins.md)). It talks directly from the browser to:

- OpenAI's [Responses API](https://developers.openai.com/api/docs/guides/function-calling) at `https://api.openai.com/v1/responses`, with `store: false`, so that OpenAI keeps nothing between requests;
- Anthropic's [Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages) at `https://api.anthropic.com/v1/messages`, with the `anthropic-dangerous-direct-browser-access` header, without which Anthropic does not answer pages in a browser;
- Ollama's [chat API](https://docs.ollama.com/api) at `http://localhost:11434`.

The conversation is kept in one format of the plugin's own, and each service's part of the plugin turns it into the format of that API, so that the tools work the same with all of them.

OpenAI and Anthropic only take the model's thinking back while everything sent before it is unchanged, and bill less for a beginning they have already seen ([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). So with them the beginning of every request stays the same: the fixed instructions, the tools, and the conversation, to which messages are only added. The project summary and the personality, which change, go with the user's message that starts each turn, and stay the same while the model calls tools in that turn. The thinking is sent back only within the turn; if a service still refuses it, the plugin sends the request once more without it.

The names, tones and avatars of Dia and Aid, and the words shown while the model thinks, are in `persona.ts`, next to the plugin. The panel uses TissUUmaps' theme colors (CSS variables such as `--foreground` and `--muted`), so that it stays readable with the light and the dark theme.

---

_Made by Alessia Giustolisi_
