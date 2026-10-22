---
sidebar_position: 2
---

# AI chat

The **AI chat** plugin adds a chat panel in which you can ask questions about the open project and change how it is shown, in plain words: _"show only the cell labels"_, _"color the cells by leiden"_, _"cluster 3 in red, the others grey"_. The answers come from a language model that runs through [Ollama](https://ollama.com), either on your own computer or online.

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

While the chat is empty, a few example requests for the open project are shown; click one to send it.

:::caution

Language models make mistakes, especially small ones running on a laptop. Check the result in the viewer, and use _Undo_ when something went wrong. The chat shows every tool call the model makes and its result.

:::

## Installing Ollama

Download Ollama from [ollama.com/download](https://ollama.com/download) and install it like any other program. It then runs in the background. If the status line of the panel says that Ollama is not running, start it from the Start menu (Windows) or the Applications folder (Mac), then click _Check again_.

## Choosing a model

The menu at the top of the panel lists the models installed in Ollama and a few suggested ones:

- **local** models run on your computer. They are free, private and work offline, but can be slow on a computer without a good graphics card. `gemma4:e2b` is a small model that runs on most laptops.
- **not downloaded** models are local models that are not on your computer yet: choose one and click _Download_. This happens only once and takes a few GB of disk space.
- **online** models (their names end in `cloud`, e.g. `gemma4:cloud`) run on Ollama's servers. They are fast on any computer and make fewer mistakes, but your messages and the project summary are sent to Ollama. `gemma4:cloud` is included in Ollama's free plan.

The panel remembers the chosen model in the browser for the next session.

### Online models

Online models need you to be signed in to Ollama: open the Ollama app, go to _Settings_ and sign in, or create a free account there. In a terminal, `ollama signin` does the same.

## Using TissUUmaps from a website

For safety, Ollama only accepts requests from pages opened from your own computer, such as a TissUUmaps started with `pnpm run dev`. When TissUUmaps is opened from a website, for example [tissuumaps.github.io](https://tissuumaps.github.io), Ollama blocks it, and the status line says so: _"Ollama is running, but it blocks this website"_. This applies to online models too, since they are also reached through Ollama on your computer.

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

The plugin is implemented in the [`@tissuumaps/plugins`](../api/@tissuumaps/plugins/index.md) package as `aiChatPlugin`, and registered by the application on startup (see [Plugins](../development/plugins.md)). It talks to Ollama's [chat API](https://docs.ollama.com/api) at `http://localhost:11434`.
