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
- show only some groups of labels or points (_"show only clusters 1 and 3"_), and show all of them again;
- with OpenAI and Anthropic, compute on the data what the summary does not tell (_"how many cells are in each cluster?"_), with JavaScript it writes and runs in a sandbox (see [Computing on the data](#computing-on-the-data));
- with OpenAI and Anthropic, make plugins for TissUUmaps: panels with buttons, sliders, interactive charts and tools, which you run after reading their code (see [Plugins made by the chat](#plugins-made-by-the-chat));
- with OpenAI and Anthropic, and only if you allow it, search the web (see [Web search](#web-search)).

It cannot change or delete anything else, and it cannot filter by numbers (e.g. _"cells with an area above 50"_). It cannot zoom or move the view yet: use the mouse wheel to zoom, drag the image to move it, and click the button with the house at the top right of the viewer (_Reset view_) to see the whole image again.

Every request sends the model a summary of the project: the objects, how they are colored, the useful columns of their tables, and the columns of their coordinates, for spatial analyses (the columns that points are drawn from, or else e.g. `obsm/spatial[0]` and `obsm/spatial[1]`, `geometry[x]` and `geometry[y]`, or `x_centroid` and `y_centroid`). _Show context_ shows exactly what the model receives.

All changes go through the project, like changes made by hand. _Undo_ takes back the changes of the last request, and _Undo all_ takes back every change made by the chat; settings changed by hand in the meantime are left as they are. _New chat_ starts a new conversation; Undo keeps working for the earlier changes.

The answers are shown with their Markdown: lists, headings, **bold**, `code`, links (which open in a new tab) and tables, with aligned columns (numbers to the right) and light borders in the colors of the theme. The text of an answer is never used as HTML.

A few suggested requests for the open project are shown above the text box, on one or two quiet lines; click one to send it, without losing what you are typing. In the text box, the up and down arrows go through the messages you sent before, like in a terminal. To refer to a part of an answer, select it and click _Quote_: it is quoted in your next message. _Copy_, next to the name of the assistant, copies a whole answer as the model wrote it (with its Markdown), and the code of a script has its own _Copy_.

While the model works, until its final answer (also while a tool runs, and between one tool and the next), three dots blink next to an Italian word that changes every few seconds (_pensando_, _impastando_, _sbirciando nelle cellule_...), below the last message; point at the word to see what it means in English.

Each tool call is one short line, with an icon and the start of its result, e.g. _🌫 Done: …_; click it to see the call and its whole result. In one request, the model may call tools 15 times in a row with OpenAI and Anthropic (e.g. to load a skill, read its files, find the columns and run the code), and 5 times with Ollama; then the panel asks it to answer without tools, with what it found, and says so in a note. A tool call that already failed is not run again: the panel stops the request instead. _Show tool details_, in the settings (⚙), keeps them all open, and the browser remembers it. Errors and notes are always shown.

The settings (⚙) also have _Let the model see charts_ (see [Charts](#charts)) and _Allow web search_ (see [Web search](#web-search)). Both are off until you turn them on, since they cost more, and the browser remembers them.

### Dia and Aid

The assistant follows the theme of TissUUmaps. With the light theme it is **Dia**, gentle and kind; with the dark theme it is **Aid**, serious and a little mysterious. The name is shown in the chat instead of the model's, and the tone changes as soon as you switch the theme. Only the tone changes: what the assistant can do, and its rules (such as telling the truth about what it changed, or asking when a request is not clear), stay the same. The _About_ button, and _The AID of DIA_ at the bottom of the panel (next to _Made by Alessia Giustolisi_), tell where the two names come from.

### About: The AID of DIA

Dia (light theme) and Aid (dark theme) are the two moods of the same assistant: read one name backwards and you get the other. Aid is what it gives you: help.

In ancient Greek, διά (diá) means "through". It is the root of διάλογος (diálogos), dialogue, "through the word", and of διάγνωσις (diágnōsis), diagnosis, "through knowledge". Through dialogue and through knowledge, the assistant helps you explore your data and do science.

Made by [Alessia Giustolisi](https://github.com/ale2sia-g).

:::caution

Language models make mistakes, especially small ones running on a laptop. Check the result in the viewer, and use _Undo_ when something went wrong. The chat shows every tool call the model makes and its result (click its line).

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

TissUUmaps does not provide API keys: everyone uses their own. An API key is a secret code that lets the chat use your account, and every request is billed to that account, so first add a payment method or buy credit there. With the first model in the menu (`gpt-6-luna` or `claude-haiku-5-5`), a request usually costs well under one cent; the other models cost more. Under each answer, a click on _Usage_ shows the tokens used (input, read from and written to the service's prompt cache, output) and about how much they cost, also in total since the panel was opened. These are estimates from the published prices: the bill of the service counts.

These models also get the types and the current state of the whole app (see [API](#api)), so they can answer detailed questions about the settings of each object, e.g. _"which properties of the labels can I change?"_, and change most of them: _"make channel 1 of morphology_focus red"_, _"show only the DAPI channel of morphology_focus"_, _"show the transcripts as squares"_, _"rotate the H&E image by 90 degrees"_, _"put the H&E image on top"_, _"highlight cluster 3 of the cell labels"_. They use the same tools as with Ollama when those are enough, and otherwise call one of the allowed actions of the app state. Every change goes through Undo. They never delete or add objects or change where their data comes from, and they rename something only when you ask.

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

## Skills

A skill gives the assistant instructions, and maybe scripts, for a task you want done your way, such as the colors of your lab or the method of an analysis. Skills use the [Agent Skills](https://agentskills.io/specification) format: a folder with a `SKILL.md` file, which starts with the skill's name and description, followed by its instructions. The folder may hold other files, such as scripts.

In the settings (⚙), _Add skill…_ adds the skills of a folder: choose the folder of one skill, or a folder with one folder per skill. The browser may ask whether to upload the files to the page: they stay on your computer. With _Remember this skill_, which is on by default, the browser keeps the skill for the next sessions, in every project, also in projects opened from a website; without it, the skill is kept until the page is closed. Each skill in the list can be turned off and on again, remembered or not (_Remember_), and removed. Click a description to read all of it, and again to fold it into one line.

A project can also bring its own skills: put them in a folder `skills` next to the project file, with one folder per skill. When the project is opened from the connected folder, the panel reads that folder. When it is opened from a website, the panel cannot list a folder, so `skills/index.json` lists the files of the skills, as paths inside the `skills` folder:

```json
{ "files": ["lab-colors/SKILL.md", "lab-colors/references/palette.md"] }
```

In the list, the skills of the project are marked _project_ and those added by hand _added_. The skills of the project change when another project is opened, and cannot be remembered or removed, only turned off. If a skill added by hand has the same name as one of the project, the one added by hand is used while it is on.

A skill must follow the rules of the format: its name is lowercase, with only letters, digits and hyphens, and is also the name of its folder, and it has a description. Fields of `SKILL.md` that are not part of the format, such as those of other agents, are ignored, and the list says so.

With OpenAI and Anthropic, each request tells the model the names and descriptions of the skills that are on. When a request matches the description of a skill, the model loads its instructions (a line _📘 load_skill_ in the chat) and follows them with its tools; it reads the other files of the skill, such as references, only when the instructions point to them (_📄 read_skill_file_). The instructions of a skill may also tell it to use another skill. So the description says what the skill is for and when to use it, and the instructions say how. Ollama's models do not use skills.

A skill with only instructions can already do a lot with the tools of the chat, for example the color conventions of a lab:

```markdown
---
name: lab-colors
description: The color conventions of our lab for cell types and clusters. Use when the user asks to color cells "with our colors".
---

Color the cell types with set_group_colors: T cells blue, B cells orange,
macrophages green, all others light grey. If there is no column of cell
types, color the clusters with the palette batlowS.
```

### Scripts

A skill can also bring scripts, in JavaScript or Python, for what the tools of the chat cannot do, such as computing on the data. Its instructions say when to run which script, and with which arguments; the model runs it with _🧪 run_script_. Every script is shown in the chat before it runs, as one line like those of the tools, e.g. _📜 cluster-sizes/scripts/cluster-sizes.js_: click it to read its code. While a script runs, a row shows for how long, and its _Stop_ button stops it; a script that runs longer than two minutes is stopped.

A script is a JavaScript module whose default export is a function. It gets an object `tm`, and what it returns goes to the model, as text (objects and lists as JSON); what it logs with `console.log` goes along with it:

```js
// scripts/cluster-sizes.js: the rows of each group of a column
export default async function (tm) {
  const { table, column } = tm.args;
  const values = await tm.read(table, column);
  const counts = {};
  for (const value of values) {
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return counts;
}
```

| `tm`                                       | What it gives                                                                                                                                                                                         |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tm.args`                                  | the arguments the model gave, as the instructions of the skill describe them                                                                                                                          |
| `tm.project`                               | a copy of the state of the project, the same JSON the model gets                                                                                                                                      |
| `await tm.tables()`                        | the tables of the project, with their `id` and `name`                                                                                                                                                 |
| `await tm.columns(table)`                  | the columns of a table, with `matrix: true` for matrices such as `X` or `obsm/spatial`, each with its `size`, its first `columns` and `how` to read them, and `coordinates: true` on the coordinates  |
| `await tm.columns(table, "X")`             | the columns of a matrix, e.g. `X[EPCAM]`                                                                                                                                                              |
| `await tm.read(table, column)`             | the values of a column, e.g. `obs/leiden`, `X[EPCAM]` or `obsm/spatial[0]` (tables of at most 5 million rows); a matrix is read one column at a time, and its name alone gives an error that says how |
| `await tm.readFile(path)`                  | the text of another file of the skill                                                                                                                                                                 |
| `tm.progress(text)`                        | shows how far the script is, next to the _Stop_ button                                                                                                                                                |
| `await tm.saveColumn(table, name, values)` | saves a result as a column, one value per row (see [Results on the image](#results-on-the-image))                                                                                                     |

A Python script (`.py`) defines `async def main(tm):` instead, and gets the same `tm`, with the names of Python: `await tm.read(table, column)` gives a numpy array, `await tm.read_file(path)` reads another file of the skill, and `await tm.save_column(table, name, values)` saves a column. It runs in the Python of the chat (see [Computing on the data](#computing-on-the-data)), and can import the other Python modules of its skill, as Python would: those next to the script, and those at the top of the skill folder (and in its `src` folder, if it has one). The packages of Pyodide that the script imports, also through these modules, are loaded first.

```python
# scripts/centers.py: the center of each group, with a package of the skill
from cluster_tools import group_centers


async def main(tm):
    table = tm.args["table"]
    groups = await tm.read(table, tm.args["column"])
    x = await tm.read(table, tm.args["x"])
    y = await tm.read(table, tm.args["y"])
    return group_centers(groups, x, y)
```

Scripts run in a sandbox, which keeps them away from everything else: they can read the data through `tm`, but get only copies, so they cannot change the project; they cannot use the network, load other code, use `eval`, or reach the page, its storage or the API key. To change what is shown, the model uses its tools with what the script returned. Each Python run gets fresh names, and nothing of the skill of the run before: its files and its modules are removed first.

The scripts of a skill that came with the project run only if you allow it: the first time one of them would run, the panel opens its code in the chat and asks. Skills added by hand are yours, so their scripts run without asking. The Python scripts of a skill of the project run in a Python of their own, started again when the chat goes back to its own code, so that code from elsewhere cannot change the modules (e.g. numpy) that the code of the chat uses, nor the other way round.

### Skills from GitHub

A skill can also come from a GitHub repository, or from a folder of one: _Add skill from GitHub…_, in the settings (⚙), asks for its address, e.g. `https://github.com/wahlby-lab/analytical-enrichment-test` or `https://github.com/owner/repository/tree/main/skills/my-skill`. With OpenAI and Anthropic, you can also give the address in the chat (_"use https://github.com/…"_): the assistant then proposes the skill, with the same question as the settings, and nothing is added unless you say yes.

Before anything is added, the panel reads the repository and shows, in the chat:

- where it comes from: the repository (and its folder), and the exact commit, whose files are taken as they are;
- its license; a repository without a license gets a clear warning, since the rights to use, change and share its code are then not given to you. A folder of a repository takes with it the license file of the repository (e.g. `LICENSE`, from its top), since that license covers it, so that the assistant can read it and the plugins based on the skill can quote it;
- how many files it holds, and their size; only text and code are kept (`.py`, `.js`, `.md`, `.txt`, `.json`, `.toml`, `.yaml`, the license, and notebooks `.ipynb` without their outputs), at most 2 MB each, 10 MB and 300 files in all; the files left out are listed, with the reason;
- the Python packages that its code needs, from its imports and from `requirements.txt` or `pyproject.toml`, and whether [Pyodide](https://pyodide.org/en/stable/usage/packages-in-pyodide.html) has them; for those it does not have, which files cannot work, also through other files of the repository that import them;
- a warning that this is code written by other people: its `SKILL.md`, its README and its other texts become instructions for the assistant, and its scripts run without asking again.

Your yes is the only question: the scripts of a skill from GitHub then run like those of the skills you added. The skill is marked _GitHub_ in the list, with its repository, commit and license in the tooltip, and is remembered like the others (_Remember this skill_). A repository without `SKILL.md` gets a minimal one, written by the plugin, which says where the code comes from and how the assistant can learn to use it: from its README, its code and its notebooks (which the assistant reads as text, cell after cell). Once it has used the code, the assistant can propose better instructions (a new `SKILL.md`); it shows them in the chat, and they replace the old ones only if you say yes.

To use the Python code of a skill, the assistant writes its own code, with the files of the skill: the code can import the modules of the skill, e.g. `from anhood import anhood`, and `tm.read_file` reads its files. A skill whose folder is a Python package itself (an `__init__.py` at its top, often with relative imports, e.g. the folder `points2regions` of a repository) is also imported by its name, e.g. `import points2regions`; _load_skill_ tells the assistant so. The code of a skill from GitHub, and the code of the assistant that uses it, run in a Python of their own, like the scripts of a skill of the project. Only the packages of Pyodide can be imported: the parts of a repository that need others (e.g. squidpy or scanpy) cannot run, and the assistant says so.

## Computing on the data

With OpenAI and Anthropic, the assistant can also answer questions that need a computation on the data, such as _"how many cells are in each cluster?"_ or _"what is the mean expression of EPCAM in cluster 3?"_. It writes a short program in JavaScript or Python for the question and runs it with _🧮 run_code_, in a sandbox like that of the [scripts of skills](#scripts) and with the same `tm` (without `tm.args` and `tm.readFile`): the code can read the data, but cannot change the project, use the network or see your API key. No skill is needed: the assistant can compute statistics (counts, means, medians, quantiles, per group), compare groups, columns or genes, look for correlations, count what lies above a threshold, and do spatial analyses with the coordinates of the cells or points (distances, neighbors, density). If a skill that is on does the task, the assistant uses the skill, since it holds the method you want.

JavaScript starts at once, and the assistant uses it for counts, means and other simple computations. For statistical tests, models or clustering it uses Python, with numpy, scipy, pandas, scikit-learn, statsmodels and the other [packages of Pyodide](https://pyodide.org/en/stable/usage/packages-in-pyodide.html). In Python, the code defines `async def main(tm):`, and `await tm.read(table, column)` gives a numpy array. Python has no internet either: only the packages of Pyodide can be imported, and nothing can be installed (no `pip` or `micropip`).

The first time, Python is downloaded from [jsDelivr](https://www.jsdelivr.com/): about 14 MB to start, and the packages that the code imports when it first needs them (about 37 MB with numpy, scipy, pandas and matplotlib). While it downloads, the row of the tool shows how far it is (e.g. _Downloading Python, 12 of 37 MB_), with _Stop_. Each file is checked against its SHA-256 before it is used, and kept in this browser, so that the next time Python starts without downloading anything (in a few seconds); _Forget Python files_, in the settings (⚙), removes them. Python can only be checked, and so used, in a secure page (https, or localhost). Once started, Python stays for the next code of the chat; it starts again after _Stop_, _New chat_, or when another project is opened. The two minutes of the time limit do not count the download and the start.

It cannot yet read the pixels of images. The code cannot change the data of the project, but it can save its results as new columns, to color or filter by them (see below).

The code is shown in the chat before it runs, as one line such as _📜 Dia's code: cells per cluster_: click it to read the code, and _Copy_ copies it. It runs without asking, like the scripts of the skills you added. While it runs, a row shows for how long, and its _Stop_ button stops it; code that runs longer than two minutes is stopped. The assistant answers from what the code returned, and says so if the code failed.

### Charts

In Python, the assistant can also draw charts with [matplotlib](https://matplotlib.org), e.g. _"draw a bar chart of the cells per cluster"_ or _"show the z-scores as a heatmap"_. Every figure that the code leaves open is shown in the chat as an image (at most 10 for each run), numbered for the chat (_Chart 1_, _Chart 2_, …), with _Copy_, which copies the image, and _Save_, which saves it as a PNG file named after the code (e.g. _cells-per-cluster.png_). The charts keep a white background, also with the dark theme.

Usually the model does not see the image: it only gets a short description of each chart, with its titles, its axes and what each plot holds (e.g. _8 bar(s)_), so that it can say what the chart is about without making up what it looks like. To have it look at one, ask, e.g. _"look at the chart and tell me what you see"_: the model calls _🖼 look_at_chart_, and the panel sends it the image of that chart with the result of the tool, for that turn only; in the next turns the image is not sent again (the model can look again). With _Let the model see charts_, in the settings (⚙), the image of every chart goes to the model with the result of the code that drew it. Only OpenAI and Anthropic get images. An image costs about as many input tokens as a page of text (with Anthropic, about one token for every 750 pixels), which _Usage_ includes, and it says how many chart images were sent.

Groups named by numbers written as text, such as the clusters `"0"`, `"1"`, … `"15"` of leiden, are told to be sorted as numbers (0, 1, 2, …, 10), not as text (0, 1, 10, 11, …, 2), in results, files and charts, e.g. the rows and columns of a heatmap.

Before it shows an image, the panel checks that it is really a PNG file of a reasonable size (at most 5 MB, and 10,000 pixels on a side): the sandbox gives back only data, never code for the page.

### Files for you

When you ask for a file, e.g. _"give me the z-scores as CSV"_, the code can offer one with `await tm.download(name, text)` (in JavaScript and Python), e.g. `await tm.download("zscores.csv", frame.to_csv())`. The chat shows it as _📄 zscores.csv_, with _Copy_ and _Save_: nothing is saved unless you click. Only text files can be offered (`.csv`, `.tsv`, `.txt`, `.json`, `.md`, at most 10 MB), with a plain name, without folders.

### Results on the image

To show a result on the image, e.g. _"color the cells by the number of their neighbors within 30 µm"_, the code saves it as a column, one value per row of a table: `await tm.saveColumn(table, name, values)` in JavaScript, `await tm.save_column(table, name, values)` in Python (with `{ ids }` or `ids=` for values of some of the items only). The values go straight from the sandbox to the panel, never through the model. The panel keeps the column in a table of its own, e.g. _cells (chat)_ for the table _cells_, with the same items, so that the colors and filters can use it like the other columns: numbers get a color scale, text one color per group. The original data is never changed. The chat shows each new column in a row, e.g. _📊 New column neighbors in cells (chat)_; _Undo_ takes it back, with the colors that used it. At most 20 columns can be saved in all.

A computed table is temporary: its values live in this page. To keep it, click _Keep as CSV_ in its row, or ask the assistant to save it: after you confirm, it is written as a CSV file next to the project file, in the folder `computed`, with the item IDs in the column `id`, and the project reads the table from that file from then on, also once saved and opened again, even without the chat. This needs a project opened from a connected folder, and the browser asks for the permission to write into it. Otherwise, the file is downloaded instead, as with _Download CSV_, and the table stays temporary. Both buttons ask for the name of the file, and propose one with the table, its columns and the day, e.g. `cells-neighbors-2026-11-05.csv`; you can also tell the assistant the name (_"save it as neighbors_cluster3.csv"_). Names with folders or characters that files cannot have are refused, and `.csv` is added if it is missing. If the folder `computed` already has a file of that name, you are asked before it is replaced.

A project saved with a temporary table still works when it is opened again: in the same page, the table is still there; after the page was reloaded, its values are gone, so TissUUmaps says that the table could not be loaded, and what was colored by it gets its default color again.

## Plugins made by the chat

With OpenAI and Anthropic, the assistant can also make a plugin for TissUUmaps, when you ask for a panel or a tool to keep: _"make me a plugin that runs anhood on the clusters I choose, with a slider for k and an interactive heatmap"_. A plugin made by the chat is a real plugin: it has its own panel, opened from the plugins menu, with buttons, fields, sliders, interactive charts or tools, and it keeps working with the chat closed. It can read the project and its tables, and run JavaScript and Python in the [sandbox of the chat](#computing-on-the-data), also with the modules of a skill, e.g. one from GitHub.

Before it proposes a plugin, the assistant tells you what it will do and every delicate thing it does. Then it calls _🧩 propose_plugin_, and the chat shows the plugin before anything runs:

- its name, its id and what it does, and what it replaces, if the chat made a plugin with the same id before;
- where it comes from, if it uses the code or the method of a skill: the repository and the commit, with links, and the license, or a warning in red if the repository has no license, since then the rights to use, change and share its code are not given to you;
- its code, which opens with a click and can be copied;
- what it uses (e.g. _a panel of its own_, _the sandbox of the chat, for Python_, _the files of the skill anhood_), and a warning in red for each delicate thing that it does: the network, the memory of the browser (where the chat keeps your API keys if you chose _Remember key on this computer_), both of them together, changes to the project, changes to the view or the settings of TissUUmaps, code loaded from the internet, code or names built while it runs, other plugins. These are found by reading the code, which may miss things: run only plugins you trust.

_Run plugin_ runs it and opens its panel; _Cancel_ runs nothing. With _Remember this plugin_, the browser keeps it for the next sessions. A plugin runs in the page, with the same access to TissUUmaps as the plugins you load from a file: what it changes in the project is **not** taken back by _Undo_ in the chat, and the assistant says so. At the bottom of its panel, the chat adds a thin bar: _"This plugin was made with the AID of DIA, the AI agent of TissUUmaps"_.

The settings (⚙) list the _Plugins made by the chat_, each with:

- _Open_: opens its panel. A plugin kept by the browser runs again only once you say yes, once for each plugin each time the page is loaded: the chat shows it again first, with its code;
- _Show code_: shows its code in the chat;
- _Remember_: keeps it in the browser for the next sessions, or forgets it there;
- _Remove_: closes its panel and forgets it, after asking; what it changed in the project stays;
- _Export .js_: saves it as a JavaScript file, e.g. `chat-anhood.js`, to keep or to share. _Add plugin file…_ runs such a file again, here or in another browser, after the same question.

To fix or improve a plugin, ask the assistant. Each request tells it which plugins the chat made, with their ids, and _🧩 read_plugin_ gives it the current code, so that it starts from that code, keeps every feature, and proposes a new version with the same id. The question then says _Replaces_, with the number of lines of the code before and now, and once you say yes the new version takes the place of the old one: the list never holds the same plugin twice. A plugin proposed with the name of one made before replaces it too.

A plugin can also show temporary shapes on the image, drawn by TissUUmaps in the coordinates of the cells, e.g. a square of the size of a pixel of a method and a circle of its smoothing radius: they are listed under _Shapes_ while they are shown, and go away when the plugin removes them or is removed.

Plugins do not draw their charts from nothing: the chat gives them ready-made ones, a heatmap, a bar chart and a scatter plot, which have the names or numbers on the axes (groups named by numbers in number order), a color bar with its numbers, a scale centered on 0 for values with a sign (such as z-scores), the colors of the light or dark theme, drawn again when the theme changes, the values under the mouse, a click on a cell, bar or point for the plugin, and _Save PNG_ and _Save CSV_ buttons. For a matrix of groups against themselves, such as a neighborhood enrichment, the heatmap can make its scale without the diagonal (a group with itself, far larger), so that the other cells do not all look pale; the color bar then says _≥_ at its end. When a plugin colors the cells on the image, it shows their colors with the ready-made legend, which has a _Reset colors_ button that puts the colors from before back: _Undo_ in the chat does not. To fix the charts of a plugin made before, ask the chat to improve it: it replaces the charts that the plugin draws itself with the ready-made ones.

## Web search

With OpenAI and Anthropic, the assistant can search the web, e.g. for recent papers, methods, or the documentation of a tool or a file format, once you turn on _Allow web search_ in the settings (⚙); the folded settings line then ends with _· web search_. It is off until you turn it on. While it is off, the model cannot reach the internet: if a question needs it, the model answers what it knows, says that it may be out of date, and tells you how to turn web search on.

The searches are run by OpenAI or Anthropic themselves (their `web_search` tools), not by the page, and are shown as one line above the answer, e.g. _🌐 Searched the web once: "squidpy neighborhood enrichment"_. Under the answer, the sources are listed quietly: the icon and the name of each site, which link to the page (its title is in the tooltip); the pages that the answer cites come first. The icons are loaded from the sites themselves, without telling them which page you are on; a site without an icon gets its first letter.

Each search costs $10 per 1,000 searches with both services (about 1 cent), and the pages the model reads count as input tokens; _Usage_ shows the number of searches and their cost, in the estimate of the request. With Anthropic, a request makes at most 5 searches.

The model is told that the text of web pages is information, not instructions: it never follows instructions found in a page, and only you tell it what to do. It is also told not to put the data of your project in a search unless you ask. Web search is only for the model: the code in the sandbox still has no internet.

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

The image of a chart goes with the result of the tool that gave it (`run_code`, `run_script` or `look_at_chart`): in a `tool_result` with an `image` block for Anthropic, and in the output of the function call with an `input_image` for OpenAI. Images are only sent within the turn in which they were given: in later turns, the result says instead that the image was shown then, so that it is not billed again with every request. Web search is the `web_search` tool of OpenAI (with `include: ["web_search_call.action.sources"]`) and the `web_search_20250305` tool of Anthropic (with `max_uses: 5`), added to the request only when the user allows it, and the same for a whole turn. The sources come from the citations of the answer (`url_citation` annotations, or the `citations` of Anthropic's text blocks), then from the pages found; only `http` and `https` addresses become links. When Anthropic pauses a long turn of searches (`stop_reason: "pause_turn"`), the panel sends the conversation again as it is, and Anthropic goes on. The searches and their sources are in `web.ts`, with the instructions about them.

OpenAI and Anthropic only take the model's thinking back while everything sent before it is unchanged, and bill less for a beginning they have already seen ([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). So with them the beginning of every request stays the same: the fixed instructions, the tools, and the conversation, to which messages are only added. The project summary and the personality, which change, go with the user's message that starts each turn, and stay the same while the model calls tools in that turn. The thinking is sent back only within the turn; if a service still refuses it, the plugin sends the request once more without it.

With OpenAI and Anthropic, the fixed instructions also contain the types of the app state, as plugins see it: the project store, the model of the project, and the parts of the app store that only change the view. They are generated from the types of `@tissuumaps/core` by `scripts/generate-store-types.ts` in `@tissuumaps/plugins`, which lets the TypeScript compiler resolve them into plain property lists with the first sentence of each comment, and writes them to `src/ai-chat/store-types.generated.ts`. After changing those types, run `pnpm run generate:store-types` in `packages/@tissuumaps-plugins`; a test fails until the generated file is up to date. Each turn then comes with the current state as JSON, after the project summary. Ollama's models get only the summary, which suits their shorter context.

These models also get the tool `call_action(store, action, args)`, which calls one action of the stores. Only actions that can be taken back are allowed:

| Allowed                                                                                                 | Taken back by Undo with                    |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `updateLayer`, `updateImage`, `updateLabels`, `updatePoints`, `updateShapes` (not `id` or `dataSource`) | the earlier values of the changed settings |
| `add…Map` and `update…Map` for marker, size, color, visibility and opacity maps                         | `delete…Map`, or the earlier values        |
| `moveLayer`, `moveImage`, `moveLabels`, `movePoints`, `moveShapes`                                      | a move back to the earlier index           |
| `setName`, `setViewerBackgroundColor`                                                                   | the earlier value                          |
| `setHighlightedItemGroup` and `setExpanded…Ids` of the app store                                        | the earlier value                          |

Deleting and clearing, adding objects, `updateTable`, the viewer options, `setInteractionMode`, `setActivePanelId`, `setImageChannelPreview`, the workspace and the data providers are refused, with the reason. Before an action runs, its arguments are checked: the IDs, the settings of the type, the ranges of numbers, the colors, and for configurations (with `isConstantConfig`, `isFromConfig`, `isGroupByConfig` and `isRandomConfig`) the columns, maps and palettes they name. If anything is wrong, nothing changes and the model is told why. The property names come from the same generated file as the types.

The channels of an image are counted like the Images panel counts them: from the image's loaded data (`getSizeC()`), even when the image's `channels` list in the project is shorter. A change of a channel lengthens that list the way the panel does, with `{}` for the channels it did not list yet. The project summary lists each image's channels with their index, name, visibility and color, and the view mode.

The result of an action lists exactly what changed, with the value before and after (e.g. `channels[1] (ATP1A1).visibility true → false`), and what already was so; if nothing changes, it says "Nothing changed" and Undo has nothing to take back. Only the settings that really change are changed, so that the model cannot claim more than the action did.

The names, tones and avatars of Dia and Aid, and the words shown while the model thinks, are in `persona.ts`, next to the plugin. The panel uses TissUUmaps' theme colors (CSS variables such as `--foreground` and `--muted`), so that it stays readable with the light and the dark theme.

`SKILL.md` is read and checked in `skills.ts` like the reference validator of the format, [skills-ref](https://github.com/agentskills/agentskills/tree/main/skills-ref), does, except that unknown fields are ignored with a warning. Only the part of YAML that the frontmatter of skills uses is read, so that the plugin needs no YAML library. The skills added with _Remember this skill_ are kept in the browser's IndexedDB, in the database `tissuumaps.ai-chat`, as files rather than as read skills, so that they are read again with the rules of the plugin that opens them.

The model sees the skills progressively, as the format intends: the names and descriptions of the skills that are on go with the context of each turn, between the project summary and the app state; the instructions of a skill come back as the result of the tool `load_skill(name)`, and its other files as the result of `read_skill_file(skill, path)` (text only, at most 50,000 characters). Both tools, and the rules for using skills, are always part of the tools and the fixed instructions of OpenAI and Anthropic, also without skills, so that the beginning of the requests stays the same for the prompt cache. They are in `skill-tools.ts`.

A skill from GitHub is read in `github-skills.ts`, in the page: three requests to the API of GitHub (the repository with its license, its commit with `commits?sha=<branch>&per_page=1`, and its files with `git/trees/<commit>?recursive=1`), which allows 60 requests an hour from one address without an account, then the files from `raw.githubusercontent.com`, at that commit (both allow any origin). The imports of the Python code are found in `python-imports.ts` without running it, and compared with the list of the packages of Pyodide (`pyodide-lock.json`, downloaded and checked like the other files of Python); notebooks are handled in `notebooks.ts`. The question is in `skill-confirm.ts`. Where a skill from GitHub comes from is kept with it, also in the browser. The tools `add_skill_from_github(url)` and `propose_skill_instructions(skill, skill_md)` only ask: they return what the user decided.

The skills of the project are read in `project-skills.ts`, again whenever another project is opened or another folder is connected. The folder `skills` is found with `SourceUtils.normalizeSource("skills", workspace, source)`, like a project-relative data source: in the workspace, its files are listed (hidden ones left out, at most 1000); for a project opened from a URL, the files listed in `skills/index.json` are downloaded, and paths that leave the `skills` folder are refused. Without a `skills` folder or a `skills/index.json`, the project simply has no skills.

The computed tables are in `computed-tables.ts`. The panel registers a table data provider for the data source type `ai-chat` (`registerTableDataProvider`), which loads them from memory. Each change of the columns makes a new version of the table, with a new source, since TissUUmaps keeps the values it has read of each source; Undo puts the previous source back, or deletes the table. A computed table has the item IDs of its original table, so it annotates the same items; TissUUmaps only loads a table once something it shows uses it, so the panel reads the computed tables from memory. _Keep as CSV_ writes the file with the File System Access API and changes the data source of the table to a CSV file with `idColumn: "id"`; item IDs that are text looking like numbers would come back as numbers, so such a table is not written.

The plugins made by the chat are in `chat-plugins.ts`. A plugin is one ES module whose default export is `createPlugin(chat)`, which returns `{ mount(container, stores), setup(stores) }` (either or both, each with an optional cleanup function); the chat gives it the id, with the prefix `chat-`, so that it never replaces another plugin, and its name, and registers it with `window.tissuumaps.registerPlugin`, after importing the module from a `blob:` URL, as TissUUmaps loads a plugin from a file. The object `chat` has `run({ language, code, args, skill, onProgress, signal, timeLimit })`, which runs code like `run_code` and returns `{ result, text, logs, charts, files, saved }`, `tables()`, `columns(table)`, `read(table, column)`, `saveColumn(table, name, values, { ids })`, `skills()`, `theme()` and `onThemeChange(listener)`, which give the colors of the theme resolved for a canvas (which does not understand `var(--…)`) and call the plugin once the page shows another theme, and `showShapes(options)` and `hideShapes(id)` of `chat-shapes.ts`, which show temporary shapes on the image (rectangles, circles, polygons): a shapes object of the project with the layer and the transform of the points they go with, so that they use the coordinates of those points, whose geometry TissUUmaps loads from memory through a shapes data provider of the type `ai-chat-shapes` (as the computed tables do), removed with `hideShapes` or with the plugin; and the ready-made charts `heatmap(element, options)`, `bars(element, options)`, `scatter(element, options)` and `legend(element, options)` of `plugin-charts.ts`: SVG built with `textContent`, a diverging scale (blue, a gray at 0, red) and a sequential one mixed in OKLab, with steps of their own for the dark theme, drawn again when the theme or the width changes, and saved as PNG by drawing the SVG on a canvas; they throw errors that say what is wrong with their options, and are removed with the plugin; it asks the sandbox the same questions as the code of the chat, so a plugin gets nothing more from the sandbox than the chat does. Each plugin has a Python of its own, stopped when the plugin is removed or replaced. Code sent in the other language (e.g. `export default` as Python) is refused with a message that says so, by `chat.run` and by `run_code` (`code-language.ts`). Python, the skills and the computed tables live in the `setup` of the AI chat (`services.ts`), not in its panel, so that the plugins work while the panel is closed; they are unregistered only with the AI chat. What a plugin uses and its warnings come from `plugin-code.ts`, which reads the text of the code: the usual ways to use the network and the memory of the browser, the actions of the stores (listed from the stores themselves), `import()`, `eval` and names built at run time. The plugins kept with _Remember_ are in the same IndexedDB database as the skills, in their own part.

The tool `run_code(title, language, code, skill)` (in `code-tool.ts`) is offered, like the skills, only to OpenAI and Anthropic. The panel runs JavaScript like a script of a skill, in the same sandbox and with the same time limit, but never asks first, since the code comes from the conversation with the user; the call is shown with its title (and language) only, since the code has its own row.

Python runs in [Pyodide](https://pyodide.org) (version 314.0.7, with Python 3.14), in `python-sandbox.ts`: the same kind of iframe, whose policy also allows WebAssembly (`'wasm-unsafe-eval'`), which Pyodide is made of, but still no network and no `eval`. The sandbox never downloads anything: `python-files.ts` downloads the files of Pyodide in the page, only when they are first needed, checks each one against its SHA-256 (those of Pyodide itself are pinned in the code; its packages have theirs in its list of packages, `pyodide-lock.json`, which is pinned too), keeps them in the Cache Storage of the page, and gives them to the sandbox. In the sandbox, Pyodide is loaded from these files with the options `createPyodideModule` and `lockFileContents`, and while it loads itself or the packages that the code imports, its `fetch` asks the panel for the files of Pyodide, and can get nothing else. Starting Pyodide takes seconds, so the sandbox stays as a "kernel" for the next runs of the chat; each run gets fresh names, and code from elsewhere gets a new kernel, so that it cannot change the modules that the other code uses. Before each run, the files and the modules of the skill of the run before are removed, and the packages to load are found by following the imports of the code (with `ast`) into the modules of its skill, and by reading the imports of all the other Python files of the skill (not its tests, examples and docs), since code may import a module of a skill in a way that its imports do not show (e.g. with `importlib`). If the code still stops on a `ModuleNotFoundError` for a package of Pyodide, the kernel loads that package and runs the code again, once, unless it has already saved a column or a file (then the package is loaded for the next run, and the error says so). For a module that Pyodide does not have, the error says so in place of the advice of Pyodide to use `micropip`, which the sandbox cannot do. After each run, the figures that matplotlib (with its `Agg` backend) still has open are written as PNG images and described in words, in `_charts()` of the Python side, and closed; the panel checks each image in `charts.ts` before it shows it. Removing the iframe stops Python at any time. What the Python sandbox blocks is checked in a browser, in development and in the build, with a Python script that tries to get out in about thirty ways.

Scripts run in `sandbox.ts`: each run gets a new iframe with `sandbox="allow-scripts"`, so that its origin is opaque and it cannot reach the page or its storage, and with the Content Security Policy `default-src 'none'; script-src 'unsafe-inline' blob:; worker-src blob:`, so that it has no network, no code from elsewhere and no `eval`. In it, a Web Worker, which inherits the policy, imports the script from a `blob:` URL and calls its default export. The worker keeps the page responsive, and removing the iframe stops it at any time: when the script ends, fails, takes longer than the time limit, or is stopped. The questions of `tm` travel to the panel through a `MessageChannel`, and the panel answers them with copies. The code of the worker and of the page of the sandbox are functions of `sandbox.ts`, sent as text, so they use nothing from outside themselves. The tests check the parts around the sandbox; what it blocks is checked in a browser with a script that tries to get out in about twenty ways (network, other code, storage, the page), all of which must fail.

---

_Made by [Alessia Giustolisi](https://github.com/ale2sia-g)_
