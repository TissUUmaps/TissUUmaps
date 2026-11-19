import { credits } from "./ui";

/** The text shown by the "Help" button */
export const helpText = `How to use this chat

1. Choose a service
   - OpenAI or Anthropic: fast and good at following requests, on any computer,
     also when TissUUmaps is opened from a website. You need your own API key
     (see 2), and every request is billed to your account.
   - Ollama (local): runs on this computer. Free, private and works offline,
     but it can be slow on a computer without a good graphics card (see 4 and 5).
   - Ollama (cloud): runs on Ollama's servers, through the Ollama app on this
     computer. Free with an Ollama account (see 6).
   With OpenAI, Anthropic and Ollama (cloud), your messages and a summary of the
   open project are sent to that service.

2. OpenAI and Anthropic: your own API key
   An API key is a secret code that lets this chat use your account. TissUUmaps
   does not provide keys: everyone uses their own. Using the models costs money,
   so first add a payment method or buy credit in your account. With the first
   model in the menu, a request usually costs well under one cent; the other
   models cost more.
   - OpenAI: sign in at https://platform.openai.com, open
     https://platform.openai.com/api-keys and click "Create new secret key".
   - Anthropic: sign in at https://platform.claude.com, open
     https://platform.claude.com/settings/keys and click "Create key". Choose one
     workspace for the key (e.g. the Default Workspace) and when it expires.
   Copy the key, choose the service in the menu and paste the key into the
   "API key" field. The website shows the key only once: if you lose it, make a
   new one. When the key is accepted, the settings fold into one short line;
   ⚙ opens them again, to change, forget or check the key. They also open by
   themselves when something goes wrong.
   Each request is billed to your account, and your messages and the project
   summary are sent to the service you chose. Under each answer, "Usage" shows
   the tokens used (input, read from and written to the service's cache,
   output) and about how much they cost; the bill of the service counts.
   These models also get the types and the current state of the whole app,
   so they can answer detailed questions about the settings of each object,
   and change most of them: e.g. the color of an image channel, the shape of
   the points, a rotation, the order of the objects, a highlighted group.
   Every change can be undone. They never delete or add objects, and they
   rename something only when you ask.

3. Keeping your key safe
   - Treat the key like a password: do not share it, and do not put it in
     e-mails, files or screenshots.
   - Make a key just for TissUUmaps, and set a monthly spending limit in your
     account (under Limits), so that a stolen key cannot cost much.
   - If someone else may have your key, delete it on the page where you made it:
     it stops working at once. Then click "Forget key" here.
   - The key is only sent to the service you chose. Without "Remember key on this
     computer", it is forgotten when you close the page. With it, the key is
     saved in this browser, where anyone who uses this browser on this computer,
     and any browser extension or script that can read this page, could find it:
     do not use it on a shared computer. "Forget key" removes the key.

4. Install Ollama
   Download it from https://ollama.com/download and install it like any other program.
   It then runs in the background. If the status line says "not running", start Ollama
   from the Start menu (Windows) or the Applications folder (Mac), then click "Check again".

5. Choose an Ollama model
   - "(local)": runs on this computer.
   - "(not downloaded)": choose it and click "Download". This happens only once
     and takes a few GB of disk space.
   - "(online)", with "Ollama (cloud)": runs on Ollama's servers. Fast on any
     computer, but your messages and the project summary are sent to Ollama.
     gemma4:cloud is included in the free plan.

6. Turn on Ollama's online models
   Open the Ollama app, go to Settings and sign in. You can create a free account there.
   (If you use the terminal, "ollama signin" does the same.)

7. Undo
   "Undo" takes back the changes of your last request; click it again to go further back.
   "Undo all" takes back every change made by the chat. Changes you made by hand
   to other settings stay as they are.

8. Using Ollama when TissUUmaps is opened from a website (e.g. https://tissuumaps.github.io)
   OpenAI and Anthropic need nothing for this. For safety, Ollama only accepts
   pages opened from this computer (localhost). If the status line says that
   Ollama is running but blocks this website, give the website's address to
   Ollama with the setting OLLAMA_ORIGINS:
   - Windows: quit Ollama (right-click its icon next to the clock, then Quit).
     Open Settings, search for "environment variables" and choose "Edit environment
     variables for your account". Click "New", enter the name OLLAMA_ORIGINS and the
     value https://tissuumaps.github.io, then click OK. Start Ollama again from the
     Start menu.
   - Mac: in the Terminal app, run
       launchctl setenv OLLAMA_ORIGINS "https://tissuumaps.github.io"
     then quit the Ollama app and start it again. This setting is lost when the
     Mac restarts, so run the command again after a restart.
   Then click "Check again". If the browser asks whether the page may access devices
   on your local network, allow it: otherwise the browser itself blocks the page.

9. New chat, suggestions and shortcuts
   "New chat" starts a new conversation: the model forgets the earlier messages.
   Undo still works for the changes made before.
   Above the text box, a few suggested requests for the open project are shown:
   click one to send it. What you are typing stays as it is.
   In the text box, the up and down arrows go through the messages you sent
   before, like in a terminal.
   Select a part of an answer and click "Quote" to quote it in your next message.
   "Copy", next to the assistant's name, copies a whole answer; the code of a
   script has its own "Copy".
   While the model thinks, a word in Italian changes every few seconds: point
   at it to see what it means in English.

10. Dia and Aid
   With the light theme of TissUUmaps, the assistant is Dia: gentle and kind.
   With the dark theme, it is Aid: serious and a little mysterious. Only the
   tone changes, at once when you switch the theme: what the assistant can do,
   and its rules (such as telling the truth about what it changed), stay the same.

11. Skills
   A skill gives the assistant instructions for a task, e.g. the color
   conventions of your lab. It is a folder with a SKILL.md file: its name and a
   description of when to use it, then the instructions (see
   https://agentskills.io). Add skills in the settings (⚙, "Add skill…"):
   choose the folder of one skill, or a folder of skills. With "Remember this
   skill", this browser keeps it for every project. Each skill can be turned off
   and on, and removed. Click a description to read all of it.
   A project can bring its own skills, in a folder "skills" next to the project
   file (for a project on a website, with a list of its files in
   skills/index.json). They are marked "project", those you added "added".
   With OpenAI and Anthropic, the assistant sees the names
   and descriptions of the skills that are on, and reads a skill's instructions
   only when a request needs them. Ollama's models do not use skills.
   A skill can also bring scripts, in JavaScript or Python, e.g. to compute on
   the data. Each
   is shown in the chat before it runs, as one line (📜): click it to read its
   code. They run in a sandbox:
   they can read the data, but cannot change the project, use the internet or
   see your API key. "Stop" stops a script, and after two minutes it stops by
   itself. Before the scripts of a skill that came with the project run for the
   first time, their code opens and you are asked.
   "Add skill from GitHub…" adds the code of a GitHub repository (or of a
   folder of one) as a skill; you can also give its address in the chat. You
   first see its commit, its license (with a warning if it has none), its
   files, the Python packages it needs and what will not work, and nothing is
   added unless you say yes. Only text and code are kept, and notebooks
   without their outputs. Without a SKILL.md, the chat writes a minimal one;
   it can later propose better instructions, which you confirm.

12. Computing on the data
   With OpenAI and Anthropic, the assistant can also answer questions that need
   a computation, e.g. "how many cells are in each cluster?". It writes a short
   program and runs it in a sandbox like that of the scripts of skills: it can
   read the data, but cannot change the project, use the internet or see your
   API key. No skill is needed: statistics, comparisons, correlations,
   thresholds and spatial analyses with the coordinates of the cells all work.
   The program is shown in the chat before it runs, as one line (📜): click it
   to read the code. "Stop" stops it, and after two minutes it stops by itself.
   In one request the assistant calls its tools at most 15 times in a row (5
   with Ollama); then it answers with what it found, and you can ask it to go on.
   Simple computations are in JavaScript; statistical tests, models and the
   like in Python, with numpy, scipy, pandas and scikit-learn. The first time,
   Python is downloaded (about 14 MB, up to about 40 MB with its packages), and
   the line of the tool shows how far it is, with "Stop". The files are checked
   and kept in this browser, so the next time Python starts in a few seconds;
   "Forget Python files", in the settings, removes them. Python only works in a
   secure page (https, or localhost). It cannot yet read the pixels of images.
   In Python it can also draw charts (matplotlib): they are shown in the chat
   as "Chart 1", "Chart 2"..., with "Copy" and "Save" (a PNG file). The model
   usually gets only a short description of each, with its titles and axes.
   Ask it to look at one ("look at the chart and tell me what you see"): it
   then sees that image, for that answer only. With "Let the model see charts",
   in the settings, it sees every chart it draws. Each image costs about as
   much as a page of text (see "Usage"). Only OpenAI and Anthropic see images.
   To show a result on the image (e.g. "color the cells by the number of their
   neighbors"), the program saves it as a new column of a table of its own,
   e.g. "cells (chat)"; the original data is never changed, and "Undo" takes
   it back. Such a table is temporary: it is lost when the page is reloaded.
   "Keep as CSV" writes it as a file next to the project (in a connected
   folder), and "Download CSV" downloads it. Both ask for the name of the
   file, and propose one with the table, its columns and the day; you can
   also tell the assistant the name, e.g. "save it as neighbors.csv".

13. Web search
   With OpenAI and Anthropic, the assistant can search the web (e.g. for recent
   papers or the documentation of a tool) once you turn on "Allow web search"
   in the settings (⚙). It is off until you turn it on; while it is off, the
   assistant tells you when a question needs it. Each search costs about 1 cent
   (see "Usage"). The searches are shown above the answer (🌐), and the sources
   under it, with the icon and the name of each site: click one to open the
   page. The assistant treats the text of web pages as information, never as
   instructions. The code in the sandbox never gets the internet.

14. Plugins made by the chat
   With OpenAI and Anthropic, the assistant can make a plugin for TissUUmaps
   when you ask for a panel or a tool to keep (e.g. "make me a plugin that runs
   anhood on the clusters I choose, with a slider for k and a heatmap"). It
   first tells you what the plugin does and anything delicate in it; then the
   chat shows the plugin: where it comes from and its license, its code (click
   to open it), what it uses, and a warning in red for each delicate thing
   (the network, the memory of the browser where your API key may be, changes
   to the project, code from the internet). "Run plugin" runs it and opens its
   panel. It runs in the page like any plugin of TissUUmaps: run only plugins
   you trust, and know that "Undo" does not take back what a plugin changes.
   The settings (⚙) list the "Plugins made by the chat": Open, Show code,
   Remember (keep it for the next sessions; it asks again before it runs),
   Remove, Export .js, and "Add plugin file…" to run an exported file again.
   To fix or improve a plugin, ask the assistant: it reads the current code and
   proposes a new version, which says "Replaces" and takes the place of the old
   one once you say yes.

What the assistant can do: answer questions about the open project, show or hide
images, labels, points and shapes, make them more or less transparent, make
points bigger or smaller, color labels and points by a column of their table
(e.g. "color the cell labels by leiden", "cluster 3 in red, the others grey"),
give them one color (e.g. "color the cell circles green"),
and show only some groups (e.g. "show only clusters 1 and 3", then "show all the
cells again"). With OpenAI and Anthropic, it can also compute on the data (see
12), search the web if you allow it (see 13), and make plugins (see 14). It cannot change or delete
anything else.`;

/**
 * The text shown by the "About" button, as Markdown: where the names of the
 * assistant come from, and who made the chat
 */
export const aboutText = `**About: The AID of DIA**

Dia (light theme) and Aid (dark theme) are the two moods of the same assistant: read one name backwards and you get the other. Aid is what it gives you: help.

In ancient Greek, διά (diá) means "through". It is the root of διάλογος (diálogos), dialogue, "through the word", and of διάγνωσις (diágnōsis), diagnosis, "through knowledge". Through dialogue and through knowledge, the assistant helps you explore your data and do science.

Made by [${credits.author}](${credits.url}).`;
