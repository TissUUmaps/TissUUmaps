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
   summary are sent to the service you chose. Under each answer, a small line
   shows the tokens used (input, read from and written to the service's cache,
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
   While the model thinks, a word in Italian changes every few seconds: point
   at it to see what it means in English.

10. Dia and Aid
   With the light theme of TissUUmaps, the assistant is Dia: gentle and kind.
   With the dark theme, it is Aid: serious and a little mysterious. Only the
   tone changes, at once when you switch the theme: what the assistant can do,
   and its rules (such as telling the truth about what it changed), stay the same.

What the assistant can do: answer questions about the open project, show or hide
images, labels, points and shapes, make them more or less transparent, make
points bigger or smaller, color labels and points by a column of their table
(e.g. "color the cell labels by leiden", "cluster 3 in red, the others grey"),
give them one color (e.g. "color the cell circles green"),
and show only some groups (e.g. "show only clusters 1 and 3", then "show all the
cells again"). It cannot change or delete anything else.`;
