# Craft your own agent — a guided walk-through

This is the friendliest way to go from *"I want an agent that talks and behaves
the way I want"* to **making one that does**. It assumes nothing: no prior
knowledge of this system, and no need to read Python (you'll copy-paste a couple
of tiny files as-is).

If you *do* want to read the code underneath, or want the full reference, see
[`docs/manual.md`](manual.md) — the links at the end of each step go there.

---

## The big picture first (60 seconds)

A few terms, in plain words:

- **Agent** — a named bundle of behavior: *who it is* (its voice), *what it may*
  *do* (its tools), and *any small rules* it always follows. Each agent lives in
  its own **folder**. The default one — the "main agent" — is the nameless root
  folder `hook_modules/`.
- **Tool** — something the agent can *do* on your behalf, like a calculator.
  Tools are **off by default**; you switch each one on if you want it.
- **Hook** — a named "little rule" the agent always follows (like "always sign
  your name"). We'll meet exactly one, and only copy-paste it.

**The one mental model to keep:** making an agent means *making a folder and
dropping a few short files into it*. There is nothing to register, compile, or
configure. Deleting the folder deletes the agent. That's the whole system, from
your side.

> We'll build one agent together, named **`tutor`**, that explains things like a
> patient math teacher. Replace the words where you want your own personality.

---

## Step 1 — Give it a voice (a persona)

This is the part everyone needs. It's a plain-English file that says *who this
agent is*.

1. Make the agent's folder:

   ```bash
   mkdir hook_modules/tutor
   ```

2. Inside it, create a file named `prompt.md` and write, in plain English, who
   the agent should be. Copy this, then edit it into your own voice:

   ```markdown
   You are a patient, encouraging math tutor.
   You explain each step slowly, the way you would to a curious student, and you
   never rush to the final number. When something is unclear, you ask one gentle
   question before answering. You are kind, and you celebrate small wins.
   ```

   That's it for a voice. The agent now *is* your text — no code involved.

3. **See it work.** Open the REPL (see [Running below](#running)); you should
   already have the backend running (see below). Then:

   ```
   /agent tutor
   you> I'm stuck on why a negative times a negative is a positive.
   ```

   You should feel the tone shift toward your tutor's voice. The agent was made
   by the appearance of its folder + `prompt.md`; nothing else.

   > How does this work? An agent folder is "replacing" the default agent for the
   > time that agent is selected, and its `prompt.md` becomes the instructions it
   > starts every conversation with. Details: [manual — "The four channels"]
   > (/manual.md#the-four-channels).

---

## Step 2 — Turn on a tool (optional)

Tools are **off unless you list them**. To let your tutor actually calculate,
add one small file so it may use the built-in `calculator`.

1. In the same `hook_modules/tutor/` folder, create a file named `tools.py` with
   this content (copy-paste as-is; you don't need to read it):

   ```python
   from agent_seed.hooks import Hooks, RunContext

   # This is the only line that matters to you: the list of tool names.
   ALLOWED_TOOLS: list[str] = ["calculator"]


   def enter_agent(ctx: RunContext) -> None:
       ctx.allowed_tools = list(ALLOWED_TOOLS)


   def make() -> Hooks:
       return Hooks(enter_agent=enter_agent)
   ```

   The only line you'll ever change is `ALLOWED_TOOLS`. Add a name to allow more;
   remove a name to take one back. `["calculator"]` is enough for math.

2. **Restart the backend** (hooks load when the server starts), then in the REPL:

   ```
   /agent tutor
   you> what is 17 times 17?
   agent> ... (your tutor's patient voice) ... 17 × 17 = 289.
   ```

   It now *used* the tool to get the number, inside its own voice. You can confirm
   with `/tools` — `calculator` should be listed.

   > Why does a tool need a whole file? "Allowing a tool" is a rule that runs the
   > moment the agent takes the stage; this system's rule mechanism is the "hook".
   > Details: [manual — "Per-agent tool access (strict whitelist)"]
   > (/manual.md#per-agent-tool-access-strict-whitelist).

---

## Step 3 — Add one always-on rule (a "hook")

Want a small, *guaranteed* behavior — like always ending with a little sign-off?
That's a hook. It's just a few lines that run at the end of every answer.

1. In `hook_modules/tutor/`, create a file named `sign.py` (copy-paste as-is):

   ```python
   from agent_seed.hooks import Hooks, RunContext


   def finish(ctx: RunContext) -> None:
       # Add a kind sign-off to the end of every answer.
       if ctx.final_answer is not None:
           ctx.final_answer = f"{ctx.final_answer}\n\n— your tutor"


   def make() -> Hooks:
       return Hooks(finish=finish)
   ```

   Change the `— your tutor 💡` line to whatever sign-off you like.

2. **Restart the backend**, then in the REPL:

   ```
   /agent tutor
   you> quick sanity check: 2 + 2?
   agent> 2 + 2 = 4.
           — your tutor
   ```

   Your rule is now *always* there, no matter what else it says. To undo it,
   delete `sign.py`.

   > A "hook" is a fixed point the system calls at a moment (before it answers,
   > here) where you may tuck in a little bit of behavior. Details: [manual —
   > "The full list of hooks (reference)"](/manual.md#the-full-list-of-hooks-reference).

---

## Step 4 — Your full agent, end to end

You now have a folder that is a complete, tailor-made agent:

```
hook_modules/tutor/
    prompt.md    who it is          <- your words (Step 1)
    tools.py     what it may do     <- the calculator (Step 2, optional)
    sign.py      its one rule       <- the sign-off (Step 3, optional)
```

Drive it in the REPL:

```
/agents             <- tells you it exists:  {"agents": ["tutor"]}
/agent tutor        <- switch to it
you> teach me long division on 247 ÷ 5
                      ... (patient voice, uses the calculator, signs off) ...
/quit
```

To **use the default agent again**, type `/agent` (with no name) — or start a
fresh session. Nothing about the main agent changed: your `tutor` is purely
*additive*; it only takes the stage when you select it.

---

## Step 5 — Undo / take it back

To remove the agent entirely:

```bash
rm -rf hook_modules/tutor
```

Restart the backend. It's as if it was never there. No cleanup anywhere else.
(If it was only a voice + one rule + one tool, you can also just delete the
files you like inside the folder and keep the folder.)

---

## Running (the two commands that matter)

You need two terminals while developing.

**Terminal 1 — the backend** (the "brain" server):

```bash
uv sync             # first time only: sets up the project
cp .env.example .env  # first time only: then put your LLM settings in .env
uv run agent-seed   # starts the backend at http://127.0.0.1:8787
```

**Terminal 2 — the REPL** (where you talk to the agent):

```bash
uv run agent-repl
```

Every time you *edit one of your agent's files* (Steps 1–3), **restart the
backend in Terminal 1** (`Ctrl-C` it, then `uv run agent-seed` again) so the
changes load. The REPL in Terminal 2 stays open and just reconnects to the
backend.

---

## Going further

You've used the three levers that cover most personal agents: **voice, tools,
rules.** When you want more, each has a one-line entry point:

- **More behavior rules** — any of the named "hooks" (`before_llm`,
  `after_tool_call`, a budget meter, …). The full list, in plain terms:
  [manual — "Which hook to use, by goal"](/manual.md#which-hook-to-use-by-goal).
- **More prompt nuance** — per-round context, few-shot examples.
  [manual — "Prompting — the content layer"](/manual.md#prompting--the-content-layer).
- **Writing your own tool** (a new capability) in Python.
  [manual — "Adding tools"](/manual.md#adding-tools).
- **Choosing which model it uses.** [manual — "Model selection"]
  (/manual.md#model-selection-dynamic-sticky).
- **Several agents at once.** Each is just another folder under `hook_modules/`;
  switch between them with `/agent`. [manual — "Agents: hook subdirectories"]
  (/manual.md#agents-hook-subdirectories).

The full user manual is [`docs/manual.md`](manual.md).
