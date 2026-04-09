<div align="center">
<a href="https://www.producthunt.com/products/mindsdb?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-mindsdb-anton-2" target="_blank" rel="noopener noreferrer"><img alt="MindsDB Anton - Business intelligence that doesn't just answer — it acts. | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1115197&amp;theme=light&amp;t=1775607473112"></a>
</div>

<br>

```
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀          █▀█ █▀▀ █▀▀ ▄▀█ ▀█▀
    ▐   ▐            █▄█ ▄▄█ █▄▄ █▀█  █
    ▐   ▐
```

# OSCAT — OpenShore Custom Analytics Tool

Business intelligence was supposed to give you the right data, at the right time, to get real work done.

That is OSCAT. You ask questions in plain language, and OSCAT takes ownership of the entire analytical process:
it pulls and unifies data from multiple sources, runs the analysis, surfaces insights, builds rich dashboards, suggests next steps, and can even take action - A business intelligence agent that works like an expert analyst — 24/7, at machine speed.

![ezgif-24b9e7c74652f0dc](https://github.com/user-attachments/assets/c92f87c1-ff30-4272-92ba-49a8585d5954)


## Quick start 
**macOS - Desktop App:**

<a href="https://mindsdb-anton.s3.us-east-2.amazonaws.com/anton-latest-universal-signed.pkg">
<img width="64" alt="DesktopApp" src="https://github.com/user-attachments/assets/ed7c1e3a-3700-45cc-a9a8-efb57b43dcfd" />
</a>

 Click [here to download](https://mindsdb-anton.s3.us-east-2.amazonaws.com/anton-latest-universal-signed.pkg) the OSCAT Desktop App for MacOS.


**macOS / Linux - CLI:**
```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" 
```

**Windows CLI** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
```

That's it, you can now run it by simply typing the command:
```
(oscat)
```

## Using OSCAT

Talk to OSCAT like a person, for example, ask OSCAT this:

```
I hold 50 AAPL, 200 NVDA, and 10 AMZN. Get today's prices, calculate my
total portfolio value, show me the 30-day performance of each stock, and
any other information that might be useful. Give me a complete dashboard.
```

What happens next is the interesting part. At first, OSCAT doesn't have any particular skill related to this question. However, it figures it out live: scrapes live prices, writes code on the fly, crunches the numbers, and builds you a full dashboard — all in one conversation, with no setup.


```text
(oscat)> Dashboard is open in your browser.
Summary: Concentration risk is your #1 issue. If you're comfortable being a high-conviction NVDA...
```

<p align="center"> 
        <img width="800" alt="OSCAT's response" src="https://github.com/user-attachments/assets/6dc6ee81-2a2c-4358-be05-bfe884c32685" />
</p>

**Key features**
- **Credential vault** - prevents secrets from being exposed to LLMs.
- **Isolated code execution** - protected, reproducible “show your work” environment.
- **Multi-layer memory & continuous learning** - session, semantic and long-term business knowledge.

#### Connect your data
Although you can use OSCAT with just public data, the real power happens when you combine that with your own data. This can be anything: files,  databases, application APIs,... etc. Open the Local Vault with `/connect` command, then follow the prompts to add your secrets. OSCAT only has access to secret names - secret values remain hidden.

```powershell
/connect

(oscat) Choose a data source:

         Primary
           0. Custom datasource (connect anything via API, SQL, or MCP)

         Most popular
           1. Amazon Redshift
           2. Databricks
           3. Google BigQuery
           4. HubSpot
           5. MariaDB
           6. Microsoft SQL Server
           7. MySQL
           8. Oracle Database
           9. PostgreSQL
          10. Salesforce
          11. Shopify
          12. Snowflake

(oscat) Enter a number or type a name:

```

Tell OSCAT to connect and ask questions about your data. It will look for credentials in the vault (by their name), fetch the schema, and retrieve the necessary data. 
```test
YOU> Connect to STAPLECACHE company data. Check if there is a correlation between the discount given 
and the review rating in the last 6 months?

OSCAT>
⎿ Scratchpad (connecting and fetching schema…) 
   ~3s
```

---

### Explainable by default

You can always ask OSCAT to explain what it did. Ask it to dump its scratchpad and you get a full notebook-style breakdown: every cell of code it ran, the outputs, and errors — so you can follow its reasoning step by step.

---

## What's inside

<p align="center"><img width="800"  alt="image" src="/assets/oscat-diagram.png" /></p>

For the full architecture of OSCAT, file formats, and developer guide, see **[oscat/README.md](oscat/README.md)**.

---

## Workspace layout

When you run `oscat` in a directory:

- `.oscat/` — workspace folder containing scratchpad state, episodic memory, and local secrets.  
- `.oscat/oscat.md` — optional project context (OSCAT reads this at conversation start).  
- `.oscat/.env` — workspace configuration variables file (local file). 
- `.oscat/episodes/*` — episodic memories, one file per session.
- `.oscat/memory/rules.md` - behavioral rules: Always/never/when rules (e.g., never hardcode credentials, how to build HTML)     
- `.oscat/memory/lessons.md` - factual knowledge: Things I've learned (stock API quirks, dashboard patterns, data fetching notes)   
- `.oscat/memory/topics/*` - topic-specific lessons:  Deeper notes organized by subject (dashboard-visualization, stock-data-api, etc.) 
                                         
Override the working folder:
```bash
(oscat) --folder /path/to/workspace
```

---

## Memory systems

OSCAT provides two human-readable memory systems:

- **Semantic memory** — rules, lessons, identity and domain expertise stored as markdown at global and project scope.  
- **Episodic memory** — a timestamped archive of every conversation (JSONL in `.oscat/episodes/`). OSCAT can recall prior sessions with the `recall` tool.

Configure memory via `/setup` > Memory or via environment variables.

---

### Prerequisites

- `git` — required  
- Python **3.11+** (OSCAT will bootstrap an environment if missing)  
- `curl` — macOS / Linux installs  
- Internet connection (scratchpad may access web sources)

### Windows scratchpad firewall

The Windows installer can add a firewall rule so the scratchpad can reach the internet. If you skipped it, run in an elevated PowerShell:

```powershell
netsh advfirewall firewall add rule name="OSCAT Scratchpad" dir=out action=allow program="$env:USERPROFILE\.oscat\scratchpad-venv\Scripts\python.exe"
```

---

## How OSCAT differs from coding agents

OSCAT is a *doing* agent: code is a tool to get results. Where coding agents focus on producing code for a codebase, OSCAT focuses on delivering the outcome — a dataset, report, dashboard, or automated workflow — and will write whatever code is necessary to achieve that goal.

---

## What is OSCAT?

OSCAT is OpenShore's Custom Analytics Tool — an AI system built to collaborate with people to accomplish data and analytics tasks.

## Why the name "OSCAT"?

OSCAT stands for OpenShore Custom Analytics Tool — built to give your team the right data, analysis, and actions at machine speed.

---

## Analytics

OSCAT collects anonymous usage events (e.g. session started, first query) to help us understand how the product is used. No personal data or query content is sent.

To disable analytics, set the environment variable:

```bash
export OSCAT_ANALYTICS_ENABLED=false
```

Or add it to your workspace config (`.oscat/.env`):

```
OSCAT_ANALYTICS_ENABLED=false
```

---

## License

AGPL-3.0 license
