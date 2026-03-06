# Yield curve dislocation (Polymarket)

Calendar-spread strategy: static dislocation signal, backtest + live execution.

## Essential files

| File | Purpose |
|------|--------|
| `curve_pipeline.py` | Core algorithm: universe, panel, signal, hedge weights, trade builder. |
| `live_execution.py` | Fast live runner (signals + optional order placement). No backtest rerun. |
| `curve_dislocation_backtest.ipynb` | Backtest: config, universe, panel, trades, dedup, summary. |
| `analytics.ipynb` | Analytics, diagnostics, calibration. |
| `ALGORITHM.md` | Full algorithm and runbook. |
| `requirements.txt` | Python deps. |
| `.env.example` | Env template; copy to `.env` and fill for live trading. |

## Run

- **Backtest:** Open `curve_dislocation_backtest.ipynb`, run all cells.
- **Analytics:** Open `analytics.ipynb`, run as needed.
- **Live (one shot):** `python3 live_execution.py`
- **Live (loop):** `python3 live_execution.py --execute-live --loop-seconds 300`

Generated at runtime (gitignored): `.cache/`, `logs/` (cycle_log.jsonl, execution_log.jsonl, execution_attempts_latest.csv).

---

## Putting your bot on the cloud (simple version)

**In one sentence:** You rent a small computer on Amazon’s servers, copy your code and keys onto it, and start the bot there so it runs 24/7 even when your laptop is off.

**What we’re doing:** Right now the script runs on your laptop. Close the laptop and it stops. To have it run **all the time**, we put it on a **computer that Amazon runs for you**. You pay by the hour (the smallest option is only a few cents per hour). That computer stays on 24/7, so your bot can keep running.

**What you need:** An Amazon (AWS) account. Sign up at aws.amazon.com if you don’t have one.

**Words we use (so nothing is mysterious):**
- **Cloud / server** = a computer that Amazon runs; you use it over the internet.
- **Log in (to the server)** = connect from your Mac to that computer so you can type commands on it.
- **Terminal** = the app on your Mac where you type text commands (find it in Applications → Utilities, or search “Terminal”).
- **Run a command** = type a line in Terminal and press Enter.

---

### Step 1: Create your computer in the cloud

1. Log into AWS. In the search bar at the top, type **EC2** and open it. (EC2 is just “rent a computer from Amazon.”)
2. Click the orange **“Launch instance”** button. (“Instance” = one rented computer.)
3. Fill in the page:
   - **Name:** Any name, e.g. “trading-bot”.
   - **Operating system:** Keep the default (Amazon Linux) or choose **Ubuntu**.
   - **Computer size:** Choose the **smallest** option (e.g. “t3.micro”). It’s cheap and enough for this bot.
   - **Key pair:** Click “Create new key pair”, name it, then **download the file** and keep it somewhere safe (e.g. Downloads). You’ll need this file to log in — it’s like a key to your cloud computer.
   - **Network / firewall:** Leave defaults or make sure “SSH” is allowed (so you can log in from your Mac).
4. Click **“Launch instance”**. Wait until the status says “running”. On the same page you’ll see a **public IP address** (numbers like `54.123.45.67`). Copy it — you need it for the next step.

---

### Step 2: Log into that computer from your laptop

Open **Terminal** on your Mac. You’ll “log in” to the computer in the cloud using the key file you downloaded.

- If you chose **Amazon Linux**, run (replace the path and the IP with yours):
  ```bash
  ssh -i /path/to/your-downloaded-key.pem ec2-user@YOUR_PUBLIC_IP
  ```
- If you chose **Ubuntu**, run:
  ```bash
  ssh -i /path/to/your-downloaded-key.pem ubuntu@YOUR_PUBLIC_IP
  ```

Example: if your key is in Downloads and named `my-key.pem`, and the IP is `54.123.45.67`:
```bash
ssh -i ~/Downloads/my-key.pem ec2-user@54.123.45.67
```

The first time it might ask “Are you sure you want to connect?” — type `yes` and press Enter. You’re now “inside” the cloud computer; the prompt will change.

---

### Step 3: Install your project on that computer

You’re still “inside” the cloud computer (your prompt looks like `[ec2-user@ip-...]$`). Run these four lines, one after the other. They **download your code from GitHub** and **install everything the bot needs** (Python and libraries):

```bash
git clone https://github.com/AnranSeverac/polymarketTermStructure.git
cd polymarketTermStructure
chmod +x scripts/aws-setup.sh
./scripts/aws-setup.sh
```

Wait until it finishes. It’s just setting up Python and your project on this computer.

---

### Step 4: Put your API keys on the server

The bot needs your Polymarket keys. On the same cloud computer, run:

```bash
cp .env.example .env
nano .env
```

In the editor that opens, paste your real **API key**, **private key**, and **funder address** (same as in your local `.env`). Save and exit: press **Ctrl+O**, Enter, then **Ctrl+X**.

---

### Step 5: Start the bot and leave it running

First line = “use the Python we set up.” Second line = “start the bot in the background and send its output to a log file.” Run:

```bash
source .venv/bin/activate
nohup python3 live_execution.py --execute-live --loop-seconds 300 > live.log 2>&1 &
```

That starts the bot in the **background** so it keeps running even after you close the connection. To watch what it’s doing:

```bash
tail -f live.log
```

Press **Ctrl+C** to stop watching (the bot keeps running). Type `exit` and Enter to log out of the server. The bot will keep running on AWS.

---

### Updating the code later

If you change the code on GitHub and want the server to use the new version: log in again (Step 2), then run:

```bash
cd polymarketTermStructure
git pull
```

Then restart the bot (stop the old one if needed, and run the same command from Step 5 again).
