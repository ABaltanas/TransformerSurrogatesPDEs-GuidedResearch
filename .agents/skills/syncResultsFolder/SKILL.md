---
name: syncResultsFolder
description: Synchronize simulation and experiment results (or specific subfolders inside results/) between the Laptop (fedora) and Tower workstation (TorreArturo). Automatically detects whether it is executed on the Laptop or Tower, prompts the user to select the target subfolder, constructs the correct rsync command, and always requests user approval before execution.
---

# Sync Results Folder

This skill synchronizes PDE surrogate modeling and experiment results between the local workstation and remote workstation (Laptop ↔ Tower). It detects the current execution environment (Laptop vs Tower) and constructs the appropriate `rsync` command, while prompting the user to choose which folder inside `results/` to synchronize.

## Configuration & Environment Details

| Property | Laptop | Tower (TorreArturo) |
| :--- | :--- | :--- |
| **Hostname / User** | `fedora` / `arturobaltanas` | `TorreArturo` / `artur` (Win), `arturo` (WSL) |
| **Tailscale IP** | `100.118.27.38` | `100.64.212.13` |
| **Repo Path** | `/home/arturobaltanas/University/GuidedResearch` | `/home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch` |
| **Results Path** | `/home/arturobaltanas/University/GuidedResearch/results/` | `/home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/results/` |
| **SSH Target** | `arturobaltanas@100.118.27.38` | `artur@100.64.212.13` (runs with `--rsync-path="wsl rsync"`) |

---

## Workflow Instructions

### 1. Environment Detection
Determine whether the command is running on the **Laptop** or the **Tower**:
- Run `uname -n` or `whoami`:
  - **Laptop**: `uname -n` contains `fedora` (or `whoami` is `arturobaltanas`).
  - **Tower**: `uname -n` contains `TorreArturo` / `torre` (or `whoami` is `arturo`).

Determine the synchronization direction:
- **When running on Laptop**:
  - Default: **Pull from Tower to Laptop** (Tower → Laptop)
  - If user requested push: **Push from Laptop to Tower** (Laptop → Tower)
- **When running on Tower**:
  - Default: **Push from Tower to Laptop** (Tower → Laptop)
  - If user requested pull: **Pull from Laptop to Tower** (Laptop → Tower)

### 2. Target Folder Selection
- Inspect the local `results/` directory to discover existing folders (e.g., `DatasetAnalysis`, `TrainingAnalysis`).
- If the user did not specify a subfolder in their initial prompt:
  - Ask the user which folder inside `results/` they want to synchronize.
  - Use the `ask_question` tool with the detected subdirectories as options, an option to sync the entire `results/` directory, and allowing a custom write-in folder name.
  - Wait for the user's choice before proceeding.
- If the user specified a subfolder (e.g., `DatasetAnalysis` or `TrainingAnalysis`), use that subfolder.
- If the user specifies "all", "everything", or root `results/`, target the entire `results/` directory.

### 3. Verification and Directory Preparation
- Verify and prepare the target destination directory:
  - If syncing to the local machine:
    - For a subfolder `<subfolder>`: run `mkdir -p results/<subfolder>/` if it does not already exist.
    - For the entire directory: ensure `results/` exists.
  - If syncing to the remote machine: ensure the local source directory exists before syncing.

### 4. Mandatory Approval Step (CRITICAL)
> [!IMPORTANT]
> **Do NOT execute the rsync command without prior user confirmation.**
> You must explicitly present the detected environment, transfer direction, and the exact rsync command to the user and receive approval before proceeding.

Present the synchronization details based on the detected host:

#### A. Running on Laptop (`fedora`)
- **Default (Pull from Tower → Laptop)**:
  - Subfolder:
    ```bash
    rsync -avzP --rsync-path="wsl rsync" artur@100.64.212.13:/home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/results/<subfolder>/ /home/arturobaltanas/University/GuidedResearch/results/<subfolder>/
    ```
  - Entire directory:
    ```bash
    rsync -avzP --rsync-path="wsl rsync" artur@100.64.212.13:/home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/results/ /home/arturobaltanas/University/GuidedResearch/results/
    ```
- **Push to Tower (Laptop → Tower)**:
  - Subfolder:
    ```bash
    rsync -avzP --rsync-path="wsl rsync" /home/arturobaltanas/University/GuidedResearch/results/<subfolder>/ artur@100.64.212.13:/home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/results/<subfolder>/
    ```
  - Entire directory:
    ```bash
    rsync -avzP --rsync-path="wsl rsync" /home/arturobaltanas/University/GuidedResearch/results/ artur@100.64.212.13:/home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/results/
    ```

#### B. Running on Tower (`TorreArturo`)
- **Default (Push from Tower → Laptop)**:
  - Subfolder:
    ```bash
    rsync -avzP /home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/results/<subfolder>/ arturobaltanas@100.118.27.38:/home/arturobaltanas/University/GuidedResearch/results/<subfolder>/
    ```
  - Entire directory:
    ```bash
    rsync -avzP /home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/results/ arturobaltanas@100.118.27.38:/home/arturobaltanas/University/GuidedResearch/results/
    ```
- **Pull from Laptop (Laptop → Tower)**:
  - Subfolder:
    ```bash
    rsync -avzP arturobaltanas@100.118.27.38:/home/arturobaltanas/University/GuidedResearch/results/<subfolder>/ /home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/results/<subfolder>/
    ```
  - Entire directory:
    ```bash
    rsync -avzP arturobaltanas@100.118.27.38:/home/arturobaltanas/University/GuidedResearch/results/ /home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/results/
    ```

Ask the user:
> *"Do you approve running the rsync command? (You can also request a `--dry-run` first)."*

Wait for the user's explicit response. If the user rejects or requests adjustments, do not run the command.

### 5. Execution
Once approval is granted, execute the approved rsync command.

### 6. Post-Execution Summary
- Summarize the transferred files and transfer status.
- In case of failure:
  - If SSH connection timed out: check if Tailscale is running on both machines (Laptop: `100.118.27.38`, Tower: `100.64.212.13`).
  - If running from Laptop to Tower and rsync fails: check WSL status on the remote Windows host (`--rsync-path="wsl rsync"`).
  - If running from Tower to Laptop: ensure SSH server is running on Laptop and key authentication succeeds.
