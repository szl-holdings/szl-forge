# Owned Agent Control v2 Runbook

This runbook qualifies and operates one local Windows OAC installation. Its
examples require PowerShell 7 or newer (`pwsh`). Run it as the Windows identity
that will own the controller state. Do not share that identity with untrusted
code.

## 1. Build and qualify the runtime

```powershell
git clone https://github.com/szl-holdings/szl-forge.git
Set-Location .\szl-forge
py -3.12 -m pip install build==1.5.0 setuptools==83.0.0 wheel==0.47.0
py -3.12 -m build --wheel .\agent-forge
$Wheel = (Resolve-Path .\agent-forge\dist\szl_owned_agent_control-2.0.0-py3-none-any.whl).Path
Get-FileHash -Algorithm SHA256 $Wheel
py -3.12 -m venv .venv-oac
.\.venv-oac\Scripts\python.exe -m pip install --upgrade pip
.\.venv-oac\Scripts\python.exe -m pip install --requirement .\agent-forge\requirements.lock
.\.venv-oac\Scripts\python.exe -m pip install --no-deps $Wheel
.\.venv-oac\Scripts\owned-agent-control.exe doctor
.\.venv-oac\Scripts\owned-agent-control.exe self-test
```

Stop if `doctor` is not ready or `self-test` does not return
`VERIFIED_LOCAL_WINDOWS_ENFORCEMENT`. Linux results verify portable contracts
only. In production, use the approved CI wheel and compare its SHA-256 with the
reviewed release value before installation.

## 2. Initialize owner-only state

Use an NTFS directory on a fixed local drive. OAC rejects reparse points and
applies an owner-only ACL.

```powershell
$Oac = '.\.venv-oac\Scripts\owned-agent-control.exe'
$State = 'C:\ProgramData\SZL\OwnedAgentControl'
& $Oac init --state-dir $State
```

Back up operator private keys separately. State is locally tamper-evident, not
immutable; protect and back it up with normal Windows controls.

## 3. Establish the operator quorum

Use distinct passphrases and preferably separate operator custody. The example
uses environment variables to avoid command-line passphrases.

```powershell
$env:OAC_ALICE_PASSPHRASE = Read-Host 'Alice key passphrase' -AsSecureString |
  ConvertFrom-SecureString -AsPlainText
$env:OAC_BOB_PASSPHRASE = Read-Host 'Bob key passphrase' -AsSecureString |
  ConvertFrom-SecureString -AsPlainText

& $Oac keygen --operator alice --private-key .\alice.pem --public-key .\alice.pub `
  --passphrase-env OAC_ALICE_PASSPHRASE
& $Oac keygen --operator bob --private-key .\bob.pem --public-key .\bob.pub `
  --passphrase-env OAC_BOB_PASSPHRASE
& $Oac operator-add --state-dir $State --operator alice --public-key .\alice.pub
& $Oac operator-add --state-dir $State --operator bob --public-key .\bob.pub
& $Oac trust-seal --state-dir $State
```

Unset passphrase variables immediately after signing operations.

## 4. Register the target

Registration pins the resolved executable and its SHA-256. Do not replace the
binary in place: register a new target ID for a new executable hash.

```powershell
& $Oac register --state-dir $State --target owned-agent:worker-v1 `
  --cwd 'C:\SZL\worker' -- 'C:\Python312\python.exe' '-m' 'worker'
```

For qualification only, register the harmless built-in fixture:

```powershell
& $Oac register-demo --state-dir $State --target owned-agent:demo
```

## 5. Generate the context gate

Review and edit `agent-forge\examples\context-input.json` for the exact target.
The input must contain at least one cross-step comparison. Empty comparison
sets fail closed.

```powershell
& $Oac context-generate --state-dir $State --target owned-agent:worker-v1 `
  --input .\agent-forge\examples\context-input.json
& $Oac context-show --state-dir $State --target owned-agent:worker-v1
```

Starting requires a latest trace whose consistency, legal transition, and
policy gates are stabilized. Context never grants authority and does not block
an otherwise authorized stop.

## 6. Start and inspect

```powershell
& $Oac start --state-dir $State --target owned-agent:worker-v1 --timeout 15
& $Oac status --state-dir $State --target owned-agent:worker-v1
& $Oac audit-verify --state-dir $State
```

The supervisor configures the Job first, creates the child suspended with the
Job-list process attribute, proves membership, rechecks the executable hash,
then resumes it. A crash or supervision failure must be investigated before
restart.

## 7. Authorize and apply isolation

Requests expire quickly, bind the controller state root and immutable target
registration, and can be consumed only once.

```powershell
& $Oac request-new --state-dir $State --target owned-agent:worker-v1 `
  --ttl 120 --out .\isolate.unsigned.json
& $Oac request-sign --state-dir $State --request .\isolate.unsigned.json `
  --operator alice --private-key .\alice.pem --passphrase-env OAC_ALICE_PASSPHRASE `
  --out .\isolate.alice.json
& $Oac request-sign --state-dir $State --request .\isolate.alice.json `
  --operator bob --private-key .\bob.pem --passphrase-env OAC_BOB_PASSPHRASE `
  --out .\isolate.quorum.json
& $Oac request-verify --state-dir $State --request .\isolate.quorum.json
& $Oac apply-isolation --state-dir $State --request .\isolate.quorum.json --timeout 15
& $Oac audit-verify --state-dir $State
```

Never retry by editing an envelope. Create and sign a new request.

## 8. Export evidence to A11oy

```powershell
& $Oac context-export --state-dir $State --target owned-agent:worker-v1 `
  --out .\a11oy-context-projection.json
```

The export is read-only evidence. It does not cause an A11oy mutation or remote
call and cannot carry process-control authority.

## Recovery and rotation

- **Executable drift:** investigate the file replacement, register a new target
  ID, regenerate context, and collect a new request. Never rewrite the stored
  hash.
- **Lost operator key before sealing:** replace it, then seal. **After sealing,**
  this release has no in-place key-rotation protocol; create and qualify a new
  state root, then retire the old root under a documented migration.
- **Database or audit failure:** stop starts, preserve the complete state root,
  and investigate from a copy. Do not delete rows or repair hashes manually.
- **Supervisor crash:** run `status` and `audit-verify`; reconcile only through
  controller commands.
- **State backup:** back up the complete directory while the controller is idle.
  Restore only to a fixed local NTFS path owned by the operating identity, then
  rerun `doctor`, `audit-verify`, and `self-test`.
