# Code signing

**Releases are currently unsigned.** An application to
[SignPath Foundation](https://signpath.io) — which signs qualifying open-source
projects for free — is pending, and their review is manual, so it may not be
granted.

The steps are written and sit commented out in
[`release.yml`](../.github/workflows/release.yml) under a "Release signing"
banner. Turning them on is three things: delete the `# ` prefixes, restore the
job-level `env` shown in the banner, and add the four settings below.

## What signing does not do

**It does not remove the SmartScreen warning.** Microsoft removed the instant
bypass that EV certificates used to give in 2024; every certificate type now
builds reputation the same way, per file, as downloads accumulate. A brand-new
release has none, so the first users still see a prompt.

Signing buys attribution — the publisher name in the UAC and SmartScreen
dialogs, and a chain that enterprise policy can allow — and reputation that
carries forward. It is not a switch that turns the warning off.

The only route with no warning at all is publishing an MSIX through the
Microsoft Store, where Microsoft re-signs the package.

## Enrolling

1. Apply at <https://signpath.io/foundation>. The project has to be open
   source, which this one is: MIT or Apache-2.0.
2. Once accepted, SignPath's dashboard gives an organization id, a project
   slug, and a signing-policy slug. Create two artifact configurations named
   exactly `executable` and `installer` — the workflow passes those names.
3. Add to this repository:

   | Kind | Name | Value |
   |---|---|---|
   | Secret | `SIGNPATH_API_TOKEN` | the API token |
   | Variable | `SIGNPATH_ORGANIZATION_ID` | organization id |
   | Variable | `SIGNPATH_PROJECT_SLUG` | project slug |
   | Variable | `SIGNPATH_POLICY_SLUG` | signing policy slug |

   Settings → Secrets and variables → Actions. The token is a secret; the other
   three are variables, so they show up in logs and are easier to correct.

4. Uncomment the signing steps and restore the job-level `env`.
5. Tag a release. `SIGNING_ENABLED` turns true as soon as the token exists, so
   a build with the steps uncommented but no token still releases unsigned
   rather than failing.

The publisher shown to users will be **SignPath Foundation**, not RoomMute —
that is the trade for a free certificate.

## Order matters

The executable is signed **before** the installer is built, because the
installer embeds it and Windows judges the extracted binary on its own
signature. Signing only the installer leaves the thing users actually run
unsigned, which is a common and invisible mistake.

Checksums are generated **after** signing: a signature changes the bytes, so
`SHA256SUMS.txt` would otherwise describe files nobody received.

Each signing step verifies the result with `Get-AuthenticodeSignature` and
fails the release if the status is not `Valid`. A release that claims to be
signed and is not is worse than an honest unsigned one.

## Using a certificate of your own instead

If you switch to Azure Artifact Signing (~$10/month, open to organizations in
the EU but to individuals only in the USA and Canada) or a traditional OV
certificate, the shape stays the same — replace the two
`signpath/github-action-submit-signing-request` steps with whatever invokes
`signtool`, keeping the order and the verification.

With a local `signtool`, Inno can sign the installer and its uninstaller
itself, which SignPath cannot do because it signs artifacts after the fact:

```
[Setup]
SignTool=signtool
SignedUninstaller=yes
```

```
ISCC /Ssigntool="signtool.exe sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 $f" ...
```

Always timestamp (`/tr`). Without it every signature stops verifying the day
the certificate expires, including on copies already downloaded.
