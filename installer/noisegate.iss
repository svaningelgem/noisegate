; NoiseGate installer (Inno Setup 6).
;
; Per-user install on purpose: it needs no administrator rights, and it matches
; how the app registers start-with-Windows (HKCU\...\Run). A per-machine
; install would put the exe somewhere the user can't update and still write
; autostart per-user, which is the worst of both.
;
; Build:  ISCC.exe installer\noisegate.iss
; Expects cargo build --release --features onnx to have run first.

#define AppName        "NoiseGate"
; Overridable from the command line so CI can stamp the tag:
;   ISCC.exe /DAppVersion=1.2.3 installer\noisegate.iss
#ifndef AppVersion
  #define AppVersion   "0.1.0"
#endif
#define AppPublisher   "NoiseGate contributors"
#define AppURL         "https://github.com/Yashsomalkar/noisegate"
#define AppExe         "noisegate.exe"
#define SourceDir      "..\target\release"

[Setup]
AppId={{4E1F0B9A-9E3C-4B27-B0E5-7F3A2C6D8E10}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; No admin: everything lands under the user's profile.
PrivilegesRequired=lowest
UsePreviousPrivileges=no
OutputDir=..\dist
OutputBaseFilename=NoiseGate-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 10 1809 and up (WASAPI + the console-attach behaviour we rely on).
MinVersion=10.0.17763
LicenseFile=..\LICENSE-MIT
UninstallDisplayIcon={app}\{#AppExe}
; We stop the app ourselves in PrepareToInstall; Restart Manager can't help
; with a windowless tray app and its prompt would only add a step.
CloseApplications=no
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "startup"; Description: "Start {#AppName} when I sign in to Windows"; GroupDescription: "Startup:"
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\{#AppExe}";      DestDir: "{app}"; Flags: ignoreversion
; ONNX Runtime is MIT licensed and freely redistributable. Bundled rather than
; downloaded: it turns a required download into no download at all.
Source: "{#SourceDir}\onnxruntime.dll"; DestDir: "{app}"; Flags: ignoreversion
; DeepFilterNet3, dual MIT/Apache-2.0, so redistributable with attribution —
; models\NOTICE.md carries it. Without this the app installs and then quietly
; falls back to RNNoise, which does not remove background speech at all: the
; whole reason NoiseGate exists. The app picks it up automatically from
; alongside the executable.
;
; This is the export we build ourselves from the published checkpoint with
; scripts/export_dfn3.py, not a prebuilt file of unknown provenance.
Source: "..\models\dfn3_ours.tar.gz";  DestDir: "{app}"; Flags: ignoreversion
Source: "..\models\NOTICE.md";         DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md";                DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "..\LICENSE-MIT";              DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE-APACHE";           DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}";           Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";     Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; Exactly the value name and quoted format the app writes itself, so the tray's
; "Start with Windows" checkbox reflects what the installer did and the two
; never disagree.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\{#AppExe}"""; \
    Flags: uninsdeletevalue; Tasks: startup

[Run]
; Interactive: a checkbox on the finish page, ticked by default.
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName} now"; \
    Flags: nowait postinstall skipifsilent
; Silent upgrade: bring it back only if we were the ones who stopped it, so an
; unattended update is invisible rather than leaving someone without a mic.
Filename: "{app}\{#AppExe}"; Flags: nowait; Check: ShouldRelaunchSilently

[UninstallDelete]
; Logs and config live outside {app}; leave the user's settings alone but take
; the logs, which are ours and can be large.
Type: filesandordirs; Name: "{userappdata}\{#AppName}\logs"

[Code]
// The app itself explains the virtual-cable requirement in plain language and
// offers to open the download page, so the installer deliberately says nothing
// about it -- one explanation, in the place where it can actually be acted on.

// Stop a running NoiseGate before overwriting its files.
//
// Restart Manager is the usual mechanism, but it works by asking windows to
// close, and this is a tray app with no window -- there is nothing to send
// WM_CLOSE to. So terminate it outright. Nothing is lost: every setting is
// written to config.toml the moment it changes, and Windows releases the audio
// endpoints when the process exits.
//
// Run unconditionally rather than only when the tray mutex is held, because
// `--record` and `--denoise` instances lock the same exe without taking it.
var
  StoppedRunningInstance: Boolean;

procedure StopRunningInstance();
var
  ResultCode: Integer;
begin
  if CheckForMutexes('Local\NoiseGate.SingleInstance') then
    StoppedRunningInstance := True;
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/IM noisegate.exe /F',
       '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  // taskkill returns before the handles are actually released.
  if ResultCode = 0 then
    Sleep(1000);
end;

function ShouldRelaunchSilently(): Boolean;
begin
  Result := StoppedRunningInstance and WizardSilent();
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  StopRunningInstance();
  if CheckForMutexes('Local\NoiseGate.SingleInstance') then
    Result := 'NoiseGate is still running and could not be closed automatically.' + #13#10#13#10 +
              'Please quit it from the tray icon (right-click, "Quit NoiseGate") and run ' +
              'this installer again.'
  else
    Result := '';
end;

// Same courtesy on the way out, so uninstalling doesn't leave the tray icon
// behind pointing at files that no longer exist.
function InitializeUninstall(): Boolean;
begin
  StopRunningInstance();
  Result := True;
end;
