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
#define AppVersion     "0.1.0"
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
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName} now"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Logs and config live outside {app}; leave the user's settings alone but take
; the logs, which are ours and can be large.
Type: filesandordirs; Name: "{userappdata}\{#AppName}\logs"

[Code]
// The app itself explains the virtual-cable requirement in plain language and
// offers to open the download page, so the installer deliberately says nothing
// about it -- one explanation, in the place where it can actually be acted on.

function InitializeSetup(): Boolean;
var
  Running: Boolean;
begin
  // Installing over a running tray app leaves a locked exe and a confusing
  // half-upgrade.
  Running := CheckForMutexes('Local\NoiseGate.SingleInstance');
  if Running then
  begin
    MsgBox('NoiseGate is currently running.' + #13#10#13#10 +
           'Please quit it first (right-click the tray icon, then "Quit NoiseGate") ' +
           'and run this installer again.', mbError, MB_OK);
    Result := False;
  end
  else
    Result := True;
end;
