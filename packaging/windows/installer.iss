; Inno Setup script for the Windows installer.  Built by packaging/build.py:
;   iscc /DAppVersion=1.0.0 /DSourceDir=<dist\ElmerStudio> /DOutputDir=<dist> /DRepoDir=<repo> installer.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6B0E7C3A-3E5D-4D5B-9C1E-2A7D5E4F9B01}
AppName=Elmer Studio
AppVersion={#AppVersion}
AppPublisher=AltElmer
AppPublisherURL=https://github.com/AltElmer/ElmerStudio
DefaultDirName={autopf}\Elmer Studio
DefaultGroupName=Elmer Studio
LicenseFile={#RepoDir}\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename=ElmerStudio-{#AppVersion}-windows-x64-setup
SetupIconFile={#RepoDir}\packaging\icons\elmerstudio.ico
UninstallDisplayIcon={app}\ElmerStudio.exe
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
ChangesAssociations=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "assoc"; Description: "Open .esm model files with Elmer Studio"; GroupDescription: "File associations:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Elmer Studio"; Filename: "{app}\ElmerStudio.exe"
Name: "{group}\Uninstall Elmer Studio"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Elmer Studio"; Filename: "{app}\ElmerStudio.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.esm"; ValueType: string; ValueName: ""; ValueData: "ElmerStudio.Model"; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\ElmerStudio.Model"; ValueType: string; ValueName: ""; ValueData: "Elmer Studio model"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\ElmerStudio.Model\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\ElmerStudio.exe,0"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\ElmerStudio.Model\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\ElmerStudio.exe"" ""%1"""; Tasks: assoc

[Run]
Filename: "{app}\ElmerStudio.exe"; Description: "{cm:LaunchProgram,Elmer Studio}"; Flags: nowait postinstall skipifsilent
