# Database setup with Aksha

Use the `aksha initialize` command, not internal Python functions. It records
database locations and installation metadata in `hmm_databases.json`.
The package contains a database catalog, not the biological databases.

## List and install

```bash
aksha initialize --show_available
aksha initialize --show_installed
aksha initialize --hmms PFAM
aksha initialize --hmms PFAM,KOFAM
```

`--show_available` lists catalog entries not marked installed;
`--show_installed` lists installed locations and recorded versions where
available. Listing initializes the local catalog if needed but does not
download a database. Use explicit, case-sensitive names from that catalog.
Do not use a bare `--hmms` as an install-all shortcut.

The initial catalog is copied from the package. Newly added catalog entries
are merged into existing configuration without resetting installed flags.
Installations download upstream files and prepare pressed HMM indexes.
An already installed database is normally left alone.

## Configuration and storage locations

On Linux, the configuration file is normally
`~/.config/Aksha/hmm_databases.json`, or
`$XDG_CONFIG_HOME/Aksha/hmm_databases.json` when `XDG_CONFIG_HOME` is set.
Find the exact path without changing it:

```bash
python -c 'from pathlib import Path; from aksha.initialize import astra_config_dir; print(Path(astra_config_dir()) / "hmm_databases.json")'
```

Database storage is separate: the top-level `db_path` in that JSON controls
new download locations. The current packaged catalog retains the legacy
default `$HOME/.config/Astra`; changing the app name did not move databases.
Setting `XDG_CONFIG_HOME` relocates configuration, not this storage value.

To choose another location, first run `aksha initialize --show_available`,
then edit the existing `db_path` value to a writable absolute path, such as
`/data/hmm-databases`. Preserve the other fields and entries. Do this before
installing: changing `db_path` does not move existing files or update the
per-database `installation_dir` records.

## Existing databases and searches

The rename does not automatically migrate Astra's installed-database records.
You can use existing HMM files directly without redownloading or editing
installed flags:

```bash
aksha search --hmm_in /data/PFAM/Pfam-A.hmm --prot_in proteins.faa --cut_ga --outdir results
```

For a database installed through Aksha:

```bash
aksha search --installed_hmms PFAM --prot_in proteins.faa --cut_ga --outdir results
```

Choose thresholds appropriate to the database. `--cut_ga` requires profiles
with gathering cutoffs; it is not a universal option for arbitrary HMMs.

## Pressing and replacing an installation

Installers normally press downloaded HMMs automatically. To request pressing
of a named, registered installation:

```bash
aksha initialize --press --hmms PFAM
```

Specify the database explicitly. To fetch a replacement release, add
`--force` to an explicit install command, for example
`aksha initialize --hmms PFAM --force`.
**This deletes the existing recorded installation directory before downloading
the replacement.** Back up anything needed and check the location with
`--show_installed` first; a failed download does not restore the old files.

GPU searches additionally require the optional backend and a local manifest
for the installed, pressed database. See the [installation guide](release/INSTALL.md)
for those commands. Regenerate the manifest after replacing database files or
changing the native runtime.
